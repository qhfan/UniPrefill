"""Inference-only LLaMA model compatible with HuggingFace weights."""

from collections.abc import Iterable
from itertools import islice

from numpy import dtype
import torch
from torch import nn
from transformers import LlamaConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import (
    Attention,
    EncoderOnlyAttention,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)

from vllm.model_executor.layers.fused_top_p_selection_tp_pd import topselectionvarlen
#from vllm.model_executor.layers.drop_kv_cache_batch_1 import TopPSelectionVarLen

from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)


from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType

from .adapters import as_embedding_model, as_seq_cls_model
from .interfaces import (
    SupportsEagle,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.config import CUDAGraphMode



logger = init_logger(__name__)



def _get_step_info_from_metadata():
    fwd_ctx = get_forward_context()
    attn_metadata = fwd_ctx.attn_metadata

    if isinstance(attn_metadata, dict):
        for meta in attn_metadata.values():
            if hasattr(meta, 'query_start_loc') and hasattr(meta, 'seq_lens'):
                cu_seqlens  = meta.query_start_loc          # [num_reqs+1], int32
                max_seq_len = int(meta.max_query_len)
                query_lens  = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)
                kv_seq_lens = meta.seq_lens                 # [num_reqs], int32
                is_prefill  = max_seq_len > 1
                return cu_seqlens, max_seq_len, query_lens, kv_seq_lens, is_prefill

    return None, 0, None, None, False


def _update_metadata_after_drop(
    token_mask:        torch.Tensor,   # [total_tokens_before_drop], bool
    new_cu_seqlens:    torch.Tensor,   # [num_reqs+1], int32
    new_max_seq_len:   int,
    current_layer_idx: int,
    kv_block_size:     int = 16,
):
    """
    """
    fwd_ctx = get_forward_context()
    device  = token_mask.device

    new_cu_seqlens  = new_cu_seqlens.to(torch.int32)
    new_seq_lens    = (new_cu_seqlens[1:] - new_cu_seqlens[:-1]).to(torch.int32)
    new_total_tokens = int(token_mask.sum().item())
    num_reqs        = new_seq_lens.shape[0]

    query_lens = getattr(fwd_ctx, '_query_lens_this_step', None)
    if query_lens is not None:
        is_decode = (query_lens[:num_reqs] == 1)   # decode req: query_len == 1
    else:
        is_decode = (new_seq_lens == 1)             # fallback

    kv_seq_lens = getattr(fwd_ctx, '_kv_seq_lens_this_step', None)
    if kv_seq_lens is None:
        if isinstance(fwd_ctx.attn_metadata, dict):
            kv_seq_lens = next(iter(fwd_ctx.attn_metadata.values())).seq_lens


    fwd_ctx.token_drop_applied = True
    fwd_ctx.new_cu_seqlens     = new_cu_seqlens
    fwd_ctx.new_seq_lens       = new_seq_lens


    mixed_seq_lens = new_seq_lens.clone()
    if kv_seq_lens is not None and is_decode.any():

        _is_mixed = getattr(fwd_ctx, '_is_mixed_or_prefill_step', False)
        if not _is_mixed:

            mixed_seq_lens[is_decode] = kv_seq_lens[is_decode].to(torch.int32)
    mixed_max_seq_len = int(mixed_seq_lens.max().item())


    block_table = None
    if isinstance(fwd_ctx.attn_metadata, dict):
        for meta in fwd_ctx.attn_metadata.values():
            if hasattr(meta, 'block_table'):
                block_table = meta.block_table
                break


    new_slot_mapping = None
    if block_table is not None:

        req_idx  = torch.repeat_interleave(
            torch.arange(num_reqs, device=device, dtype=torch.int64),
            new_seq_lens.long()
        )
        tok_idx  = torch.arange(new_total_tokens, device=device, dtype=torch.int64)
        intra    = tok_idx - new_cu_seqlens[req_idx].long()  
        blk_ids  = intra // kv_block_size
        offsets  = intra % kv_block_size
        phys_blk = block_table[req_idx, blk_ids]
        new_slot_mapping = (phys_blk * kv_block_size + offsets).to(torch.int64)


        if is_decode.any() and kv_seq_lens is not None:
            dec_req_idx = torch.where(is_decode)[0]             
            dec_lpos    = (mixed_seq_lens[is_decode] - 1).long() 
            dec_blk     = dec_lpos // kv_block_size
            dec_off     = dec_lpos % kv_block_size
            dec_pblk    = block_table[dec_req_idx, dec_blk]
            dec_slots   = (dec_pblk * kv_block_size + dec_off).to(torch.int64)

            dec_tok_pos = new_cu_seqlens[dec_req_idx].long()
            new_slot_mapping[dec_tok_pos] = dec_slots


    if isinstance(fwd_ctx.slot_mapping, dict):
        for layer_name, sm in list(fwd_ctx.slot_mapping.items()):
            try:
                layer_idx = int(layer_name.split('.')[2])
            except (IndexError, ValueError):
                continue
            if layer_idx > current_layer_idx:
                if new_slot_mapping is not None:
                    fwd_ctx.slot_mapping[layer_name] = new_slot_mapping
                elif sm.shape[0] == token_mask.shape[0]:
                    fwd_ctx.slot_mapping[layer_name] = sm[token_mask]

    if isinstance(fwd_ctx.attn_metadata, dict):
        for layer_name, meta in fwd_ctx.attn_metadata.items():
            try:
                layer_idx = int(layer_name.split('.')[2])
            except (IndexError, ValueError):
                continue
            if layer_idx > current_layer_idx:
                if hasattr(meta, 'seq_lens'):
                    meta.seq_lens = mixed_seq_lens
                if hasattr(meta, 'query_start_loc'):
                    meta.query_start_loc = new_cu_seqlens
                if hasattr(meta, 'max_query_len'):
                    meta.max_query_len = new_max_seq_len
                if hasattr(meta, 'max_seq_len'):
                    meta.max_seq_len = mixed_max_seq_len
                if hasattr(meta, 'num_actual_tokens'):
                    meta.num_actual_tokens = new_total_tokens
                if hasattr(meta, 'slot_mapping') and new_slot_mapping is not None:
                    meta.slot_mapping = new_slot_mapping




class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        prefix: str = "",
        reduce_results: bool = True,
        disable_tp: bool = False,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            disable_tp=disable_tp,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=disable_tp,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class LlamaAttention(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        head_dim = getattr(config, "head_dim", None)
        self.head_dim = head_dim or self.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self._init_rotary_emb(config, quant_config=quant_config)

        sliding_window = None
        if layer_types := getattr(config, "layer_types", None):
            # Fix for Eagle3 compatibility:
            # for draft models, subtract target layer count
            # to get draft-relative layer index starting from 0
            if hasattr(config, "target_layer_count"):
                # This is a draft model,
                # adjust layer_idx to be relative to draft layers
                effective_layer_idx = layer_idx - config.target_layer_count
            else:
                # This is a target model, use layer_idx directly
                effective_layer_idx = layer_idx
            assert effective_layer_idx < len(layer_types), (
                f"effective_layer_idx: {effective_layer_idx} "
                f"is out of bounds for layer_types: {layer_types}"
            )

            is_sliding = layer_types[effective_layer_idx] == "sliding_attention"
            if is_sliding:
                sliding_window = config.sliding_window

        attn_cls = (
            EncoderOnlyAttention
            if attn_type == AttentionType.ENCODER_ONLY
            else Attention
        )

        self.attn = attn_cls(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            attn_type=attn_type,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        return_qk: bool = False   # new
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        if return_qk:
            return output, q, k
        return output

    def _init_rotary_emb(
        self,
        config: LlamaConfig,
        quant_config: QuantizationConfig | None,
    ) -> None:
        is_neox_style = True
        is_gguf = quant_config and quant_config.get_name() == "gguf"
        if is_gguf and config.model_type == "llama":
            is_neox_style = False

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=getattr(config, "rope_parameters", None),
            is_neox_style=is_neox_style,
        )


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        config: LlamaConfig | None = None,
        attn_layer_type: type[nn.Module] = LlamaAttention,
    ) -> None:
        super().__init__()

        config = config or vllm_config.model_config.hf_config
        self.config = config
        cache_config = vllm_config.cache_config
        quant_config = self.get_quant_config(vllm_config)

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False
        )
        bias_o_proj = attention_bias
        # support internlm/internlm3-8b with qkv_bias
        if hasattr(config, "qkv_bias"):
            attention_bias = config.qkv_bias

        # By default, Llama uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. parasail-ai/GritLM-7B-vllm)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = attn_layer_type(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            bias_o_proj=bias_o_proj,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        do_token_drop: bool = False,        # new
        drop_config: dict | None = None,    # new
        cu_seqlens: torch.Tensor | None = None,   # new
        max_seq_len: int = 0,               # new
        actual_layer_idx: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if do_token_drop:
            # get (q, k) for importance
            hidden_states, q, k = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                return_qk=True,
            )
        else:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
            )

        if do_token_drop and drop_config is not None:
            # num_tokens = q.shape[0]
            # q_3d = q.reshape(num_tokens, self.self_attn.num_heads, self.self_attn.head_dim)
            # k_3d = k.reshape(num_tokens, self.self_attn.num_kv_heads, self.self_attn.head_dim)

            token_mask, new_max_seq_len, new_cu_seqlens = topselectionvarlen(
                q, k, self.self_attn.head_dim,
                cu_seqlens,
                max_seq_len,
                drop_config['block_size'],
                drop_config['attention_sink'],
                drop_config['last_q'],
                drop_config['p'],
            )



            kept_indices = token_mask.nonzero(as_tuple=True)[0]
            n_kept = kept_indices.shape[0]

            hidden_states[:n_kept] = hidden_states[kept_indices]
            residual[:n_kept] = residual[kept_indices]
            positions[:n_kept] = positions[kept_indices]

            hidden_states = hidden_states[:n_kept]
            residual = residual[:n_kept]
            positions = positions[:n_kept]


            _update_metadata_after_drop(
                token_mask, new_cu_seqlens, new_max_seq_len,
                current_layer_idx=actual_layer_idx,
                kv_block_size=drop_config.get('kv_block_size', 16),  
            )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        if do_token_drop and drop_config is not None:
            return hidden_states, residual, positions, token_mask, new_cu_seqlens, new_max_seq_len

        return hidden_states, residual

    def get_quant_config(self, vllm_config: VllmConfig) -> QuantizationConfig | None:
        """Get quantization config for this layer. Override in subclasses."""
        return vllm_config.quant_config


def llama_model_invariants(
    input_ids, positions, intermediate_tensors=None, inputs_embeds=None
):
    """Shape invariants for Llama model compilation, those are translated to
    runtime assertions for unbacked dynamic shapes and are compiled away for
    backed"""
    if input_ids is not None:
        torch._check(positions.size()[0] == input_ids.size()[0])


@support_torch_compile(
    # TODO[#32068]: Investigate recompilation
    # mark_unbacked_dims={"input_ids": 0},
    shape_invariants=llama_model_invariants
)
class LlamaModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = LlamaDecoderLayer,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: layer_type(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.aux_hidden_state_layers = tuple[int, ...]()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        self.drop_layer_indices = [
            7, 11, 15, 19, 23, 27
        ]
        self.drop_config = {
            'block_size': 64,
            'attention_sink': 128,
            'last_q': 128,
            'p': 0.99,
            'kv_block_size': vllm_config.cache_config.block_size,
        }
        self.per_layer_seq_lens: dict[str, list[tuple[int, int]]] | None = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **extra_layer_kwargs,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]


        cu_seqlens, max_seq_len, query_lens, kv_seq_lens, is_prefill_step = \
            _get_step_info_from_metadata()

        cu_seqlens  = cu_seqlens   # query_start_loc，[num_reqs+1]
        max_seq_len = max_seq_len  # max_query_len，int

        if is_prefill_step and len(self.drop_layer_indices) > 0 \
                and cu_seqlens is not None:
            fwd_ctx = get_forward_context()

            fwd_ctx._query_lens_this_step   = query_lens    # [num_reqs], int32
            fwd_ctx._kv_seq_lens_this_step  = kv_seq_lens   # [num_reqs], int32


            req_ids_snapshot: list[str] | None = getattr(fwd_ctx, '_req_ids_snapshot', None)

            self.per_layer_seq_lens = {}  

            if req_ids_snapshot is not None:
                num_reqs = query_lens.shape[0]
                for i in range(num_reqs):
                    q_len = int(query_lens[i].item())
                    if q_len > 1:
                        rid = req_ids_snapshot[i]

                        kv_len_after_prefill = int(kv_seq_lens[i].item())
                        self.per_layer_seq_lens[rid] = [(0, kv_len_after_prefill)]

        aux_hidden_states = []
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            if idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states + residual)

            actual_layer_idx = self.start_layer + idx
            should_drop = (
                is_prefill_step
                and actual_layer_idx in self.drop_layer_indices
                and cu_seqlens is not None
                and hidden_states.shape[0] > (
                    self.drop_config['attention_sink'] + self.drop_config['last_q'])
            )


            if should_drop:
                


                hidden_states, residual, positions, token_mask, new_cu_seqlens, new_max_seq_len = layer(
                    positions, hidden_states, residual,
                    do_token_drop=True,
                    drop_config=self.drop_config,
                    cu_seqlens=cu_seqlens,
                    max_seq_len=max_seq_len,
                    actual_layer_idx=actual_layer_idx
                )
                
                cu_seqlens = new_cu_seqlens
                max_seq_len = new_max_seq_len
                


                fwd_ctx = get_forward_context()
                req_ids_snapshot = getattr(fwd_ctx, '_req_ids_snapshot', None)


                

                if req_ids_snapshot is not None and self.per_layer_seq_lens is not None:

                    new_seq_lens_after_drop = (
                        new_cu_seqlens[1:] - new_cu_seqlens[:-1]
                    ).to(torch.int32)
                    num_reqs = query_lens.shape[0]

                    for i in range(num_reqs):
                        q_len = int(query_lens[i].item())
                        if q_len > 1:  
                            rid = req_ids_snapshot[i]
                            dropped_len = int(new_seq_lens_after_drop[i].item())
                            if rid in self.per_layer_seq_lens:
                                self.per_layer_seq_lens[rid].append(
                                    (actual_layer_idx, dropped_len)
                                )


            else:
                    
                hidden_states, residual = layer(
                    positions, hidden_states, residual,
                    **extra_layer_kwargs,
                )


        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        
        

        hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            if "scale" in name or "zero_point" in name:
                # Remapping the name of FP8 kv-scale or zero point.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class LlamaForCausalLM(
    nn.Module, SupportsLoRA, SupportsPP, SupportsEagle, SupportsEagle3
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # LoRA specific attributes
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = LlamaDecoderLayer,
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.model = self._init_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            layer_type=layer_type,
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(
                config.vocab_size, scale=logit_scale
            )
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        """Override to return default layers for Llama

        Note: The GPU model runner will override this with layers from
        the speculative config if available, providing dynamic configuration.
        """
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def _init_model(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = LlamaDecoderLayer,
    ):
        return LlamaModel(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        model_output = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits



    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)


class LlamaBidirectionalForSequenceClassification(as_seq_cls_model(LlamaForCausalLM)):
    # This class sets the correct attention type and pooling type
    # through LlamaBidirectionalConfig.
    pass


class LlamaBidirectionalModel(as_embedding_model(LlamaForCausalLM)):
    # This class sets the correct attention type and pooling type
    # through LlamaBidirectionalConfig.
    pass
