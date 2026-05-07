from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn
from transformers import Gemma3TextConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import GeluAndMul
from vllm.model_executor.layers.attention import (
    Attention,
    EncoderOnlyAttention,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
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
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (
    AutoWeightsLoader,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)

# from transformers.models import gemma3

import re
import copy

def _extract_layer_idx(layer_name: str) -> int | None:
    m = re.search(r'layers\.(\d+)', layer_name)
    return int(m.group(1)) if m else None


def _get_step_info_from_metadata():
    fwd_ctx = get_forward_context()
    attn_metadata = fwd_ctx.attn_metadata
    if not isinstance(attn_metadata, dict):
        return None, 0, None, None, False
    for meta in attn_metadata.values():
        if (hasattr(meta, 'query_start_loc') and
            hasattr(meta, 'seq_lens') and
            hasattr(meta, 'block_table')):
            cu_seqlens  = meta.query_start_loc
            max_seq_len = int(meta.max_query_len)
            query_lens  = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)
            kv_seq_lens = meta.seq_lens
            is_prefill  = max_seq_len > 1
            return cu_seqlens, max_seq_len, query_lens, kv_seq_lens, is_prefill
    return None, 0, None, None, False




def _update_metadata_after_drop(
    token_mask:        torch.Tensor,
    new_cu_seqlens:    torch.Tensor,
    new_max_seq_len:   int,
    current_layer_idx: int,
    kv_block_size:     int = 16,
    drop_layer_indices=None
):
    fwd_ctx = get_forward_context()
    device  = token_mask.device

    new_cu_seqlens   = new_cu_seqlens.to(torch.int32)
    new_seq_lens     = (new_cu_seqlens[1:] - new_cu_seqlens[:-1]).to(torch.int32)
    new_total_tokens = int(token_mask.sum().item())
    num_reqs         = new_seq_lens.shape[0]


    query_lens = getattr(fwd_ctx, '_query_lens_this_step', None)
    if query_lens is not None:
        is_decode = (query_lens[:num_reqs] == 1)
    else:
        is_decode = (new_seq_lens == 1)


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

    layer_block_table: dict[int, torch.Tensor] = {}
    default_block_table = None
    if isinstance(fwd_ctx.attn_metadata, dict):
        for lname, meta in fwd_ctx.attn_metadata.items():
            li = _extract_layer_idx(lname)
            if li is not None and hasattr(meta, 'block_table') \
                    and meta.block_table is not None:
                layer_block_table[li] = meta.block_table
                if default_block_table is None:
                    default_block_table = meta.block_table

    req_idx = torch.repeat_interleave(
        torch.arange(num_reqs, device=device, dtype=torch.int64),
        new_seq_lens.long()
    )
    tok_idx = torch.arange(new_total_tokens, device=device, dtype=torch.int64)
    intra   = tok_idx - new_cu_seqlens[req_idx].long()

    def _compute_slot_mapping(block_table):
        blk_ids  = intra // kv_block_size
        offsets  = intra % kv_block_size
        phys_blk = block_table[req_idx, blk_ids]
        sm = (phys_blk * kv_block_size + offsets).to(torch.int64)

        if is_decode.any() and kv_seq_lens is not None:
            dec_req_idx = torch.where(is_decode)[0]
            dec_lpos    = (mixed_seq_lens[is_decode] - 1).long()
            dec_blk     = dec_lpos // kv_block_size
            dec_off     = dec_lpos % kv_block_size
            dec_pblk    = block_table[dec_req_idx, dec_blk]
            dec_slots   = (dec_pblk * kv_block_size + dec_off).to(torch.int64)
            dec_tok_pos = new_cu_seqlens[dec_req_idx].long()
            sm[dec_tok_pos] = dec_slots
        return sm

    _sm_cache: dict[int, torch.Tensor] = {}

    def _get_slot_mapping(li: int) -> torch.Tensor | None:
        bt = layer_block_table.get(li, default_block_table)
        if bt is None:
            return None
        bt_id = id(bt)
        if bt_id not in _sm_cache:
            _sm_cache[bt_id] = _compute_slot_mapping(bt)
        return _sm_cache[bt_id]

    if isinstance(fwd_ctx.slot_mapping, dict):
        for layer_name, sm in list(fwd_ctx.slot_mapping.items()):
            li = _extract_layer_idx(layer_name)  
            if li is None or li <= current_layer_idx:
                continue
            new_sm = _get_slot_mapping(li)
            if new_sm is not None:
                fwd_ctx.slot_mapping[layer_name] = new_sm
            elif sm.shape[0] == token_mask.shape[0]:
                fwd_ctx.slot_mapping[layer_name] = sm[token_mask]

    if isinstance(fwd_ctx.attn_metadata, dict):
        cloned: dict[int, object] = {}

        for layer_name in list(fwd_ctx.attn_metadata.keys()):
            li = _extract_layer_idx(layer_name)  
            if li is None or li <= current_layer_idx:
                continue

            old_meta = fwd_ctx.attn_metadata[layer_name]
            old_id   = id(old_meta)

            if old_id not in cloned:
                new_meta = copy.copy(old_meta)
                new_meta.seq_lens          = mixed_seq_lens
                new_meta.query_start_loc   = new_cu_seqlens
                new_meta.max_query_len     = new_max_seq_len
                new_meta.max_seq_len       = mixed_max_seq_len
                new_meta.num_actual_tokens = new_total_tokens
                new_sm = _get_slot_mapping(li)
                if new_sm is not None:
                    new_meta.slot_mapping = new_sm
                cloned[old_id] = new_meta

            fwd_ctx.attn_metadata[layer_name] = cloned[old_id]


class Gemma3MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_activation: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_activation != "gelu_pytorch_tanh":
            raise ValueError(
                "Gemma3 uses `gelu_pytorch_tanh` as the hidden activation "
                "function. Please set `hidden_act` and `hidden_activation` to "
                "`gelu_pytorch_tanh`."
            )
        self.act_fn = GeluAndMul(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Gemma3Attention(nn.Module):
    def __init__(
        self,
        config: Gemma3TextConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        attn_logits_soft_cap: float | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
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
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = config.query_pre_attn_scalar**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        layer_idx = extract_layer_index(prefix)
        self.layer_idx = layer_idx
        layer_type = config.layer_types[layer_idx]
        self.is_sliding = layer_type == "sliding_attention"
        sliding_window = config.sliding_window if self.is_sliding else None

        # Initialize the rotary embedding.
        if layer_type in config.rope_parameters:
            # Transformers v5 rope config.
            rope_parameters = config.rope_parameters[layer_type]
        else:
            # Transformers v4 rope config.
            # Global attention. Use the values in config.json.
            rope_parameters = config.rope_parameters
            # Local attention. Override the values in config.json.
            if self.is_sliding:
                rope_parameters = dict(
                    rope_type="default", rope_theta=config.rope_local_base_freq
                )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=True,
        )

        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

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
            attn_type=attn_type,
            logits_soft_cap=attn_logits_soft_cap,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        return_qk: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # logger.info(type(self.attn.impl))
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_norm(q)
        q = q.flatten(-2, -1)
        k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
        k = self.k_norm(k)
        k = k.flatten(-2, -1)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        if return_qk:
            return output, q, k
        return output


class Gemma3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Gemma3TextConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Gemma3Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_logits_soft_cap=None,
            prefix=f"{prefix}.self_attn",
        )
        self.hidden_size = config.hidden_size
        self.mlp = Gemma3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        do_token_drop: bool = False,
        drop_config: dict | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seq_len: int = 0,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if do_token_drop and drop_config is not None:
            hidden_states, q, k = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                return_qk=True,
                **kwargs,
            )

            token_mask, new_max_seq_len, new_cu_seqlens = topselectionvarlen(
                q, k, self.self_attn.head_dim,
                cu_seqlens, max_seq_len,
                drop_config['block_size'],
                drop_config['attention_sink'],
                drop_config['last_q'],
                drop_config['p'],
            )

            kept_indices = token_mask.nonzero(as_tuple=True)[0]
            n_kept = kept_indices.shape[0]



            hidden_states[:n_kept] = hidden_states[kept_indices]
            residual[:n_kept]      = residual[kept_indices]
            positions[:n_kept]     = positions[kept_indices]

            hidden_states = hidden_states[:n_kept]
            residual      = residual[:n_kept]
            positions     = positions[:n_kept]

            _update_metadata_after_drop(
                token_mask, new_cu_seqlens, new_max_seq_len,
                current_layer_idx=self.self_attn.layer_idx,
                kv_block_size=drop_config.get('kv_block_size', 16),
                drop_layer_indices=drop_config.get('drop_layer_indices', None), 
            )
        else:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                **kwargs,
            )

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, residual = self.pre_feedforward_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)

        if do_token_drop and drop_config is not None:
            return hidden_states, residual, positions, token_mask, new_cu_seqlens, new_max_seq_len

        return hidden_states, residual


@support_torch_compile
class Gemma3Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Gemma3DecoderLayer(
                config, cache_config, quant_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


        normalizer = self.config.hidden_size**0.5
        self.register_buffer("normalizer", torch.tensor(normalizer), persistent=False)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        self.drop_layer_indices = [
            29, 35, 41
        ]

        self.drop_config = {
            'block_size': 64,
            'attention_sink': 128,
            'last_q': 128,
            'p': 0.975,
            'kv_block_size': vllm_config.cache_config.block_size,
            'drop_layer_indices': set(self.drop_layer_indices), 
        }
        self.per_layer_seq_lens: dict[str, list[tuple[int, int]]] | None = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # NOTE(woosuk): Only apply the normalizer to the output of
        # vocab embedding. Don't apply it to the vision embedding.
        return self.embed_tokens(input_ids) * self.normalizer

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
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

        fwd_ctx = get_forward_context()
        fwd_ctx._layer_types = getattr(self.config, 'layer_types', None)
        fwd_ctx._sliding_window_size = getattr(self.config, 'sliding_window', 0)



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

            self.per_layer_seq_lens = {}  # 每次 prefill step 重置

            if req_ids_snapshot is not None:
                num_reqs = query_lens.shape[0]
                for i in range(num_reqs):
                    q_len = int(query_lens[i].item())
                    if q_len > 1:
                        rid = req_ids_snapshot[i]

                        kv_len_after_prefill = int(kv_seq_lens[i].item())
                        self.per_layer_seq_lens[rid] = [(0, kv_len_after_prefill)]


        for layer in islice(self.layers, self.start_layer, self.end_layer):


            actual_layer_idx = layer.self_attn.layer_idx

            should_drop = (
                is_prefill_step
                and actual_layer_idx in self.drop_layer_indices
                and cu_seqlens is not None
                and hidden_states.shape[0] > 1024
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
                    **kwargs
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
                    positions,
                    hidden_states,
                    residual,
                    **kwargs,
                )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:

            if (
                self.quant_config
                and self.quant_config.get_name() == "gguf"
                and name.endswith("norm.weight")
            ):
                loaded_weight -= 1

            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                # Loading kv cache scales for compressed-tensors quantization
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = loaded_weight[0]
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue

            # Check if this is a scale parameter that needs remapping first
            if name.endswith((".k_scale", ".v_scale", ".q_scale", ".prob_scale")):
                # Try to remap the scale name first
                remapped_name = maybe_remap_kv_scale_name(name, params_dict)
                if remapped_name is not None and remapped_name in params_dict:
                    # Successfully remapped, use the remapped name
                    param = params_dict[remapped_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(remapped_name)
                    continue
                # If remapping failed, continue with normal processing

            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
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
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


class Gemma3ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = Gemma3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

        self.logits_processor = LogitsProcessor(
            config.vocab_size, soft_cap=config.final_logit_softcapping
        )
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )
        return hidden_states

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
