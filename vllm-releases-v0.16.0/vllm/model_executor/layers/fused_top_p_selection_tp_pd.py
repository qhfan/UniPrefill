import torch
import triton
import triton.language as tl
from fla.utils import autocast_custom_fwd, contiguous
from vllm.distributed import get_tp_group, all_gather
from vllm.logger import init_logger

logger = init_logger(__name__)



def get_part_gemm_configs():
    configs = []
    warps = [4, 8]
    stages = [2, 3, 4]
    q_tiles = [32, 64, 128]
    k_tiles = [32, 64, 128]
    for w in warps:
        for s in stages:
            for qt in q_tiles:
                for kt in k_tiles:
                    configs.append(
                        triton.Config(
                            {
                                'Q_TILE_SIZE': qt,
                                'K_TILE_SIZE': kt
                            },
                            num_warps=w,
                            num_stages=s
                        )
                    )
    return configs

@triton.autotune(
    configs=get_part_gemm_configs(),
    key = ['num_q_heads', 'num_k_heads', 'LAST_Q'],
    prune_configs_by={
        'early_config_prune': lambda configs, named_args, **kwargs: [
            c for c in configs 
            if c.kwargs['Q_TILE_SIZE'] <= named_args['LAST_Q'] 
        ]
    }
)
@triton.jit
def fused_part_gemm_kernel(
    Q_ptr, K_ptr, O_ptr,
    stride_qm, stride_qh, stride_qd,
    stride_kn, stride_kh, stride_kd,
    stride_om, stride_on, stride_oh,
    cu_seqlens, max_seqlen,
    num_q_heads, num_k_heads, 
    LAST_Q: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr,
    D_HEAD: tl.constexpr
):
    key_tile_index = tl.program_id(0).to(tl.int64)
    offset_zh = tl.program_id(1).to(tl.int64)

    offset_batch = offset_zh // num_q_heads
    group_size = num_q_heads // num_k_heads

    offset_q_heads = offset_zh % num_q_heads
    offset_k_heads = offset_q_heads // group_size

    cu_seqlens_k_start = tl.load(cu_seqlens + offset_batch)
    cu_seqlens_k_end = tl.load(cu_seqlens + offset_batch + 1)
    key_len = cu_seqlens_k_end - cu_seqlens_k_start

    if key_tile_index * K_TILE_SIZE > key_len:
        return
    
    cu_seqlens_q_start = cu_seqlens_k_end - LAST_Q
    cu_seqlens_q_end = cu_seqlens_k_end

    Q_base_ptr = Q_ptr + offset_q_heads * stride_qh + cu_seqlens_q_start * stride_qm
    K_base_ptr = K_ptr + offset_k_heads * stride_kh + cu_seqlens_k_start * stride_kn
    O_base_ptr = O_ptr + offset_q_heads * stride_oh + cu_seqlens_k_start * stride_on

    offset_k = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
    offset_dim = tl.arange(0, D_HEAD)

    k = tl.load(
        K_base_ptr + offset_k[:, None] * stride_kn + offset_dim[None, :] * stride_kd,
        mask=(offset_k[:, None] < key_len) & (offset_dim[None, :] < D_HEAD),
        other=0.0
    )

    
    for i in range(0, LAST_Q, Q_TILE_SIZE):
        offset_q = i + tl.arange(0, Q_TILE_SIZE)
        q = tl.load(
            Q_base_ptr + offset_q[:, None] * stride_qm + offset_dim[None, :] * stride_qd,
            mask=(offset_q[:, None] < LAST_Q) & (offset_dim[None, :] < D_HEAD),
            other=0.0
        )

        part_sum = tl.dot(q, tl.trans(k))

        q_seq_pos = (key_len - LAST_Q) + offset_q
        causal_mask = (q_seq_pos[:, None] >= offset_k[None, :])
        part_sum = tl.where(causal_mask, part_sum, float('-inf'))

        tl.store(
            O_base_ptr + offset_q[:, None] * stride_om + offset_k[None, :] * stride_on,
            part_sum.to(O_ptr.dtype.element_ty),
            mask=(offset_q[:, None] < LAST_Q) & (offset_k[None, :] < key_len)
        )

def get_part_softmax_configs():
    configs = []
    warps = [4, 8]
    stages = [2, 3, 4]
    k_tiles = [32, 64, 128]
    for w in warps:
        for s in stages:
            for kt in k_tiles:
                configs.append(
                    triton.Config(
                        {
                            'K_TILE_SIZE': kt
                        },
                        num_warps=w,
                        num_stages=s
                    )
                )
    return configs

@triton.autotune(
    configs=get_part_softmax_configs(),
    key = ['num_q_heads', 'num_k_heads', 'LAST_Q'],
)
@triton.jit
def fused_block_softmax_kernel(
    S_ptr, O_ptr, scale,
    stride_sm, stride_sn, stride_sh,
    stride_on, stride_oh,
    cu_seqlens, max_seqlen,
    num_q_heads, num_k_heads,
    LAST_Q: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr
):
    '''
    assert Q_TILE_SIZE == LAST_Q
    parallel in the batch and q_heads dimension
    '''
    offset_zh = tl.program_id(0).to(tl.int64)

    offset_batch = offset_zh // num_q_heads
    group_size = num_q_heads // num_k_heads

    offset_q_heads = offset_zh % num_q_heads
    offset_k_heads = offset_q_heads // group_size

    cu_seqlens_k_start = tl.load(cu_seqlens + offset_batch)
    cu_seqlens_k_end = tl.load(cu_seqlens + offset_batch + 1)
    key_len = cu_seqlens_k_end - cu_seqlens_k_start

    S_base_ptr = S_ptr + offset_q_heads * stride_sh + cu_seqlens_k_start * stride_sn
    O_base_ptr = O_ptr + offset_q_heads * stride_oh + cu_seqlens_k_start * stride_on


    sm_scale = scale * 1.4426950408889634 # log2(e)

    m = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)
    l = tl.full((Q_TILE_SIZE,), 0.0, dtype=tl.float32)

    offset_q = tl.arange(0, Q_TILE_SIZE)

    for j in range(0, key_len, K_TILE_SIZE):
        offset_k = j + tl.arange(0, K_TILE_SIZE)
        s = tl.load(
            S_base_ptr + offset_q[:, None] * stride_sm + offset_k[None, :] * stride_sn,
            mask=(offset_q[:, None] < LAST_Q) & (offset_k[None, :] < key_len),
            other=float('-inf')
        )
        s = s * sm_scale
        m_new = tl.maximum(m, tl.max(s, axis=1))
        s -= m_new[:, None]
        p = tl.exp2(s)
        l_j = tl.sum(p, axis=1)
        alpha = tl.exp2(m - m_new)
        alpha_mask = (alpha != alpha)
        alpha = tl.where(alpha_mask, 1.0, alpha)
        l = l * alpha + l_j
        m = m_new

    l = tl.where(l > 0, l, 1.0)

    for j in range(0, key_len, K_TILE_SIZE):
        offset_k = j + tl.arange(0, K_TILE_SIZE)
        s = tl.load(
            S_base_ptr + offset_q[:, None] * stride_sm + offset_k[None, :] * stride_sn,
            mask=(offset_q[:, None] < LAST_Q) & (offset_k[None, :] < key_len),
            other=float('-inf')
        )
        s = s * sm_scale
        s -= m[:, None]
        
        p = tl.exp2(s) / l[:, None]
        p = tl.sum(p, axis=0) # / LAST_Q
        tl.store(
            O_base_ptr + offset_k * stride_on,
            p.to(S_ptr.dtype.element_ty),
            mask=(offset_k < key_len)
        )

def get_block_mean_configs():
    configs = []
    warps = [4, 8]
    stages = [2, 3, 4]
    k_tile_size = [64, 128]
    for w in warps:
        for s in stages:
            for kt in k_tile_size:
                configs.append(
                    triton.Config(
                        {
                            'K_TILE_SIZE': kt
                        },
                        num_warps=w,
                        num_stages=s
                    )
                )
    return configs

@triton.autotune(
    configs=get_block_mean_configs(),
    key = ['num_q_heads'],
    prune_configs_by={
        'early_config_prune': lambda configs, named_args, **kwargs: [
            c for c in configs 
            if named_args['BLOCK_SIZE'] <= c.kwargs['K_TILE_SIZE'] 
        ]
    }
)
@triton.jit
def fused_block_mean_kernel(
    O_ptr, B_ptr,
    stride_on, stride_oh,
    stride_bn,
    cu_seqlens, max_seqlen,
    cu_block_seqlens, 
    num_q_heads, num_q_heads_next_power_of_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr
):
    key_tile_index = tl.program_id(0).to(tl.int64)
    offset_batch = tl.program_id(1).to(tl.int64)

    cu_seqlens_k_start = tl.load(cu_seqlens + offset_batch)
    cu_seqlens_k_end = tl.load(cu_seqlens + offset_batch + 1)
    key_len = cu_seqlens_k_end - cu_seqlens_k_start

    if key_tile_index * K_TILE_SIZE > key_len:
        return
    
    cu_block_seqlens_k_start = tl.load(cu_block_seqlens + offset_batch)
    cu_block_seqlens_k_end = tl.load(cu_block_seqlens + offset_batch + 1)
    key_block_len = cu_block_seqlens_k_end - cu_block_seqlens_k_start
    
    O_base_ptr = O_ptr + cu_seqlens_k_start * stride_on 
    B_base_ptr = B_ptr + cu_block_seqlens_k_start * stride_bn

    lo = key_tile_index * K_TILE_SIZE
    hi = min(lo + K_TILE_SIZE, key_len)

    offset_heads = tl.arange(0, num_q_heads_next_power_of_2)

    for j in range(lo, hi, BLOCK_SIZE):
        offset_k = j + tl.arange(0, BLOCK_SIZE)
        score = tl.load(
            O_base_ptr + offset_heads[:, None] * stride_oh + offset_k[None, :] * stride_on,
            mask=(offset_heads[:, None] < num_q_heads) & (offset_k[None, :] < key_len),
            other=0.0
        )
        block_sum = tl.sum(score)
        block_mean = block_sum # / (num_q_heads * BLOCK_SIZE)
        tl.store(
            B_base_ptr + (j // BLOCK_SIZE) * stride_bn,
            block_mean.to(B_ptr.dtype.element_ty),
            mask=(j // BLOCK_SIZE) < key_block_len
        )

def get_top_p_configs():
    configs = []
    warps = [4, 8]
    stages = [2, 3, 4]
    for w in warps:
        for s in stages:
            configs.append(
                triton.Config(
                    {},
                    num_warps=w,
                    num_stages=s
                )
            )
    return configs

@triton.autotune(
    configs=get_top_p_configs(),
    key=[]
)
@triton.jit
def fused_top_p_kernel(
    B_ptr, M_ptr,
    p,
    stride_bn,
    stride_mn,
    cu_block_seqlens, 
    MAX_BLOCK_SEQLEN_NEXT_POWER_OF_2: tl.constexpr,
):
    offset_zh = tl.program_id(0).to(tl.int64)
    offset_batch = offset_zh

    cu_block_seqlens_k_start = tl.load(cu_block_seqlens + offset_batch)
    cu_block_seqlens_k_end = tl.load(cu_block_seqlens + offset_batch + 1)
    key_block_len = cu_block_seqlens_k_end - cu_block_seqlens_k_start

    B_base_ptr = B_ptr + cu_block_seqlens_k_start * stride_bn
    offset_block = tl.arange(0, MAX_BLOCK_SEQLEN_NEXT_POWER_OF_2)

    block_score = tl.load(
        B_base_ptr + offset_block * stride_bn,
        mask=offset_block < key_block_len,
        other=0.0
    )


    block_score_f32 = block_score.to(tl.float32)
    score_sum = tl.sum(block_score_f32, axis=0)
    block_score_f32 = block_score_f32 / (score_sum)
    # block_score_f32 = tl.where(offset_block < key_block_len, block_score_f32, float('-inf'))

    # Float -> sortable int (单调映射)
    scores_i32 = block_score_f32.to(tl.int32, bitcast=True)
    scores_sortable = tl.where(
        scores_i32 < 0,
        scores_i32 ^ (-1),            # 0xFFFFFFFF
        scores_i32 ^ (-2147483648)    # 0x80000000
    )

    # Pack
    scores_i64 = scores_sortable.to(tl.int64) & 0xFFFFFFFF
    idx_i64 = offset_block.to(tl.int64) & 0xFFFFFFFF
    packed = (scores_i64 << 32) | idx_i64

    # sort
    packed_sorted = tl.sort(packed, descending=True)

    # Unpack
    sorted_indices = (packed_sorted & 0xFFFFFFFF).to(tl.int32)
    sorted_sortable = (packed_sorted >> 32).to(tl.int32)

    # Sortable int -> float
    sorted_score_i32 = tl.where(
        sorted_sortable < 0,
        sorted_sortable ^ (-2147483648),  # 0x80000000
        sorted_sortable ^ (-1)            # 0xFFFFFFFF
    )
    sorted_score_f32 = sorted_score_i32.to(tl.float32, bitcast=True)


    sorted_score = sorted_score_f32.to(block_score.dtype)
    sorted_score = tl.where(offset_block < key_block_len, sorted_score, 0.0)


    cumsum_score = tl.cumsum(sorted_score, axis=0)
    mask = (cumsum_score - sorted_score) <= p
    # mask = (cumsum_score <= p)

    keep_mask = mask & (offset_block < key_block_len)
    keep_mask = keep_mask | (offset_block == 0)

    tl.store(
        M_ptr + (cu_block_seqlens_k_start + sorted_indices) * stride_mn,
        keep_mask,
        mask=(offset_block < key_block_len) & (sorted_indices < key_block_len) # 只写有效位置，padding 不写
    )


@triton.autotune(
    configs=get_top_p_configs(),
    key=[]
)
@triton.jit
def expand_block_mask_kernel(
    block_mask_ptr, # int8, [total_blocks]
    token_mask_ptr, # int8, [total_tokens]
    cu_seqlens, # int32, [batch+1]
    cu_block_seqlens, # int32, [batch+1]
    seqlens_out, # int32, [batch+1]
    ATTENTION_SINK: tl.constexpr,
    LAST_Q: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCK_SEQLEN_NEXT_POWER_OF_2: tl.constexpr,
):
    offset_batch = tl.program_id(0).to(tl.int64)

    token_start = tl.load(cu_seqlens + offset_batch)
    token_end = tl.load(cu_seqlens + offset_batch + 1)
    seq_len = token_end - token_start

    block_start = tl.load(cu_block_seqlens + offset_batch)
    block_end = tl.load(cu_block_seqlens + offset_batch + 1)
    key_block_len = block_end - block_start

    offset_token = tl.arange(0, MAX_BLOCK_SEQLEN_NEXT_POWER_OF_2 * BLOCK_SIZE)
    which_block = offset_token // BLOCK_SIZE 
    token_mask = tl.load(
        block_mask_ptr + block_start + which_block,
        mask=(which_block < key_block_len) & (offset_token < seq_len),
        other=0
    )

    is_sink = offset_token < ATTENTION_SINK
    is_last_q = offset_token >= (seq_len - LAST_Q)
    token_mask = token_mask | is_sink | is_last_q

    tl.store(
        token_mask_ptr + token_start + offset_token,
        token_mask,
        mask=offset_token < seq_len
    )

    token_mask_rectified = token_mask & (offset_token < seq_len)
    seq_lens_new = tl.sum(token_mask_rectified.to(tl.int32))

    tl.store(
        seqlens_out + offset_batch + 1,
        seq_lens_new.to(seqlens_out.dtype.element_ty)
    )

def get_cu_block_seqlens(cu_seqlens, block_size):
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    tiles_per_seq = (seqlens + block_size - 1) // block_size
    cu_tile_seqlens = torch.zeros(cu_seqlens.shape, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
    cu_tile_seqlens[1:] = torch.cumsum(tiles_per_seq, dim=0)
    return cu_tile_seqlens



def topselectionvarlen(
    q, k, head_dim, cu_seqlens, max_seq_len,
    block_size=64, attention_sink=128, last_q=128, p=0.99
):
    

    q = q.reshape(q.shape[0], -1, head_dim)
    k = k.reshape(k.shape[0], -1, head_dim)


    
    total_tokens, num_q_heads, d_head = q.shape
    _, num_k_heads, _ = k.shape

    batch = cu_seqlens.shape[0] - 1
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]  # [batch]


    is_prefill_req = (seq_lens > 1) # & (seq_lens >= last_q)
    prefill_indices = is_prefill_req.nonzero(as_tuple=True)[0]
    n_prefill = prefill_indices.shape[0]

    if n_prefill == 0:
        token_mask = torch.ones(total_tokens, dtype=torch.bool, device=q.device)
        return token_mask, int(max_seq_len), cu_seqlens.clone()

    # --------
    prefill_seq_lens = seq_lens[prefill_indices]
    prefill_cu_seqlens = torch.zeros(n_prefill + 1, dtype=torch.int32, device=q.device)
    prefill_cu_seqlens[1:] = torch.cumsum(prefill_seq_lens, dim=0)
    prefill_max_seq_len = int(prefill_seq_lens.max().item())
    prefill_total = int(prefill_cu_seqlens[-1].item())

    # --------
    prefill_starts = cu_seqlens[prefill_indices]  # [n_prefill]
    req_ids = torch.repeat_interleave(
        torch.arange(n_prefill, device=q.device, dtype=torch.int64),
        prefill_seq_lens.long()
    )
    local_offsets = torch.arange(prefill_total, device=q.device, dtype=torch.int64)
    intra_pos = local_offsets - prefill_cu_seqlens[req_ids].long()
    prefill_token_indices = prefill_starts[req_ids].long() + intra_pos

    prefill_q = q[prefill_token_indices]
    prefill_k = k[prefill_token_indices]

    # --------
    s = torch.full(
        (last_q, prefill_total, num_q_heads), 
        float('-inf'), 
        dtype=q.dtype, 
        device=q.device
    )

    def grid_gemm(meta):
        return (triton.cdiv(prefill_max_seq_len, meta['K_TILE_SIZE']),
                n_prefill * num_q_heads, 1)
    fused_part_gemm_kernel[grid_gemm](
        prefill_q, prefill_k, s,
        *prefill_q.stride(), *prefill_k.stride(), *s.stride(),
        prefill_cu_seqlens, prefill_max_seq_len,
        num_q_heads, num_k_heads, last_q, D_HEAD=d_head
    )

    o = torch.zeros(prefill_total, num_q_heads, dtype=q.dtype, device=q.device)

    def grid_softmax(meta):
        return (n_prefill * num_q_heads, 1, 1)
    fused_block_softmax_kernel[grid_softmax](
        s, o, 1 / (d_head ** 0.5),
        *s.stride(), *o.stride(),
        prefill_cu_seqlens, prefill_max_seq_len,
        num_q_heads, num_k_heads, last_q, last_q
    )

    cu_block_seqlens = get_cu_block_seqlens(prefill_cu_seqlens, block_size)
    num_q_heads_np2 = triton.next_power_of_2(num_q_heads)
    b = torch.zeros((cu_block_seqlens[-1].item(),), dtype=q.dtype, device=q.device)

    def grid_mean(meta):
        return (triton.cdiv(prefill_max_seq_len, meta['K_TILE_SIZE']),
                n_prefill, 1)
    fused_block_mean_kernel[grid_mean](
        o, b, *o.stride(), *b.stride(),
        prefill_cu_seqlens, prefill_max_seq_len, cu_block_seqlens,
        num_q_heads, num_q_heads_np2, block_size
    )

    tp_group = get_tp_group()
    if tp_group.world_size > 1:
        b_f32 = b.float()
        b_f32 = tp_group.all_reduce(b_f32)
        b = b_f32.to(q.dtype)

    m = torch.zeros((cu_block_seqlens[-1].item(),), dtype=torch.bool, device=q.device)

    def grid_top_p(meta):
        return (n_prefill, 1, 1)
    fused_top_p_kernel[grid_top_p](
        b, m, p, *b.stride(), *m.stride(),
        cu_block_seqlens,
        triton.next_power_of_2(triton.cdiv(prefill_max_seq_len, block_size))
    )

    prefill_token_mask = torch.zeros(prefill_total, dtype=torch.bool, device=q.device)
    prefill_seqlens_new = torch.zeros_like(prefill_cu_seqlens)

    def grid_expand(meta):
        return (n_prefill, 1, 1)
    expand_block_mask_kernel[grid_expand](
        m, prefill_token_mask,
        prefill_cu_seqlens, cu_block_seqlens, prefill_seqlens_new,
        attention_sink, last_q, block_size,
        triton.next_power_of_2(triton.cdiv(prefill_max_seq_len, block_size))
    )

    if tp_group.world_size > 1:
        torch.distributed.broadcast(prefill_token_mask, src=0, group=tp_group.device_group)
        torch.distributed.broadcast(prefill_seqlens_new, src=0, group=tp_group.device_group)

    # --------
    token_mask = torch.ones(total_tokens, dtype=torch.bool, device=q.device)
    token_mask[prefill_token_indices] = prefill_token_mask

    new_seq_lens = seq_lens.to(torch.int32).clone()
    new_seq_lens[prefill_indices] = prefill_seqlens_new[1:]

    new_cu_seqlens = torch.zeros(batch + 1, dtype=torch.int32, device=q.device)
    new_cu_seqlens[1:] = torch.cumsum(new_seq_lens, dim=0)
    new_max_seq_len = int(new_seq_lens.max().item())

    return token_mask, new_max_seq_len, new_cu_seqlens
