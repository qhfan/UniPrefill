import torch
from fla.utils import autocast_custom_fwd, contiguous
from vllm.distributed import get_tp_group
from vllm.logger import init_logger

# logger = init_logger(__name__)

class TopPSelectionVarLen(torch.autograd.Function):
    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, head_dim, cu_seqlens, max_seq_len, block_size=64, attention_sink=128, last_n=128, alpha=0.99):
        '''
        q: (b nq s d)
        k: (b nk s d)
        block_size: 
        alpha: nucleus sampling

        '''
        q = q.reshape(1, q.shape[0], -1, head_dim).transpose(1, 2)
        k = k.reshape(1, k.shape[0], -1, head_dim).transpose(1, 2)
        _, num_q_heads, _, _ = q.shape
        batch_size, num_k_heads, seq_len, head_dim = k.shape

        num_blocks = (seq_len + block_size - 1) // block_size

        q_last_n = q[:, :, -last_n:, :]
        q_last_n_start_pos = seq_len - last_n
        
        k_repeat = k.repeat_interleave(num_q_heads // num_k_heads, dim=1)

        score = q_last_n @ k_repeat.transpose(-2, -1) / (head_dim ** 0.5)
        
        query_rel_pos = torch.arange(last_n, device=score.device, dtype=torch.long)
        query_abs_pos = query_rel_pos + q_last_n_start_pos
        key_pos = torch.arange(seq_len, device=score.device, dtype=torch.long)
        causal_mask = query_abs_pos.unsqueeze(1) < key_pos.unsqueeze(0)
        score = score.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        score = torch.softmax(score, dim=-1)
        score = torch.where(torch.isnan(score), torch.zeros_like(score), score)
        
        score = score.mean(dim=1)  # (b last_n s)
        score = score.mean(dim=1)  # (b s)

        padded_len = num_blocks * block_size
        if padded_len > seq_len:
            score_padded = torch.nn.functional.pad(score, (0, padded_len - seq_len), value=0.0)
        else:
            score_padded = score

        score_block = score_padded.reshape(batch_size, num_blocks, block_size)
        block_score = score_block.mean(dim=-1)  # (b num_blocks)
        
        sink_blocks = (attention_sink + block_size - 1) // block_size
        last_n_blocks = (last_n + block_size - 1) // block_size



        tp_group = get_tp_group()
        if tp_group.world_size > 1:
            b_f32 = block_score.float()
            b_f32 = tp_group.all_reduce(b_f32)
            block_score = b_f32.to(q.dtype)

        block_prob = block_score / (block_score.sum(dim=-1, keepdim=True) + 1e-9)  # (b num_blocks)
        
        sorted_probs, sorted_indices = torch.sort(block_prob, descending=True, dim=-1)
        
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        sorted_mask = cumsum_probs <= alpha
        
        sorted_mask[:, 0] = True
        
        block_mask = torch.zeros_like(block_prob, dtype=torch.bool)
        block_mask.scatter_(1, sorted_indices, sorted_mask)
        
        block_mask[:, :sink_blocks] = True
        block_mask[:, num_blocks - last_n_blocks:] = True
        
        
        mask = block_mask.unsqueeze(-1).expand(-1, -1, block_size)  
        mask = mask.reshape(batch_size, padded_len)  
        mask = mask[:, :seq_len].squeeze(0)

        if tp_group.world_size > 1:
            torch.distributed.broadcast(mask, src=0, group=tp_group.device_group)

        new_max_seq_len = torch.sum(mask).item()
        new_cu_seqlens = torch.tensor([0, new_max_seq_len], dtype=torch.int32, device=q.device)


        
        return mask, new_max_seq_len, new_cu_seqlens


def UpdateKVCache(query_states, key_states, hidden_states, cos, sin, attention_sink, last_n, alpha, block_size):
    '''
    only support batch_size==1
    query_states: (1, num_q_heads, seq_len, head_dim)
    key_states: (1, num_k_heads, seq_len, head_dim)
    hidden_states: (bs, seq_len, dim)
    cos, sin: (1, seq_len, head_dim)
    attention_sink: attention sink states
    last_n: number of states to keep at the end
    alpha: scaling factor
    block_size: size of each block for block-wise processing
    
    Returns:
        updated_key_states: (1, num_k_heads, new_seq_len, head_dim)
        updated_value_states: (1, num_k_heads, new_seq_len, head_dim)
    '''

    attn_mask = ComputeKeyScoreLastBlockWise.apply(query_states, key_states, attention_sink, last_n, alpha, block_size)  # (1, seq_len) # type: ignore
    
    mask = attn_mask.squeeze(0) 

    print('*' * 60)
    print(mask.shape)


    selected_hidden_states = hidden_states[:, mask, :] 

    selected_cos = cos[:, mask.to(cos.device), :]  
    selected_sin = sin[:, mask.to(sin.device), :]  

    return selected_hidden_states, (selected_cos, selected_sin)
