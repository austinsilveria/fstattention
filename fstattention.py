import torch

import triton
import triton.language as tl
from triton import cdiv, jit

from utils import (
    create_tree,
    create_fst_attention_kernel_inputs,
    create_full_attention_mask,
    DEPTH_MAPPING,
    MAX_ANCESTOR_MAPPING,
)

# Adapted from Triton's fused attention implementation
# https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html

# triton=triton-nightly-2.1.0.post20240108192258
# pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

@jit
def _fwd_kernel(Q, K, V,
                ANCESTOR_IDX,  # [NUM_M_BLOCKS, MAX_ANCESTORS]    - Unique ancestors attended to by queries in this block
                ANCESTOR_MASK, # [NUM_M_BLOCKS, M, MAX_ANCESTORS] - Attention mask applied to q[m]*ancestor_idxs[m]
                LEAF_IDX,      # [NUM_M_BLOCKS, M]                - Indices of leaf nodes in this block
                shared_prefix_length,
                sm_scale,  #
                L,  #
                Out,  #
                stride_qz, stride_qh, stride_qm, stride_qk,  #
                stride_kz, stride_kh, stride_kn, stride_kk,  #
                stride_vz, stride_vh, stride_vn, stride_vk,  #
                stride_oz, stride_oh, stride_om, stride_on,  #
                stride_aim, stride_ain,                      # ANCESTOR_IDX strides
                stride_am, stride_aq, stride_an,             # ANCESTOR_MASK strides
                stride_lm, stride_lq,                        # LEAF_IDX strides
                Z, H, N_CTX,  #
                Z_H_N_CTX,  #
                BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,  #
                MAX_ANCESTORS: tl.constexpr,  #
                ):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * stride_qh
    vk_offset = off_hz * stride_kh

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_q = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, MAX_ANCESTORS)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # credits to: Adam P. Goucher (https://github.com/apgoucher):
    # scale sm_scale by 1/log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504

    # Tree - Load ancestor indices, mask, and leaf indices
    ancestor_idx_ptrs = ANCESTOR_IDX + start_m * stride_aim + offs_n * stride_ain
    ancestor_mask_ptrs = ANCESTOR_MASK + start_m * stride_am + offs_q[:, None] * stride_aq + offs_n[None, :] * stride_an
    leaf_idx_ptrs = LEAF_IDX + start_m * stride_lm + offs_q * stride_lq
    ancestor_idx = tl.load(ancestor_idx_ptrs)
    ancestor_mask = tl.load(ancestor_mask_ptrs)
    leaf_idx = tl.load(leaf_idx_ptrs)

    offs_k = tl.arange(0, BLOCK_DMODEL)
      
    Q_ptrs = Q + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    # Tree - manage low level KV pointers since ancestors non-contiguous (should still be efficient reads since head_dim = 64 @ fp16 and default transaction size is 64 bytes)
    #        https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21819-optimizing-applications-for-nvidia-ampere-gpu-architecture.pdf
    #        could potentially overfetch ancestors with quantization, but that'll just prefetch leaves and they'll end up in l2 cache and we'll use them shortly after
    K_ptrs = K + vk_offset + offs_k[:, None] * stride_kk + ancestor_idx[None, :] * stride_kn
    q = tl.load(Q_ptrs)

    q = (q * qk_scale).to(K.dtype.element_ty)
    # Tree - compute ancestor attention weights
    ancestor_kv_load_mask = ancestor_idx != -1
    k = tl.load(K_ptrs, mask=ancestor_kv_load_mask[None, :], other=0)
    qk = tl.zeros([BLOCK_M, MAX_ANCESTORS], dtype=tl.float32)
    qk += tl.dot(q, k, allow_tf32=True)
    qk = tl.where(ancestor_mask, qk, float("-inf"))

    # Tree - compute leaf self attention weights
    leaf_kv_load_mask = leaf_idx != -1
    lK_ptrs = K + vk_offset + leaf_idx[:, None] * stride_kn + offs_k[None, :] * stride_kk 
    lk = tl.load(lK_ptrs, mask=leaf_kv_load_mask[:, None], other=0)
    lqk = tl.sum(q * lk, 1)
    lqk = tl.where(leaf_kv_load_mask, lqk, float("-inf"))

    # # -- compute scaling constant ---
    m_i_new = tl.maximum(tl.max(qk, 1), lqk)
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    lp = tl.math.exp2(lqk - m_i_new)

    # Tree - compute attention weighted ancestor values
    V_ptrs = V + vk_offset + ancestor_idx[:, None] * stride_vn + offs_k[None, :] * stride_vk
    v = tl.load(V_ptrs, mask=ancestor_kv_load_mask[:, None], other=0)
    acc += tl.dot(p.to(V.dtype.element_ty), v, allow_tf32=True)

    # Tree - compute attention weighted leaf values
    lV_ptrs = V + vk_offset + leaf_idx[:, None] * stride_vn + offs_k[None, :] * stride_vk
    lv = tl.load(lV_ptrs, mask=leaf_kv_load_mask[:, None], other=0)
    acc += lp.to(V.dtype.element_ty)[:, None] * lv

    # -- update m_i and l_i --
    l_i = l_i * alpha + tl.sum(p, 1) + lp
    m_i = m_i_new

    # shared prefix attention
    lo = 0
    hi = shared_prefix_length
    for start_n in range(lo, hi, MAX_ANCESTORS):
        block_n = start_n + offs_n
        # -- load k, v --
        K_ptrs = K + vk_offset + offs_k[:, None] * stride_kk + block_n[None, :] * stride_kn
        V_ptrs = V + vk_offset + block_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
        k = tl.load(K_ptrs)
        v = tl.load(V_ptrs)
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, MAX_ANCESTORS], dtype=tl.float32)
        qk += tl.dot(q, k, allow_tf32=True)
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc *= alpha[:, None]
        acc += tl.dot(p.to(V.dtype.element_ty), v, allow_tf32=True)

        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # write back l and m
    acc = acc / l_i[:, None]
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, m_i + tl.math.log2(l_i))
    # write back O
    O_block_ptr = tl.make_block_ptr(
        base=Out,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=((qvk_offset // stride_qm) + start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    tl.store(O_block_ptr, acc.to(K.dtype.element_ty))

def fst_attention(q, k, v, ancestor_idx, ancestor_mask, leaf_idx, sm_scale, BLOCK_M=128, MAX_ANCESTORS=64):
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.empty_like(q)
    grid = (cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    num_warps = 4 if Lk <= 64 else 8

    shared_prefix_length = k.shape[2] - q.shape[2]
    _fwd_kernel[grid](
        q, k, v, ancestor_idx, ancestor_mask, leaf_idx, shared_prefix_length, sm_scale,  #
        L,  #
        o,  #
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
        ancestor_idx.stride(0), ancestor_idx.stride(1), #
        ancestor_mask.stride(0), ancestor_mask.stride(1), ancestor_mask.stride(2), #
        leaf_idx.stride(0), leaf_idx.stride(1), #
        q.shape[0], q.shape[1], q.shape[2],  #
        q.shape[0] * q.shape[1] * q.shape[2],  #
        BLOCK_M=BLOCK_M, MAX_ANCESTORS=MAX_ANCESTORS, BLOCK_DMODEL=Lk,  #
        num_warps=num_warps,  #
        num_stages=4  #
    )
    return o

def test_op(Z, H, N_CTX, D_HEAD, shared_kv_prefix=True, dtype=torch.float16):
    torch.manual_seed(20)
    # Query tree of size N_CTX
    q = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    if shared_kv_prefix:
      # KV tree of size N_CTX + fully shared (by all tree queries) KV prefix of size N_CTX
      k = (torch.empty((Z, H, 2 * N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
      v = (torch.empty((Z, H, 2 * N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    else:
      k = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
      v = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 0.5

    lineage, level_lookup = create_tree(depth=DEPTH_MAPPING[N_CTX])
    full_mask = create_full_attention_mask(lineage)
    BLOCK_M = 128
    MAX_ANCESTORS = MAX_ANCESTOR_MAPPING[N_CTX]
    ancestor_idx, ancestor_mask, leaf_idx = create_fst_attention_kernel_inputs(
        lineage,
        level_lookup,
        block_m=BLOCK_M,
        max_ancestors=MAX_ANCESTORS
    )

    if shared_kv_prefix:
      # Full attention to shared prefix
      full_mask = torch.cat((torch.ones((N_CTX, N_CTX), dtype=torch.bool), full_mask), dim=1)
      # Shift to account for the shared prefix
      ancestor_idx += N_CTX
      leaf_idx += N_CTX

    # reference implementation
    M = full_mask.to('cuda')
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)

    # triton implementation
    ancestor_idx = ancestor_idx.to('cuda')
    ancestor_mask = ancestor_mask.to('cuda')
    leaf_idx = leaf_idx.to('cuda')
    tri_out = fst_attention(q, k, v, ancestor_idx, ancestor_mask, leaf_idx, sm_scale).half()

    # compare (ignore last 4 since we added padding)
    assert torch.allclose(ref_out[:, :, :-4], tri_out[:, :, :-4], atol=1e-2, rtol=0)


try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')
# TORCH_HAS_FP8 = False
BATCH, N_HEADS, N_CTX, D_HEAD = 4, 48, 4096, 64
# vary seq length for fixed head and batch=4
configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["N_CTX"],
        x_vals=[2**i for i in range(10, 17)],
        line_arg="provider",
        line_vals=["fst"] + (["flash"] if HAS_FLASH else []),
        line_names=["Fused Sparse Tree"] + (["Flash-2"] if HAS_FLASH else []),
        styles=[("red", "-"), ("blue", "-")],
        ylabel="Wall Time (ms)",
        plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{D_HEAD}-{'fwd'}-causal={True}",
        args={
            "H": N_HEADS,
            "BATCH": BATCH,
            "D_HEAD": D_HEAD,
            "dtype": torch.float16,
            "mode": 'fwd',
            "causal": True,
        },
    ))

@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, D_HEAD, causal, mode, provider, dtype=torch.float16, device="cuda"):
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    if provider == "fst":
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        sm_scale = 1.3

        lineage, level_lookup = create_tree(depth=DEPTH_MAPPING[N_CTX])
        BLOCK_M = 128
        MAX_ANCESTORS = MAX_ANCESTOR_MAPPING[N_CTX]
        ancestor_idx, ancestor_mask, leaf_idx = create_fst_attention_kernel_inputs(
            lineage,
            level_lookup,
            block_m=BLOCK_M,
            max_ancestors=MAX_ANCESTORS
        )

        ancestor_idx = ancestor_idx.to('cuda')
        ancestor_mask = ancestor_mask.to('cuda')
        leaf_idx = leaf_idx.to('cuda')
        fn = lambda: fst_attention(q, k, v, ancestor_idx, ancestor_mask, leaf_idx, sm_scale, BLOCK_M=BLOCK_M, MAX_ANCESTORS=MAX_ANCESTORS)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        # This doesn't compute the same output, since we're not applying the tree mask, but it compares
        #   1) the speed of only loading KVs for shared ancestors + self (fstattention)
        #   2) the speed of loading all causal KVs (flashattention)
        fn = lambda: flash_attn_func(qkv, causal=causal)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms

test_op(2, 2, 2**11, 64, shared_kv_prefix=True)
test_op(2, 2, 2**11, 64, shared_kv_prefix=False)
bench_flash_attention.run(save_path=".", print_data=True)
