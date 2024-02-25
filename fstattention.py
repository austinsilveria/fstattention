import time
import pytest
import torch

import triton
import triton.language as tl
from triton import cdiv, jit

# Adapted from Triton's fused attention implementation
# https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html

# triton=triton-nightly-2.1.0.post20240108192258
# pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

import math
from collections import OrderedDict

import torch

depth = 7
topk = 4
left_weight_factor = 0.5

dense_tree_idxs = torch.empty((depth + 2), dtype=torch.int)
dense_tree_idxs[0] = 0
dense_tree_idxs[1] = topk
prev_level_size = topk
for i in range(2, depth + 2):
    prev_level_size = topk * int(prev_level_size * left_weight_factor)
    dense_tree_idxs[i] = dense_tree_idxs[i - 1] + prev_level_size
print(f'dense_tree_idxs: {dense_tree_idxs}')

indices = torch.arange(0, dense_tree_idxs[-1])
print(f'indices: {indices}')

# levels = torch.ceil(torch.log(indices.float()) / torch.log(torch.tensor(topk, device='cuda').float()))
levels = (indices.unsqueeze(1) < dense_tree_idxs.unsqueeze(0)).char().argmax(dim=1)
level_lookup = levels.long() - 1  # Convert to long for use as indices
print(f'level_lookup: {level_lookup}')

# Compute relative indices within their levels (how far from beginning/end of the level)
relative_idxs = indices - dense_tree_idxs[level_lookup]
print(f'relative_idxs: {relative_idxs}')
# Compute the parent's relative index
#   1)  
#   2) go up one level and add the parent's relative index to the beginning of the level
parent_relative_idxs = torch.floor((relative_idxs) / topk).char()
print(f'parent_relative_idxs: {parent_relative_idxs}')
parent_indices = dense_tree_idxs[level_lookup - 1] + parent_relative_idxs
# parent_indices[:topk] = 0
print(f'parent_indices: {parent_indices}')

lineage = torch.ones((indices.shape[0], depth + 1), dtype=torch.int) * -1
# parents = parent_indices.clone()
# for _ in range(depth):
#     lineage[:, level_lookup[parents]] = parents
#     parents = parent_indices[parents]
# print(f'lineage: {lineage}')
for i in range(topk, indices.shape[0]):
    idx = indices[i]
    parent = parent_indices[idx]
    print(f'idx: {idx}, parent: {parent}')
    lineage[idx, level_lookup[parent]] = parent
    for j in range(depth):
        parent = parent_indices[parent]
        if parent == dense_tree_idxs[-1]:
            break
        # print(f'idx: {idx}, parent: {parent}')
        lineage[idx, level_lookup[parent]] = parent

full_unq = torch.unique(lineage.flatten(), dim=0)[1:]

CHUNK_SIZE = 64
torch.set_printoptions(profile="full")

# unique per chunk
print(f'lineage: {lineage}')
max_ancestors = 32
unq = torch.ones((math.ceil(lineage.shape[0] / CHUNK_SIZE), max_ancestors), dtype=torch.long) * -1
mask = torch.zeros((math.ceil(lineage.shape[0] / CHUNK_SIZE), CHUNK_SIZE, max_ancestors), dtype=torch.bool)
leaf_idxs = torch.ones((math.ceil(lineage.shape[0] / CHUNK_SIZE), CHUNK_SIZE), dtype=torch.long) * -1
for i in range(0, lineage.shape[0], CHUNK_SIZE):
    chunk_idx = i // CHUNK_SIZE
    chunk = lineage[i:i+CHUNK_SIZE]
    chunk_unq = torch.unique(chunk.flatten(), dim=0)[1:]
    unq[chunk_idx, :chunk_unq.shape[0]] = chunk_unq
    chunk_leaf_idxs = torch.tensor(list(set(torch.arange(i, i+CHUNK_SIZE, dtype=torch.long).tolist()).difference(set(chunk_unq.tolist()))), dtype=torch.long)
    # leaf_idxs[chunk_idx, :chunk_leaf_idxs.shape[0]] = chunk_leaf_idxs
    leaf_idxs[chunk_idx, chunk_leaf_idxs - i] = chunk_leaf_idxs

    for k in range(i, min(i + CHUNK_SIZE, lineage.shape[0])):
        chunk[k - (chunk_idx * CHUNK_SIZE), level_lookup[k]] = k
    for j in range(chunk.shape[0]):
        mask[chunk_idx, j, :chunk_unq.shape[0]] = torch.isin(chunk_unq, chunk[j])

print(f'lineage after: {lineage}')
print(f'unq: {unq}')
print(f'unq shape: {unq.shape}')
# print(f'mask: {mask}')
print(f'mask shape: {mask.shape}')
print(f'leaf_idxs: {leaf_idxs}')
print(f'leaf_idxs shape: {leaf_idxs.shape}')
# Add self to lineage after unique
for i in range(indices.shape[0]):
    lineage[i, level_lookup[i]] = i
# print(f'lineage: {lineage}')

print(f'full lineage uniq: {full_unq}')
print(f'full num uniq keys: {full_unq.shape[0] + lineage.shape[0] - 1}')
full_mask = torch.zeros((lineage.shape[0], lineage.shape[0]), dtype=torch.bool)
for i in range(lineage.shape[0]):
    full_mask[i, lineage[i]] = 1
full_mask[:, -1] = 0
full_mask[-1, -1] = 1
# for i in range(lineage.shape[0]):
#     full_mask[i] = torch.isin(full_unq, lineage[i])
# # print(f'w/o self mask: {mask}')
# print(f'w/o self mask shape: {full_mask.shape}')
# self_mask = torch.eye(lineage.shape[0], dtype=torch.bool)
# full_leaf_idxs = torch.tensor(list(set(torch.arange(0, lineage.shape[0], dtype=torch.long).tolist()).difference(set(full_unq.tolist()))), dtype=torch.long)
# # print(f'leaf_idxs: {leaf_idxs}')
# print(f'full leaf_idxs: {full_leaf_idxs.shape}')
# self_mask = self_mask[:, full_leaf_idxs]
# print(f'self mask: {self_mask[2]}')
# print(f'self mask: {self_mask.shape}')
# full_mask = torch.cat((full_mask, self_mask), dim=1)
print(f'full mask: {full_mask[2]}')
pad_size = 4
padding = torch.zeros((full_mask.shape[0], pad_size))
full_mask = torch.cat((full_mask, padding), dim=1)
padding = torch.zeros((pad_size, full_mask.shape[1]))
full_mask = torch.cat((full_mask, padding), dim=0)
print(f'full mask[-6]: {full_mask[-6]}')
print(f'full mask[-5]: {full_mask[-5]}')
# print(f'mask: {mask}')
print(f'full mask shape: {full_mask.shape}')
torch.set_printoptions(profile="default")

# Query and key chunks to pass to kernel
torch.set_printoptions(profile="full")
queries = indices[:CHUNK_SIZE]
print(f'queries: {queries}')
print(f'queries: {queries.shape}')
ancestor_keys = unq[-1]
ancestor_mask = mask[-1]
leaf_keys = leaf_idxs[-1]
print(f'ancestor_keys: {ancestor_keys}')
print(f'ancestor_keys: {ancestor_keys.shape}')
print(f'ancestor_mask: {ancestor_mask}')
print(f'ancestor_mask: {ancestor_mask.shape}')
print(f'leaf_keys: {leaf_keys}')
print(f'leaf_keys: {leaf_keys.shape}')
torch.set_printoptions(profile="default")

@jit
def _fwd_kernel(Q, K, V,
                ANCESTOR_IDX,  # [NUM_M_BLOCKS, MAX_ANCESTORS]    - Unique ancestors attended to by queries in this block
                ANCESTOR_MASK, # [NUM_M_BLOCKS, M, MAX_ANCESTORS] - Attention mask applied to q[m]*ancestor_idxs[m]
                LEAF_IDX,      # [NUM_M_BLOCKS, M]                - Indices of leaf nodes in this block
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
                BLOCK_N: tl.constexpr,  #
                IS_CAUSAL: tl.constexpr  #
                ):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    first_p = start_m == 15 and off_hz == 1
    qvk_offset = off_hz * stride_qh
    # vk_offset = qvk_offset // stride_qm
    vk_offset = off_hz * stride_kh
    # if first_p:
    #   tl.device_print('vk_offset', vk_offset)

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_q = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # credits to: Adam P. Goucher (https://github.com/apgoucher):
    # scale sm_scale by 1/log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout

    # Tree - tree pointers
    ancestor_idx_ptrs = ANCESTOR_IDX + start_m * stride_aim + offs_n * stride_ain
    ancestor_mask_ptrs = ANCESTOR_MASK + start_m * stride_am + offs_q[:, None] * stride_aq + offs_n[None, :] * stride_an
    leaf_idx_ptrs = LEAF_IDX + start_m * stride_lm + offs_q * stride_lq

    offs_k = tl.arange(0, BLOCK_DMODEL)
    ancestor_idx = tl.load(ancestor_idx_ptrs)
    ancestor_mask = tl.load(ancestor_mask_ptrs)
    leaf_idx = tl.load(leaf_idx_ptrs)
      
    Q_ptrs = Q + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    # Tree - manage low level KV pointers since ancestors non-contiguous (should still be efficient reads since head_dim = 64 @ fp16 and default transaction size is 64 bytes)
    #        https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21819-optimizing-applications-for-nvidia-ampere-gpu-architecture.pdf
    #        could potentially overfetch ancestors with quantization, but that'll just prefetch leaves and they'll end up in l2 cache and we'll use them shortly after
    #        offs_n = ancestor_idx
    K_ptrs = K + vk_offset + offs_k[:, None] * stride_kk + ancestor_idx[None, :] * stride_kn
    # K_ptrs = K + qvk_offset + offs_k[:, None] * stride_kk + ancestor_idx[None, :] * stride_kn
    q = tl.load(Q_ptrs)

    q = (q * qk_scale).to(K.dtype.element_ty)
    # Tree - compute ancestor attention in single pass (i.e. MAX_ANCESTORS < BLOCK_N)
    #        then loop over leaf self attentions in blocks of BLOCK_N
    #        (can think about including attention to shared prefix later--queries already batched,
    #         but there are potential tradeoffs with parallel KV loading like FlashDecoding)
    # Tree - load ancestor KVs
    ancestor_kv_load_mask = ancestor_idx != -1
    k = tl.load(K_ptrs, mask=ancestor_kv_load_mask[None, :], other=0)
    # Tree - since MAX_ANCESTORS < BLOCK_N, we use last column for leaf self attentions
    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, k, allow_tf32=True)
    # Tree - apply ancestor mask
    qk = tl.where(ancestor_mask, qk, float("-inf"))

    # Tree - compute leaf QKs
    #        add masking to leaf KV loads since first block has less leaves
    leaf_kv_load_mask = leaf_idx != -1
    K_ptrs = K + vk_offset + leaf_idx[:, None] * stride_kn + offs_k[None, :] * stride_kk 
    k = tl.load(K_ptrs, mask=leaf_kv_load_mask[:, None], other=0)
    # Tree - compute leaf self attentions
    lqk = tl.sum(q * k, 1)
    lqk = tl.where(leaf_kv_load_mask, lqk, float("-inf"))
    if first_p:
      tl.device_print('lqk', lqk)

    # Tree - don't need incremental softmax since we only have one QK matrix
    # # -- compute scaling constant ---
    m_i_new = tl.maximum(tl.maximum(m_i, tl.max(qk, 1)), lqk)
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    lp = tl.math.exp2(lqk - m_i_new)
    # # -- scale and update acc --
    # acc *= alpha[:, None]

    # Tree - load ancestor Vs and compute PV
    V_ptrs = V + vk_offset + ancestor_idx[:, None] * stride_vn + offs_k[None, :] * stride_vk
    v = tl.load(V_ptrs, mask=ancestor_kv_load_mask[:, None], other=0)
    acc += tl.dot(p.to(V.dtype.element_ty), v, allow_tf32=True)

    # Tree - load leaf Vs and compute PV
    V_ptrs = V + vk_offset + leaf_idx[:, None] * stride_vn + offs_k[None, :] * stride_vk
    v = tl.load(V_ptrs, mask=leaf_kv_load_mask[:, None], other=0)
    acc += lp.to(V.dtype.element_ty)[:, None] * v

    # -- update m_i and l_i --
    l_i = l_i * alpha + tl.sum(p, 1) + lp
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
        # offsets=(vk_offset + start_m * BLOCK_M, 0),
        offsets=((qvk_offset // stride_qm) + start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    O_ptrs = Out + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    tl.store(O_block_ptr, acc.to(K.dtype.element_ty))


@jit
def _bwd_preprocess(
    Out,
    DO,
    Delta,
    BLOCK_M: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    # load
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    # compute
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_m, delta)


@jit
def _bwd_kernel_one_col_block(Q, K, V, sm_scale, qk_scale,  #
                              Out, DO,  #
                              DQ, DK, DV,  #
                              L,  #
                              D,  #
                              Q_block_ptr, K_block_ptr, V_block_ptr,  #
                              DO_block_ptr, DQ_block_ptr, DK_block_ptr, DV_block_ptr,  #
                              stride_dqa, stride_qz, stride_qh, stride_qm, stride_qk,  #
                              stride_kz, stride_kh, stride_kn, stride_kk,  #
                              stride_vz, stride_vh, stride_vn, stride_vk,  #
                              Z, H, N_CTX,  #
                              off_h, off_z, off_hz, start_n, num_block,  #
                              BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,  #
                              BLOCK_N: tl.constexpr,  #
                              SEQUENCE_PARALLEL: tl.constexpr,  #
                              CAUSAL: tl.constexpr,  #
                              MMA_V3: tl.constexpr  #
                              ):
    if CAUSAL:
        lo = start_n * BLOCK_M
    else:
        lo = 0

    Q_offset = (off_z * stride_qz + off_h * stride_qh) // stride_qm
    DQ_offset = off_z * stride_qz + off_h * stride_qh
    K_offset = (off_z * stride_kz + off_h * stride_kh) // stride_kn
    V_offset = (off_z * stride_vz + off_h * stride_vh) // stride_vn
    if SEQUENCE_PARALLEL:
        DQ_offset += stride_dqa * start_n
    DQ_offset = DQ_offset // stride_qm

    Q_block_ptr = tl.advance(Q_block_ptr, (lo + Q_offset, 0))
    K_block_ptr = tl.advance(K_block_ptr, (start_n * BLOCK_M + K_offset, 0))
    V_block_ptr = tl.advance(V_block_ptr, (start_n * BLOCK_M + V_offset, 0))
    DO_block_ptr = tl.advance(DO_block_ptr, (lo + Q_offset, 0))
    DQ_block_ptr = tl.advance(DQ_block_ptr, (lo + DQ_offset, 0))
    DK_block_ptr = tl.advance(DK_block_ptr, (start_n * BLOCK_M + K_offset, 0))
    DV_block_ptr = tl.advance(DV_block_ptr, (start_n * BLOCK_M + V_offset, 0))

    # initialize row/col offsets
    offs_n = start_n * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m = tl.arange(0, BLOCK_N)
    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_hz * N_CTX
    l_ptrs = L + off_hz * N_CTX
    # initialize dv amd dk
    dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # k and v stay in SRAM throughout
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)
    # loop over rows
    for start_m in range(lo, num_block * BLOCK_M, BLOCK_M):
        offs_m_curr = start_m + offs_m
        # load q, k, v, do on-chip
        q = tl.load(Q_block_ptr)
        # recompute p = softmax(qk, dim=-1).T
        # NOTE: `do` is pre-divided by `l`; no normalization here
        if CAUSAL:
            qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), float(0.0), float("-inf"))
        else:
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= qk_scale
        l_i = tl.load(l_ptrs + offs_m_curr)
        p = tl.math.exp2(qk - l_i[:, None])
        # compute dv
        do = tl.load(DO_block_ptr)
        dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do, allow_tf32=True)
        # compute dp = dot(v, do)
        Di = tl.load(D_ptrs + offs_m_curr)
        # dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
        dp = tl.dot(do, tl.trans(v), allow_tf32=True)
        # compute ds = p * (dp - delta[:, None])
        ds = (p * (dp - Di[:, None]) * sm_scale).to(Q.dtype.element_ty)
        # compute dk = dot(ds.T, q)
        dk += tl.dot(tl.trans(ds), q, allow_tf32=True)
        # compute dq
        if not SEQUENCE_PARALLEL:
            dq = tl.load(DQ_block_ptr)
            dq += tl.dot(ds, k, allow_tf32=True)
            tl.store(DQ_block_ptr, dq.to(Q.dtype.element_ty))
        elif SEQUENCE_PARALLEL:
            if MMA_V3:
                dq = tl.dot(ds, k, allow_tf32=True)
            else:
                # not work with mma v3, because M % 64 != 0
                dq = tl.trans(tl.dot(tl.trans(k), tl.trans(ds), allow_tf32=True))
            tl.store(DQ_block_ptr, dq.to(Q.dtype.element_ty))

        # increment pointers
        DQ_block_ptr = tl.advance(DQ_block_ptr, (BLOCK_M, 0))
        Q_block_ptr = tl.advance(Q_block_ptr, (BLOCK_M, 0))
        DO_block_ptr = tl.advance(DO_block_ptr, (BLOCK_M, 0))
    # write-back
    tl.store(DV_block_ptr, dv.to(V.dtype.element_ty))
    tl.store(DK_block_ptr, dk.to(K.dtype.element_ty))


@jit
def _bwd_kernel(Q, K, V, sm_scale,  #
                Out, DO,  #
                DQ, DK, DV,  #
                L,  #
                D,  #
                stride_dqa, stride_qz, stride_qh, stride_qm, stride_qk,  #
                stride_kz, stride_kh, stride_kn, stride_kk,  #
                stride_vz, stride_vh, stride_vn, stride_vk,  #
                Z, H, N_CTX,  #
                Z_H_N_CTX,  #
                SQ_Z_H_N_CTX,  #
                BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,  #
                BLOCK_N: tl.constexpr,  #
                SEQUENCE_PARALLEL: tl.constexpr,  #
                CAUSAL: tl.constexpr,  #
                MMA_V3: tl.constexpr  #
                ):
    qk_scale = sm_scale * 1.44269504
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H

    Q_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    if SEQUENCE_PARALLEL:
        DQ_block_ptr = tl.make_block_ptr(
            base=DQ,
            shape=(SQ_Z_H_N_CTX, BLOCK_DMODEL),
            strides=(stride_qm, stride_qk),
            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )
    else:
        DQ_block_ptr = tl.make_block_ptr(
            base=DQ,
            shape=(Z_H_N_CTX, BLOCK_DMODEL),
            strides=(stride_qm, stride_qk),
            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )

    DK_block_ptr = tl.make_block_ptr(
        base=DK,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    DV_block_ptr = tl.make_block_ptr(
        base=DV,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )

    num_block_n = tl.cdiv(N_CTX, BLOCK_N)
    if not SEQUENCE_PARALLEL:
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(Q, K, V, sm_scale, qk_scale, Out, DO,  #
                                      DQ, DK, DV,  #
                                      L,  #
                                      D,  #
                                      Q_block_ptr, K_block_ptr, V_block_ptr,  #
                                      DO_block_ptr, DQ_block_ptr, DK_block_ptr, DV_block_ptr,  #
                                      stride_dqa, stride_qz, stride_qh, stride_qm, stride_qk,  #
                                      stride_kz, stride_kh, stride_kn, stride_kk,  #
                                      stride_vz, stride_vh, stride_vn, stride_vk,  #
                                      Z, H, N_CTX,  #
                                      off_h, off_z, off_hz, start_n, num_block_n,  #
                                      BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL,  #
                                      BLOCK_N=BLOCK_N,  #
                                      SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,  #
                                      CAUSAL=CAUSAL,  #
                                      MMA_V3=MMA_V3  #
                                      )
    else:
        start_n = tl.program_id(1)
        _bwd_kernel_one_col_block(Q, K, V, sm_scale, qk_scale, Out, DO,  #
                                  DQ, DK, DV,  #
                                  L,  #
                                  D,  #
                                  Q_block_ptr, K_block_ptr, V_block_ptr,  #
                                  DO_block_ptr, DQ_block_ptr, DK_block_ptr, DV_block_ptr,  #
                                  stride_dqa, stride_qz, stride_qh, stride_qm, stride_qk,  #
                                  stride_kz, stride_kh, stride_kn, stride_kk,  #
                                  stride_vz, stride_vh, stride_vn, stride_vk,  #
                                  Z, H, N_CTX,  #
                                  off_h, off_z, off_hz, start_n, num_block_n,  #
                                  BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL,  #
                                  BLOCK_N=BLOCK_N,  #
                                  SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,  #
                                  CAUSAL=CAUSAL,  #
                                  MMA_V3=MMA_V3  #
                                  )


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, ancestor_idx, ancestor_mask, leaf_idx, causal, sm_scale, sequence_parallel=False):
        # only support for Ampere now
        # capability = torch.cuda.get_device_capability()
        # if capability[0] < 8:
        #     raise RuntimeError("Flash attention currently only supported for compute capability >= 80")
        # BLOCK_M = 128
        # BLOCK_N = 64
        BLOCK_M = 64
        BLOCK_N = 32
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q)
        grid = (cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        num_warps = 4 if Lk <= 64 else 8
        # print(f'ancestor_idx[0]: {ancestor_idx[0]}')
        # print(f'ancestor_mask[0]: {ancestor_mask[0]}')
        # print(f'leaf_idx[0]: {leaf_idx[0]}')
        # print(f'ancestor_idx shape: {ancestor_idx.shape}')
        # print(f'ancestor_mask shape: {ancestor_mask.shape}')
        # print(f'leaf_idx shape: {leaf_idx.shape}')
        # torch.cuda.synchronize()
        # start = time.time()
        _fwd_kernel[grid](
            q, k, v, ancestor_idx, ancestor_mask, leaf_idx, sm_scale,  #
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
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=Lk,  #
            IS_CAUSAL=causal,  #
            num_warps=num_warps,  #
            num_stages=4  #
        )
        # torch.cuda.synchronize()
        # print(f'kernel time: {time.time() - start}')

        ctx.save_for_backward(q, k, v, o, L)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        ctx.sequence_parallel = sequence_parallel
        return o

    @staticmethod
    def backward(ctx, do):
        capability = torch.cuda.get_device_capability()
        MMA_V3 = capability[0] >= 9
        # BLOCK = 128
        BLOCK = 64
        q, k, v, o, L = ctx.saved_tensors
        sequence_parallel = ctx.sequence_parallel
        seq_len_kv = k.shape[2]
        do = do.contiguous()
        if sequence_parallel:
            replicas = cdiv(seq_len_kv, BLOCK)
            new_dq_shape = (replicas, ) + q.shape
            dq = torch.zeros(new_dq_shape, device=q.device, dtype=q.dtype)
        else:
            dq = torch.zeros_like(q, dtype=q.dtype)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty_like(L)
        _bwd_preprocess[(cdiv(q.shape[2], BLOCK) * ctx.grid[1], )](
            o,
            do,
            delta,
            BLOCK_M=BLOCK,
            D_HEAD=ctx.BLOCK_DMODEL,
        )
        _bwd_kernel[(ctx.grid[1], cdiv(seq_len_kv, BLOCK) if sequence_parallel else 1)](
            q, k, v, ctx.sm_scale,  #
            o, do,  #
            dq, dk, dv,  #
            L,  #
            delta,  #
            o.numel(), q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            q.shape[0], q.shape[1], q.shape[2],  #
            q.shape[0] * q.shape[1] * q.shape[2],  #
            cdiv(seq_len_kv, BLOCK) * q.shape[0] * q.shape[1] * q.shape[2],  #
            BLOCK_M=BLOCK, BLOCK_N=BLOCK,  #
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,  #
            SEQUENCE_PARALLEL=sequence_parallel,  #
            CAUSAL=ctx.causal,  #
            MMA_V3=MMA_V3,  #
            num_warps=8,  #
            num_stages=1  #
        )

        if len(dq.shape) == 5:
            dq = dq.sum(dim=0)
        return dq, dk, dv, None, None, None


attention = _attention.apply

@pytest.mark.parametrize("Z, H, N_CTX, D_HEAD", [(1, 2, 1024, 64)])
@pytest.mark.parametrize("causal", [True])
def test_op(Z, H, N_CTX, D_HEAD, causal, dtype=torch.float16):
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 0.5
    dout = torch.randn_like(q)
    # reference implementation
    # M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    M = full_mask.to('cuda')
    torch.set_printoptions(profile="full")
    print(f'M: {torch.sum(M)}')
    torch.set_printoptions(profile="default")
    torch.cuda.synchronize()
    ref_start = time.time()
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    # print(f'p: {p.shape}')
    # print(f'self attention[2]: {p[0, 0, 2, 2]}')
    # print(f'self attention[3]: {p[0, 0, 3, 3]}')
    # print(f'p max[1018]: {torch.max(p[0, 0, 1018], dim=-1)}')
    # print(f'p max[1019]: {torch.max(p[0, 0, 1019], dim=-1)}')
    # print(f'p[1018]: {p[0, 1, 1018, 1018]}')
    # print(f'p[1019]: {p[0, 1, 1019, 1019]}')
    p = torch.softmax(p.float(), dim=-1).half()
    # print(f'P: {p.shape}')
    # print(f'P: {torch.sum(p, dim=-1)[:, :, :-4]}')
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)
    torch.cuda.synchronize()
    print(f'ref_time: {time.time() - ref_start}')
    # ref_out.backward(dout)
    # ref_dv, v.grad = v.grad.clone(), None
    # ref_dk, k.grad = k.grad.clone(), None
    # ref_dq, q.grad = q.grad.clone(), None
    # triton implementation
    ancestor = unq.to('cuda')
    tri_out = attention(q, k, v, ancestor, mask.to('cuda'), leaf_idxs.to('cuda'), causal, sm_scale).half()
    print(f'ref_out: {ref_out[0, 1, 1016:1020, :4]}')
    print(f'tri_out: {tri_out[0, 1, 1016:1020, :4]}')
    print(f'ref_out shape: {ref_out.shape}')
    print(f'tri_out shape: {tri_out.shape}')
    # print(f'k: {k[0, 0, ancestor, -1]}')
    # tri_out.backward(dout)
    # tri_dv, v.grad = v.grad.clone(), None
    # tri_dk, k.grad = k.grad.clone(), None
    # tri_dq, q.grad = q.grad.clone(), None
    # compare
    assert torch.allclose(ref_out[:, :, :1020], tri_out[:, :, :1020], atol=1e-2, rtol=0)

    # assert torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=0)
    # assert torch.allclose(ref_dk, tri_dk, atol=1e-2, rtol=0)
    # assert torch.allclose(ref_dq, tri_dq, atol=1e-2, rtol=0)


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
        x_vals=[1024],
        line_arg="provider",
        line_vals=["triton"] + (["flash"] if HAS_FLASH else []),
        line_names=["Triton"] + (["Flash-2"] if HAS_FLASH else []),
        styles=[("red", "-"), ("blue", "-")],
        ylabel="ms",
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
    if provider == "triton":
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        # if mode == "fwd" and TORCH_HAS_FP8:
        #     q = q.to(torch.float8_e5m2)
        #     k = k.to(torch.float8_e5m2)
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        sm_scale = 1.3
        # fn = lambda: attention(q, k, v, causal, sm_scale)
        ancestor = unq.to('cuda')
        ancestor_m = mask.to('cuda')
        leaf_i = leaf_idxs.to('cuda')
        fn = lambda: attention(q, k, v, ancestor, ancestor_m, leaf_i, causal, sm_scale)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(qkv, causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    # flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
    flops_per_matmul = 2.0 * BATCH * H * 7171 * D_HEAD
    total_flops = 2 * flops_per_matmul
    # if causal:
    #     total_flops *= 0.5
    # if mode == "bwd":
    #     total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops / ms * 1e-9


# only works on post-Ampere GPUs right now
for _ in range(10):
    test_op(1, 2, 1024, 64, True)
# test_op(1, 2, 1024, 64, True)
bench_flash_attention.run(save_path=".", print_data=True)
# @pytest.mark.parametrize("Z, H, N_CTX, D_HEAD", [(1, 2, 1024, 64)])
# @pytest.mark.parametrize("causal", [True])
# def test_op(Z, H, N_CTX, D_HEAD, causal, dtype=torch.float16):