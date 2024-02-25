import math
import torch

DEPTH_MAPPING = {
    2**10: 7,  # 1k
    2**11: 8,  # 2k
    2**12: 9,  # 4k
    2**13: 10, # 8k
    2**14: 11, # 16k
    2**15: 12, # 32k
    2**16: 13, # 64k
    2**17: 14, # 128k
}

MAX_ANCESTOR_MAPPING = {
    2**10: 64,
    2**11: 64,
    2**12: 64,
    2**13: 64,
    2**14: 64,
    2**15: 64,
    2**16: 64,
    2**17: 64,
}


def create_tree(depth=7, topk=4, left_weight_factor=0.5):
    """
    With topk=4 and left_weight_factor=0.5:
        depth 7  = 2**10 - 4 = 1024  - 4 = 1020
        depth 8  = 2**11 - 4 = 2048  - 4 = 2044
        depth 9  = 2**12 - 4 = 4096  - 4 = 4092
        depth 10 = 2**13 - 4 = 8192  - 4 = 8188
        depth 11 = 2**14 - 4 = 16384 - 4 = 16380
        depth 12 = 2**15 - 4 = 32768 - 4 = 32764

    Args:
        depth                               : The depth of the tree
        topk                                : The branching factor of the tree
        left_weight_factor                  : The factor by which the number of nodes decreases at each level

    Returns:
        lineage      [tree_size, depth + 1] : Lineage of each node (without including the node itself)
        level_lookup [tree_size]            : The tree level of each node
    """
    dense_tree_idxs = torch.empty((depth + 2), dtype=torch.long)
    dense_tree_idxs[0] = 0
    dense_tree_idxs[1] = topk
    prev_level_size = topk
    for i in range(2, depth + 2):
        prev_level_size = topk * int(prev_level_size * left_weight_factor)
        dense_tree_idxs[i] = dense_tree_idxs[i - 1] + prev_level_size

    indices = torch.arange(0, dense_tree_idxs[-1])

    levels = (indices.unsqueeze(1) < dense_tree_idxs.unsqueeze(0)).long().argmax(dim=1)
    level_lookup = levels.long() - 1  # Convert to long for use as indices

    # Compute relative indices within their levels (how far from beginning/end of the level)
    relative_idxs = indices - dense_tree_idxs[level_lookup]
    # Compute the parent's relative index
    parent_relative_idxs = torch.floor((relative_idxs) / topk).long()
    # Go up one level and add the parent's relative index to the beginning of the level
    parent_indices = dense_tree_idxs[level_lookup - 1] + parent_relative_idxs

    lineage = torch.ones((indices.shape[0], depth + 1), dtype=torch.long) * -1
    for i in range(topk, indices.shape[0]):
        idx = indices[i]
        parent = parent_indices[idx]
        lineage[idx, level_lookup[parent]] = parent
        for j in range(depth):
            parent = parent_indices[parent]
            if parent == dense_tree_idxs[-1]:
                break
            lineage[idx, level_lookup[parent]] = parent
    
    return lineage, level_lookup

def create_full_attention_mask(lineage):
    """
    Args:
        lineage   [tree_size, depth + 1] : Lineage of each node (without including the node itself)

    Returns:
        full_mask [tree_size, tree_size] : For use in the reference PyTorch tree attention
    """
    full_mask = torch.zeros((lineage.shape[0], lineage.shape[0]), dtype=torch.bool)
    for i in range(lineage.shape[0]):
        full_mask[i, lineage[i]] = 1
        full_mask[i, i] = 1
    # Remove -1 lineage paddings
    full_mask[:, -1] = 0
    # Add back last token's self attention
    full_mask[-1, -1] = 1

    pad_size = 4
    padding = torch.zeros((full_mask.shape[0], pad_size))
    full_mask = torch.cat((full_mask, padding), dim=1)
    padding = torch.zeros((pad_size, full_mask.shape[1]))
    full_mask = torch.cat((full_mask, padding), dim=0)

    return full_mask

def create_fst_attention_kernel_inputs(lineage, level_lookup, block_m=64, max_ancestors=32):
    """
    Returns:
        ancestor_idx  [num_blocks, max_ancestors]          : The unique ancestors of each block
        ancestor_mask [num_blocks, block_m, max_ancestors] : Which nodes attend to which ancestors within their block
        leaf_idx      [num_blocks, block_m]                : The leaf nodes of each block (not necessarily leaves of the full tree)

    """
    ancestor_idx = torch.ones((math.ceil(lineage.shape[0] / block_m), max_ancestors), dtype=torch.long) * -1
    ancestor_mask = torch.zeros((math.ceil(lineage.shape[0] / block_m), block_m, max_ancestors), dtype=torch.bool)
    leaf_idx = torch.ones((math.ceil(lineage.shape[0] / block_m), block_m), dtype=torch.long) * -1

    for i in range(0, lineage.shape[0], block_m):
        chunk_idx = i // block_m
        chunk = lineage[i:i+block_m]
        chunk_unq = torch.unique(chunk.flatten(), dim=0)[1:]
        ancestor_idx[chunk_idx, :chunk_unq.shape[0]] = chunk_unq
        chunk_leaf_idxs = torch.tensor(list(set(torch.arange(i, i+block_m, dtype=torch.long).tolist()).difference(set(chunk_unq.tolist()))), dtype=torch.long)
        leaf_idx[chunk_idx, chunk_leaf_idxs - i] = chunk_leaf_idxs

        for k in range(i, min(i + block_m, lineage.shape[0])):
            chunk[k - (chunk_idx * block_m), level_lookup[k]] = k
        for j in range(chunk.shape[0]):
            ancestor_mask[chunk_idx, j, :chunk_unq.shape[0]] = torch.isin(chunk_unq, chunk[j])
    
    return ancestor_idx, ancestor_mask, leaf_idx