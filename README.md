# Fused Sparse Tree Attention

Memory bandwidth efficient sparse tree attention

- (precompute) chunk the tree into query blocks
- (precompute) compute unique ancestors, attention mask, and leaves for each block
- (runtime) only load keys and values for the query block's unique ancestors and leaves
- (runtime) go fast

go forth, search the tree of possible futures

<img width="583" alt="Screen Shot 2024-02-25 at 13 27 44" src="https://github.com/austinsilveria/fstattention/assets/26588632/ed924ebd-5690-4b6a-84cb-0861ff6fa0fb">

Notes on precomputation:
  - Can probably make this fast enough for runtime with a bit more work since for a dynamic tree structure (i.e. dependent on the model's output), we only need to compute these kernel inputs once, and then they get reused by all attention layers in the model
  - Static tree structures are still very useful: [Medusa](https://arxiv.org/pdf/2401.10774.pdf) uses a static left weighted tree that gets populated via cartesian products of their multiple topk output heads 

Credits:
- [OpenAI kernel team's flash 2 triton implementation](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [Original flash attention paper](https://arxiv.org/abs/2205.14135)
- [Flash 2 paper](https://tridao.me/publications/flash2/flash2.pdf)
- [Rabe and Staats](https://arxiv.org/pdf/2112.05682v2.pdf)
- [Medusa paper](https://arxiv.org/pdf/2401.10774.pdf)
