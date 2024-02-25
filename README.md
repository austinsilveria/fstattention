# Fused Sparse Tree Attention

Memory bandwidth efficient sparse tree attention

- (precompute) chunk the tree into query blocks
- (precompute) compute unique ancestors, attention mask, and leaves for each block
- (runtime) only load keys and values for the query block's unique ancestors and leaves
- (runtime) go fast
- [A100 Colab Benchmark](https://colab.research.google.com/drive/12MvLU5TUMAARQAUVYN2VL0u24A8MwsFI?usp=sharing)

go forth, search the tree of possible futures

<img width="583" alt="Screen Shot 2024-02-25 at 14 52 59" src="https://github.com/austinsilveria/fstattention/assets/26588632/f793cd9b-4bf2-48ea-93e3-c0819422a1a4">

Notes on precomputation:
  - Can probably make this fast enough for runtime with a bit more work since for a dynamic tree structure (i.e. dependent on the model's output), we only need to compute these kernel inputs once, and then they get reused by all attention layers in the model
  - Static tree structures are still useful: [Medusa](https://arxiv.org/pdf/2401.10774.pdf) uses a size 256 static left weighted tree that gets populated via cartesian products of their multiple topk output heads to accelerate batch size 1 inference by ~3x

Credits:
- [OpenAI kernel team's flash 2 triton implementation](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [Original flash attention paper](https://arxiv.org/abs/2205.14135)
- [Flash 2 paper](https://tridao.me/publications/flash2/flash2.pdf)
- [Rabe and Staats](https://arxiv.org/pdf/2112.05682v2.pdf)
- [Medusa paper](https://arxiv.org/pdf/2401.10774.pdf)
