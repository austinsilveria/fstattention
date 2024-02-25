# Fused Sparse Tree Attention

Memory bandwidth efficient sparse tree attention

- chunk the tree into query blocks
- only load keys and values for the query block's unique ancestors and leaves
- go fast

go forth, search the tree of possible futures

<img width="583" alt="Screen Shot 2024-02-25 at 13 27 44" src="https://github.com/austinsilveria/fstattention/assets/26588632/ed924ebd-5690-4b6a-84cb-0861ff6fa0fb">
