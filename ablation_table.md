### Week 4 Deliverable: Ablation Study Results

| Strategy                     |   faithfulness |   answer_relevancy |   context_recall |   context_precision |
|:-----------------------------|---------------:|-------------------:|-----------------:|--------------------:|
| Hybrid (RRF + Cross-Encoder) |         0.7880 |             0.7632 |           0.7313 |              0.4995 |
| Dense Only (BGE-M3)          |         0.7745 |             0.6959 |           0.6600 |              0.6142 |
| BM25 Only (Sparse)           |         0.7707 |             0.6920 |           0.4600 |              0.2567 |