# Why early prediction is hard here, and what remembering earlier attempts gains

Computed by `scripts/history_monitor.py` from `data/checkpoints.parquet` and `results/`.

## Where the outcome's variance lives

- 76,182 runs on 3,586 issues (15 attempts per issue at the median, 21.2 on average).
- 77% of issues are never solved in any attempt, 1% always, 22% sometimes.
- Which issue it is explains 52% of the variance in outcomes (overall success 16.6%).
- Training split: 60,985 runs but 2,853 issues, of which 659 were ever solved.

Leave-one-out success rate of the other attempts at the same issue, as a score (test AUROC):

| steps seen | 5 | 10 | 15 | 20 | 30 |
|---|---|---|---|---|---|
| sibling success rate | 0.930 | 0.922 | 0.908 | 0.905 | 0.870 |
| our model (ecg-full) | 0.657 | 0.677 | 0.684 | 0.679 | 0.624 |

Chance an attempt succeeds, given that m other attempts at the same issue (drawn at random) all failed:

| m | 0 | 1 | 3 | 5 |
|---|---|---|---|---|
| chance of success | 16.6 % | 8.1 % | 3.4 % | 2.0 % |

## A monitor that remembers earlier attempts (test AUROC)

For each test run, m earlier attempts are drawn at random among the other attempts at the same issue (seed 0); their success rate is combined with the model's chance of success by a logistic regression per checkpoint, cross-fitted over two halves of the test repositories.

| model | 5 | 10 | 15 | 20 | 30 |
|---|---|---|---|---|---|
| our model alone | 0.657 | 0.677 | 0.684 | 0.679 | 0.624 |
| 1 earlier attempt alone | 0.729 | 0.718 | 0.700 | 0.696 | 0.727 |
| 3 earlier attempts alone | 0.848 | 0.834 | 0.820 | 0.821 | 0.822 |
| 5 earlier attempts alone | 0.879 | 0.866 | 0.851 | 0.842 | 0.865 |
| our model + 1 earlier attempt | 0.759 | 0.772 | 0.749 | 0.759 | 0.701 |
| our model + 3 earlier attempts | 0.852 | 0.843 | 0.833 | 0.841 | 0.816 |
| our model + 5 earlier attempts | 0.883 | 0.878 | 0.867 | 0.865 | 0.864 |
| simple model (same inputs) + 3 earlier attempts | 0.859 | 0.852 | 0.844 | 0.853 | 0.853 |

Gain of our model + 3 earlier attempts over our model alone (paired repository bootstrap; bold: the interval excludes zero):

| steps seen | 5 | 10 | 15 | 20 | 30 |
|---|---|---|---|---|---|
| gain | **+0.195 [+0.139, +0.250]** | **+0.165 [+0.113, +0.220]** | **+0.149 [+0.093, +0.204]** | **+0.163 [+0.106, +0.224]** | **+0.193 [+0.118, +0.266]** |
