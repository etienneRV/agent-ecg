# Shortcut checks

Could a model score well on test by reading a cue other than the agent's behaviour? Three candidates.

**1. Which agent LLM wrote the run: ruled out.** Test runs come from three agents (`swe-agent-llama-70b`
7,098 runs, 16.8 % resolved; `-8b` 446, 8.7 %; `-405b` 96, 21.9 %), so a model could rank runs by the
writing style of their thoughts. AUROC computed within each agent (weighted by checkpoints) is within
0.012 of the overall AUROC for every model, and replacing each run's score by its agent's mean score gives
an AUROC of 0.48-0.52 for the three TSLM runs. Test predictions: `results/opentslm/*/` and `results/baselines/`.

| model | AUROC 5 / 10 / 15 / 20 / 30, all runs | within each agent model |
|---|---|---|
| ecg (TSLM) | 0.583 / 0.611 / 0.616 / 0.620 / 0.615 | 0.586 / 0.613 / 0.617 / 0.618 / 0.614 |
| ecg-thought (TSLM) | 0.624 / 0.638 / 0.643 / 0.647 / 0.596 | 0.619 / 0.630 / 0.634 / 0.640 / 0.599 |
| ecg-full (TSLM) | 0.657 / 0.677 / 0.684 / 0.679 / 0.624 | 0.652 / 0.668 / 0.675 / 0.670 / 0.629 |
| signals (baseline) | 0.611 / 0.645 / 0.674 / 0.712 / 0.728 | 0.613 / 0.644 / 0.668 / 0.704 / 0.726 |
| thoughts (baseline) | 0.680 / 0.703 / 0.710 / 0.701 / 0.701 | 0.678 / 0.702 / 0.706 / 0.695 / 0.696 |
| signals+thoughts (baseline) | 0.672 / 0.692 / 0.710 / 0.712 / 0.699 | 0.671 / 0.688 / 0.703 / 0.700 / 0.691 |
| issue+signals+thoughts (baseline) | 0.691 / 0.709 / 0.720 / 0.719 / 0.701 | 0.692 / 0.707 / 0.713 / 0.708 / 0.691 |
| issue (baseline) | 0.613 / 0.619 / 0.612 / 0.616 / 0.614 | 0.613 / 0.620 / 0.611 / 0.614 / 0.609 |

With 93 % of runs from one agent, this rules out inflation here; it says nothing about transfer to a
different agent or scaffold.

**2. How hard the issue is: real, partly.** Models that read the issue can rank runs by issue difficulty.
Test repositories are never seen in training, so no issue can be memorised, and same-issue pairs (both
runs share the issue) isolate behaviour: ecg-full scores 0.596 at 10 steps, above chance. But as a live monitor
half of ecg-full's alarms fire at the earliest possible step and cluster by issue
([opentslm/scan/summary.md](opentslm/scan/summary.md)).

**3. How long the run has lasted: handled by per-checkpoint scores.** Long runs mostly fail (96 % still
going at step 30 fail), and the series length shows the step. Every per-checkpoint AUROC compares runs of
the same length; pooled AUROCs benefit from the cue, so quote per-checkpoint numbers.
