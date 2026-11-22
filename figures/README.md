# Figures

Presentation charts for the video and the website, made by `scripts/charts.py` from the result files. No number
is typed in by hand: rerun the script whenever `results/` changes.

    uv run python scripts/charts.py                # dark "heart monitor" theme -> figures/
    uv run python scripts/charts.py --theme light  # light theme for the website -> figures/light/
    uv run python scripts/charts.py --theme both

Each chart comes as SVG (text as outlines, so it renders without the font) and as a 1920x1080 PNG.

Legend: TSLM models are blue, CPU models are grey (dashed for the CPU signals monitor). Health states are
green for stable (60 and up), amber for watch (30-60), orange for low (under 30) and red for critical (15 or
under for 2 steps in a row, which fires the alarm). Each state also has a text label.

| chart | message | sources |
|---|---|---|
| `hero_ecg` | On an unseen failing run (scanpy-2859), TSLM A's health drops through watch and low and the alarm fires at step 12 ("stop the agent"). This happens while the agent re-edits the same file after rejected edits. The CPU signals monitor never alarms on this run. Below: the 19 per-step series the model reads. | `results/opentslm/scan/replay.showcase.{A,signals}.jsonl`, `data/checkpoints.parquet` (series built with `agent_ecg.opentslm_data.time_series`) |
| `failing_vs_resolved` | The same health view for the failing showcase run and a resolved run. A stays stable on the resolved run until it dips to watch at the end. The CPU signals monitor goes to low early on that resolved run. | `results/opentslm/scan/replay.showcase.{A,signals}.jsonl` |
| `auroc_by_step` | Test AUROC at 5 / 10 / 15 / 20 / 30 steps. Each TSLM stays below the CPU model with the same inputs. A comes within 0.03-0.04 of the best CPU model at 5-20 steps. The behaviour signal grows over the run: CPU signals climbs while issue-only stays flat. | `results/{signals,signals_thoughts,issue_signals_thoughts,issue}.test.json`, `results/opentslm/*/report.test.json` (`config.json` maps the runs to A / B / C) |
| `monitor_numbers` | As live monitors on 600 replayed test runs, A and the CPU signals monitor are about as precise (an alarm is right 96 % / 97 % of the time). A alarms earlier (median step 6 vs 9), and the two flag mostly different failing runs. | `results/opentslm/scan/summary.md` |
| `training_curves` | Validation AUROC over training for A, B and C. The ringed point is the checkpoint kept. Validation numbers, not test. | `results/opentslm/*/history.json`, `results/opentslm/*/config.json` |

The script fails loudly if a source file changes shape: the summary table can no longer be parsed, a TSLM run is
missing, or the showcase runs are not in the replays.
