# Figures

Presentation charts for the video and the website, made by `scripts/charts.py` from the result files. No number
is typed in by hand: rerun the script whenever `results/` changes.

    uv run python scripts/charts.py                # dark "heart monitor" theme, for the video -> figures/
    uv run python scripts/charts.py --theme light  # light theme, for the website -> site/figures/
    uv run python scripts/charts.py --theme both

The video charts come as SVG (text as outlines, so it renders without the font) and as a 1920x1080 PNG. The
website's charts (`site/figures/`, light) are the same charts without their built-in title and footer, cropped to
the content, since the page has its own headings and captions; the light run also writes
`site/social_preview.png`, the card LinkedIn and others show for a link to the site. Then run
`scripts/site.py` to rebuild the page.

Legend: TSLM models are blue, baselines are grey (dashed for the signals baseline monitor). Health states are
green for stable (60 and up), amber for watch (30-60), orange for low (under 30) and red for critical (15 or
under for 2 steps in a row, which fires the alarm). Each state also has a text label.

| chart | message | sources |
|---|---|---|
| `hero_ecg` | On an unseen failing run (scanpy-2859), ecg-full's health drops through watch and low and the alarm fires at step 12 ("stop the agent"). This happens while the agent re-edits the same file after rejected edits. The signals baseline never alarms on this run. Below: the 19 per-step series the model reads. | `results/opentslm/scan/replay.showcase.{A,signals}.jsonl`, `data/checkpoints.parquet` (series built with `agent_ecg.opentslm_data.time_series`) |
| `failing_vs_resolved` | The same health view for the failing showcase run and a resolved run. ecg-full stays stable on the resolved run until it dips to watch at the end. The signals baseline goes to low early on that resolved run. | `results/opentslm/scan/replay.showcase.{A,signals}.jsonl` |
| `auroc_by_step` | Test AUROC at 5 / 10 / 15 / 20 / 30 steps. Each TSLM stays below the baseline with the same inputs. ecg-full comes within 0.03-0.04 of the best baseline at 5-20 steps. The behaviour signal grows over the run: the signals baseline climbs while issue-only stays flat. | `results/{signals,signals_thoughts,issue_signals_thoughts,issue}.test.json`, `results/opentslm/*/report.test.json` (`figures/runs.json` maps the runs to ECG / ecg-thought / ecg-full, called C / B / A in the code) |
| `monitor_numbers` | As live monitors on 600 replayed test runs, ecg-full and the signals baseline are about as precise (an alarm is right 96 % / 97 % of the time). ecg-full alarms earlier (median step 6 vs 9), the two flag mostly different failing runs. | `results/opentslm/scan/summary.md` |
| `training_curves` | Validation AUROC over training for ECG, ecg-thought and ecg-full. The ringed point is the checkpoint kept. Validation numbers, not test. | `results/opentslm/*/history.json`, `results/opentslm/*/config.json` |

The script fails loudly if a source file changes shape: the summary table can no longer be parsed, a TSLM run is
missing, or the showcase runs are not in the replays.

## Swapping in a second-round run

`figures/runs.json` says which run in `results/opentslm/` fills each TSLM slot (A = ecg-full, B = ecg-thought, C = ecg) and what it is called.
To show a new run, point its slot at the run's directory, give it its own label (for example `A2`), and rerun
`scripts/charts.py --theme both` and `scripts/site.py`. The live-monitor charts and sections read the 600-run scan
in `results/opentslm/scan/`, which was made with ecg-full; they change only when that scan is redone with the new
model.
