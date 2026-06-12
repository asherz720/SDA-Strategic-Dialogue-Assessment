# Strategic Dialogue Assessment (SDA): The Crooked Path to Innocence

This repository accompanies our paper **SDA — Strategic Dialogue Assessment: The
Crooked Path to Innocence**. It provides the **CPD** dataset of annotated
courtroom cross-examinations and a small Python toolkit to score any model's
judgements and compare them against human annotators.

![SDA overview](figure/fig.png)

---

## 🧠 What is SDA?

Language is often used **strategically**, especially in high-stakes, adversarial
settings such as courtroom cross-examination. Most work on LLM pragmatics
assumes **cooperative** discourse; SDA targets the **non-cooperative** case.

SDA scores the strategic impact of each turn through three interpretable metrics:

- **Benefit at Turn** (*BaT*) 📈 — how much a turn advances the witness's position
- **Penalty at Turn** (*PaT*) 📉 — how much it costs the witness
- **Normalized Relative Benefit at Turn** (*NRBaT*) ⚖️ — the net, normalized balance

Each turn is rated on four pragmatic dimensions — **commitment**, **manner**
(clarity), **relevance**, and **quality** (truthfulness) — plus the turn
**outcome** and **consistency**. BaT/PaT/NRBaT are derived from commitment,
manner and relevance; quality is recorded and compared to humans but, by design,
does not enter the metric scores.

---

## 📦 Repository layout

| Path | Description |
|------|-------------|
| **`sda/`** | The toolkit — import this. |
| `sda/schema.py` | **Single source of truth**: required columns and valid value ranges. |
| `sda/validate.py` | Validate a parsed model output; warns on missing columns / out-of-range values. |
| `sda/metrics.py` | Compute **BaT / PaT / NRBaT / NRA**. |
| `sda/agreement.py` | Correlation + inter-rater agreement vs human ratings. |
| `sda/benchmark.py` | Benchmark a model id across the whole dataset (auto-pairs human files). |
| `sda/inference.py` | *Optional* reference for running a model with vLLM. |
| `benchmark.py` | CLI: score a model (or all models) on CPD → leaderboard. |
| `pipeline.py` | CLI for arbitrary file lists: validate → score → correlate. |
| `prompts/` | Prompt templates (`with_constitution`, `without_constitution`, `few_shot`). |
| `tools/normalize_model_annotations.py` | The script that produced the normalized layout (documents the model-id mapping). |
| **Dataset** | `full_collected_set/` (raw dialogues), `human annotations/`, `model_annotations/`. |

The CPD dataset is also on Hugging Face: [UT-CompLing/CPD](https://huggingface.co/datasets/UT-CompLing/CPD).

<details>
<summary>Dataset directory details</summary>

- `full_collected_set/` — source cross-examination transcripts. `*_combined.csv`
  are the full per-case sets; `exp_set/` holds the per-witness experiment splits.
- `human annotations/` — human-rated turns. Files with an `annotator` column hold
  multiple annotators per turn (used as the comparison reference).
- `model_annotations/<case>/<model_id>/<witness>.csv` — model-rated turns in a
  **normalized layout**: 6 cases (`WMT_D`, `WMT_P`, `enron_d`, `enron_p`,
  `simpson_d`, `simpson_p`) × one folder per model id (`gpt-4o-mini`, `qwq-32b`,
  `r1-distill-qwen-7b`, …; prompt ablations as `…__constitution` / `…__few-shot`).
  A model id therefore appears under every case it was run on, which is what lets
  `benchmark.py` evaluate it in one command.

</details>

---

## 🏁 Benchmark a model on CPD

```bash
pip install -r requirements.txt
```

### Run a model that's already in the dataset

```bash
python benchmark.py --list                       # available model ids
python benchmark.py --model gpt-4o-mini           # overall (mean across witnesses)
python benchmark.py --model qwq-32b --level case  # one row per case + an ALL row
python benchmark.py --all --output leaderboard.csv  # one row per model (leaderboard)
```

Each prints the **average correlation/agreement with humans on every metric**
(`bat, pat, nrbat, commitment, relevance, manner, quality, consistency` — see
[Output columns](#output-columns)). Because the committed `model_annotations/`
*are* parsed model outputs, this reproduces any published model's numbers today.

### Add and evaluate your own model

**1. Produce one CSV per witness with your model.** Run inference however you like
(the `prompts/` templates fill `{history}`/`{answer}` per turn; `sda/inference.py`
is a vLLM reference), then parse the model's JSON into these **core columns** —
one row per turn:

| Column | Values | Meaning |
|--------|--------|---------|
| `question` | text | the questioner's turn (merge key against humans) |
| `Commitment_value` | 1–4 | 1 detrimental, 2 beneficial, 3 neutral, 4 none |
| `quality_rate` | 0/1 | 1 truthful, 0 not |
| `consistency_value` | 0/1 | 1 inconsistent, 0 consistent |
| `relevance_rate` | 1–4 | 1 very relevant … 4 irrelevant |
| `manner_rate` | 1–4 | 1 very clear … 4 unclear |
| `outcome_value` | `Questioner` / `Witness` | who won the turn |

These are exactly the keys the prompts ask the model to emit, so a thin JSON parse
is enough. `sda/schema.py` is the authoritative definition; `sda.validate` checks
a file conforms (see [Programmatic API](#programmatic-api)).

**2. Drop the files into the layout**, naming each file exactly like the matching
witness so the human annotations pair automatically:

```
model_annotations/
├── WMT_D/<your-model>/JM_ofshe.csv
├── WMT_P/<your-model>/JM_detective.csv
├── enron_d/<your-model>/enron_defense_1.csv          (and enron_defense_2.csv)
├── enron_p/<your-model>/enron_prosecution_1.csv      (… _2, _3)
├── simpson_d/<your-model>/simpson_defense_5.csv       (… 6, 7, 8)
└── simpson_p/<your-model>/simpson_prosecution_8.csv   (… 9, 12)
```

You can include any subset of cases — the benchmark scores whatever is present.

**3. Run the benchmark on your model:**

```bash
python benchmark.py --model <your-model> --level case --output my_model.csv
```

→ outputs your model's average correlation/agreement per case and overall. Done.

---

<a id="programmatic-api"></a>
## 🚀 Programmatic API (and custom data)

The same steps as functions — useful for data that isn't in the CPD layout, or to
build your own workflow. The core-column contract is the table above;
`sda/schema.py` is authoritative.

### 1. Validate your parsed output

```python
import pandas as pd
from sda import validate

df = pd.read_csv("my_model_annotations.csv")
report = validate(df)          # warns about missing columns / out-of-range values
print(report.summary())
if not report.ok:
    print(report.bad_rows)     # {column: [offending row indices]}
```

`validate` is the single gatekeeper for input format; the scoring code assumes a
table that has passed it.

### 2. Compute the SDA scores

```python
from sda import score

scored = score(df)                       # adds BaT, PaT, NRBaT, NRA per turn
scored.to_csv("scored_turns.csv", index=False)   # per-turn scores as CSV
```

### 3. Correlate with human scores

For a single witness/trial, `compare_human_llm` returns the headline numbers:

```python
from sda import compare_human_llm

humans = pd.read_csv("human annotations/WMT_D_annotations.csv")
results = compare_human_llm(scored, humans)
# {'bat':…, 'pat':…, 'nrbat':…, 'nra':…, 'outcome':…, 'commitment':…,
#  'relevance':…, 'manner':…, 'quality':…, 'consistency':…}
```

Across several witnesses, `evaluate_dataset` returns a tidy CSV-ready table of
the eight reported metrics — one row per witness, or a single mean row:

```python
from sda.agreement import evaluate_dataset

pairs = [("enron_p1", scored1, humans1), ("enron_p2", scored2, humans2)]
evaluate_dataset(pairs, level="witness").to_csv("agreement_by_witness.csv", index=False)
evaluate_dataset(pairs, level="mean")                       # single mean row
```

`bat`/`pat`/`nrbat` are Spearman correlations; `commitment` uses Cohen's kappa,
`relevance`/`manner`/`quality` use free-marginal (Randolph's) kappa, and
`consistency` is a true-positive accuracy — each averaged across the human
annotators. (`compare_human_llm` also returns `outcome` and `nra`, which the
summary tables omit.)

### Or run the whole thing from the command line

Each model CSV is one witness/trial. Pass one or more, with the matching human
file(s), and pick `--level mean` (default) or `--level witness`:

```bash
# one witness
python pipeline.py \
  --model "model_annotations/WMT_D/gpt-4o-mini/JM_ofshe.csv" \
  --human "human annotations/WMT_D_annotations.csv"

# several witnesses -> per-witness rows, written to CSV
python pipeline.py \
  --model model_annotations/enron_p/gpt-4o-mini/enron_prosecution_*.csv \
  --human "human annotations/enron_prosecution_1.csv" \
          "human annotations/enron_prosecution_2.csv" \
          "human annotations/enron_prosecution_3.csv" \
  --level witness --output results.csv --scores-out scored/
```

`--human` may be one file per model or a single shared file; `--scores-out`
optionally dumps each per-turn scored table.

<a id="output-columns"></a>
**Output columns** — the average correlation/agreement with humans on each metric
(one row per witness/case/model, depending on `--level`):

| Group | Columns |
|-------|---------|
| identity | `model` / `case` / `witness` (depending on `--level`), `n_turns` |
| correlation (Spearman) | `bat`, `pat`, `nrbat` |
| agreement (kappa; accuracy for consistency) | `commitment`, `relevance`, `manner`, `quality`, `consistency` |

Each number is averaged across the human annotators (and across witnesses when
aggregated). A `NaN` can appear for a very small or single-category witness file,
where a correlation or kappa is undefined; the mean simply skips those.

> `sda.compare_human_llm` still returns the full detail per call (including
> `outcome` and `nra`); the summary tables report just the eight metrics above.

---

## 📊 Key findings (from the paper)

- LLMs show **limited pragmatic understanding** of strategic, non-cooperative language.
- **Larger models** do better on the SDA metrics.
- **Reasoning ability often hurts** — reasoning models tend to overcomplicate and confuse themselves.

---

## 📄 Citation

```bibtex
@article{zheng2026strategic,
  title={Strategic Dialogue Assessment: The Crooked Path to Innocence},
  author={Zheng, Anshun Asher and Li, Junyi Jessy and Beaver, David I.},
  journal={Dialogue \& Discourse},
  volume={17},
  number={1},
  pages={1--53},
  year={2026}
}
```
