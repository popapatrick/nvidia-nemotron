# NVIDIA Nemotron Model Reasoning Challenge — LoRA Solutions

Three approaches to the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)
on Kaggle. Each notebook is a self-contained training pipeline that produces a competition-ready
LoRA adapter for `Nemotron-3-Nano-30B-A3B-BF16`. Best **final-leaderboard** score here: **0.83**.

```
.
├── nvidia-notebook-0.83.ipynb   # Teacher chain-of-thought distillation        (best)
├── nvidia-notebook-0.80.ipynb   # Solver-grounded equation engine + teacher CoT
├── nvidia-notebook-0.74.ipynb   # Pure independent solver distillation (no external data)
├── nvidia-accuracy-gate.ipynb   # Standalone evaluator: score any trained adapter
└── README.md
```

Each notebook is standalone — there are no shared Python modules to clone. You pick one, run it
top to bottom on the target GPU, and it writes a `submission.zip`.

`nvidia-accuracy-gate.ipynb` is an optional evaluator: point it at any adapter these notebooks
produce and it reports per-category accuracy on held-out puzzles, so you can sanity-check a
submission before uploading. It loads the adapter with `PeftModel.from_pretrained`, which reads the
adapter's own `adapter_config.json` — so the **same gate works for all three adapters**, regardless
of which LoRA target modules each one trained.

---

## The competition

You are given **in-context rule-discovery puzzles**: each prompt shows a few worked examples of
a hidden transformation, then asks you to apply that same hidden rule to a new query. There are
six categories:

| Category | Example task |
|---|---|
| `bit_manipulation` | Discover a fixed bitwise transform from input→output bit-string pairs |
| `gravity` | Recover a hidden physical/scaling relation from numeric examples |
| `unit_conversion` | Infer a fixed conversion factor and apply it |
| `cipher` | Find the letter-substitution rule from demonstrated word pairs |
| `numeral` | Convert to/from a numeral system (Roman numerals) |
| `equation` | Solve symbol/digit equations where the operators are disguised (cryptarithm-style) |

**The rules of the game:**

- You **may not** change the base model. Everyone fine-tunes the same
  **`Nemotron-3-Nano-30B-A3B-BF16`** (a 30B hybrid Mamba-2 + GQA Transformer mixture-of-experts).
- Your submission is a **LoRA adapter of rank ≤ 32** — the adapter weights plus
  `adapter_config.json`, zipped — not a full model.
- Answers must be returned inside `\boxed{...}`. Scoring is accuracy: exact match for
  binary/string answers, **relative 1% tolerance** for numbers.
- The host evaluates at fixed sampling params: `temperature=0.6`, `top_p=0.95`,
  `max_tokens=8192`. All robustness has to live in the trained weights.

So the whole problem is: **teach a frozen 30B model to be a better in-context rule-inducer, using
only a small rank-32 adapter.**

---

## The three approaches

All three share the same skeleton — load the bf16 base, attach a rank-32 LoRA on the attention +
MLP projections, supervised-fine-tune for one epoch, and package the adapter. They differ entirely
in **where the training reasoning comes from.**

### `nvidia-notebook-0.83.ipynb` — Teacher chain-of-thought distillation 

Fine-tune on **natural chain-of-thought written by strong "teacher" models** (DeepSeek,
GPT-class) from dgxchen/nemotron-cot-tong dataset, solving the real competition puzzles. 
The key trick is **answer-consistency cleaning**:
a teacher trace is kept only when its final boxed answer matches ground truth; numeric traces that
land just short of the required precision get a one-line rounding note, and genuine teacher errors
are dropped. The model learns the teachers' fluent reasoning style and generalises well across all
six categories. This adapter can be downloaded from Hugging Face at **popapatrick/nemotron-3-nano-reasoning-lora**.

> **Why it wins:** natural teacher reasoning transfers better than any templated/mechanical trace —
> the model learns the *skill*, not a fixed output format.

### `nvidia-notebook-0.80.ipynb` — Solver-grounded equation engine + teacher CoT

Same teacher-CoT base, but the **equation / cryptarithm** puzzles are handled by our own
**deterministic deductive solver** instead of the teacher. For those rows the solver recovers the
disguised operator mapping, generates a short *verified* derivation, and that replaces the teacher's
(much longer) trace. This keeps natural CoT everywhere it helps while making the hardest category
exact and compact.

### `nvidia-notebook-0.74.ipynb` — Pure independent solver distillation

**No external data at all** — trained from the competition's `train.csv` only. Per-category
**deterministic solvers** recover each hidden rule directly from the in-context demonstrations,
emit a mechanical step-by-step chain-of-thought, and we train on that. Every trace is verified
against ground truth and round-tripped through `\boxed{}`, so the corpus has **zero label noise**.
It is the fully self-contained, reproducible-from-scratch solution — and on the final leaderboard
it lands surprisingly close to the teacher-distillation runs.

---

## Results

| Notebook | Approach | Public LB | Final LB |
|---|---|---|---|
| `nvidia-notebook-0.83.ipynb` | Teacher chain-of-thought distillation | 0.84 | **0.83** |
| `nvidia-notebook-0.80.ipynb` | Solver-grounded equation engine + teacher CoT | 0.80 | 0.80 |
| `nvidia-notebook-0.74.ipynb` | Pure independent solver distillation (no external data) | 0.68 | 0.74 |

**What the final scores show:** natural teacher chain-of-thought is still the strongest signal
(0.83), but the gap to the fully independent, zero-external-data solver approach is small (0.74).
The solver corpus is verified and label-noise-free, and it actually **generalised upward** from the
public to the final leaderboard (0.68 → 0.74) while the teacher run slipped slightly (0.84 → 0.83) —
a sign the verified corpus was less overfit. Swapping our own mechanical equation traces into the
teacher run (the 0.80 hybrid) didn't help over pure teacher CoT: even when provably correct, a rigid
trace template transfers a little worse than the teacher's natural reasoning.

---

## Environment & dependencies

These notebooks were trained on a single **NVIDIA RTX PRO 6000 (96 GB)** — the bf16 base (~60 GB)
fits on one GPU with no quantization.

They assume the Nemotron runtime libraries (`mamba_ssm`, `causal_conv1d`, `unsloth`, `peft`, `trl`,
`kagglehub`) are already importable. In the actual competition run the provided GPU box had
**restricted internet access**, so these were installed up front from **pre-downloaded offline
wheels** (built for the exact CUDA / PyTorch version) rather than from PyPI. That offline-install
step is environment-specific and has been left out of the published notebooks for clarity — if you
reproduce this with normal internet, a standard `pip install` of those packages before running is
enough.

---

## Reproducing a run

Each notebook is built for a single ~96 GB GPU session; one run is ~3–4 hours.

1. Make the runtime libraries available (see **Environment & dependencies** above), then open the
   notebook and attach its inputs:
   - the **competition dataset** (`nvidia-nemotron-model-reasoning-challenge`);
   - the **base model** `Nemotron-3-Nano-30B-A3B-BF16` (added via Kaggle Models / `kagglehub`);
   - **for the 0.83 and 0.80 notebooks only**, a public **teacher chain-of-thought dataset** for
     this competition (e.g. `dgxchen/nemotron-cot-tong`). The 0.74 notebook needs no extra data —
     it generates its entire training set from `train.csv`.
2. Run all cells. The notebook loads the base model, builds its training set, fine-tunes the LoRA
   adapter, runs a quick inference check, and writes `submission.zip` to `/kaggle/working`.
3. The final cell trims the working directory to exactly the adapter files + `submission.zip` —
   that zip is what you submit.

To **evaluate** a trained adapter, open `nvidia-accuracy-gate.ipynb`, point its `ADAPTER_CANDIDATES`
at your adapter folder, and run it — it scores the held-out rows at the official sampling params and
prints per-category accuracy.

The code paths auto-resolve common Kaggle input locations and fall back to the repo root, so the
notebooks also run locally if you have the model and data on disk.

---

## Notes & attribution

- The teacher chain-of-thought used by the **0.83** and **0.80** notebooks comes from a publicly
  available, permissively licensed teacher-CoT dataset for this competition (dgxchen/nemotron-cot-tong).
  If you reuse thosenotebooks, credit the upstream dataset and keep its license.
- The **0.74** notebook uses **no external data** — its corpus is generated entirely from the
  competition's own `train.csv`, so it is fully reproducible from the competition materials alone.
- The base model and competition data are subject to NVIDIA's and Kaggle's respective terms.
