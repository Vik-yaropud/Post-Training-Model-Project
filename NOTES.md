# Project Notes

Working log for fine-tuning a small open model on a legal classification task. Newest experiments go at the bottom of the experiment log.

---

## 1. Project overview

**Goal:** Fine-tune a small open-source LLM to decide whether a sentence from a court opinion overrules a previous case, and measure whether fine-tuning improves on the base model.

**Task:** LegalBench `overruling`. Binary classification: Yes (the sentence overrules a prior case) or No.

**Why it matters:** Checking that a case hasn't been overruled ("is it still good law?") is a routine step in legal research. A missed overruling means relying on a dead case, so **recall on the Yes class is the metric that matters most.**

**Data source:** [LegalBench](https://huggingface.co/datasets/nguha/legalbench) (`nguha/legalbench`, file `data/overruling/test.tsv`). Original sentences come from the CaseHOLD dataset (Zheng et al., 2021), licensed CC BY 4.0. Data is downloaded by script and not committed to the repo.

---

## 2. Environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 5070 (12 GB VRAM), desktop |
| OS | Windows (training planned in WSL2) |
| Python | 3.12.10 in a project venv |
| Model server | Ollama, local, `http://127.0.0.1:11434` |
| Base model | `qwen2.5:1.5b` (Qwen2.5 1.5B Instruct) |

---

## 3. Data preparation (`prepare_data.py`)

- LegalBench's own `train.tsv` for this task has only a handful of few-shot examples, so I used its `test.tsv` (2,394 rows) as the raw data and made my own splits.
- Columns in the file: `index`, `answer`, `text`. Kept `answer` and `text`.
- Cleaning: dropped missing values (0 found) and duplicate sentences (0 found). Kept the duplicate check to prevent data leakage between splits.
- Label balance: 1,216 Yes / 1,178 No (about 51/49). An "always Yes" model would score about 51%.
- Shuffled with `random_state=42`, then split 70 / 10 / 20:

| Split | Rows  | Yes | No  | Used for |
|---|---|---|---|---|
| train | 1,675 | 855 | 820 | Fine-tuning |
| val   | 240   | 125 | 115 | Comparing prompts and training settings |
| test  | 479   | 236 | 243 | Baseline and final score only |

Split is reproducible: rerunning gives identical counts.

---

## 4. Evaluation method (`evals.py`)

**Prompt** (must be identical in training data and eval):
```
Does the sentence:"{sentence}" overrule a previous court case? One word answer: Yes or No
```

**Generation settings:** `temperature = 0` (deterministic), `num_predict = 5` (caps reply length), request `timeout = 60` seconds.

**Parsing:** reply is stripped and lowercased. Starts with "yes" → `Yes`, starts with "no" → `No`, anything else → `Invalid`.

**Metrics:** accuracy, precision, recall, F1, with **Yes as the positive class**.

**Design decision on Invalid replies:** Invalid replies are counted separately and **excluded from the confusion matrix**. Precision, recall and F1 are therefore computed on valid replies only. Accuracy still uses all rows (Invalid counts as wrong).
Always report the Invalid count next to these metrics, since a model with many Invalid replies could otherwise show inflated recall.

---

## 5. Baseline results

**Date:** 2026-11-08 · **Model:** `qwen2.5:1.5b` (no fine-tuning) · **Split:** test (479) · **Runtime:** under 1 minute

| Metric    | Value |
|---|---|
| Accuracy  | **78.5%** |
| Precision | 87.2% |
| Recall    | **66.1%** |
| F1        | 75.2% |
| Invalid   | 0 |

**Confusion matrix**

|                | Predicted Yes | Predicted No |
|---|---|---|
| **Actual Yes** | TP = 156 | FN = 80 |
| **Actual No**  | FP = 23  | TN = 220 |

**Interpretation**
- The base model is **cautious**: when it says Yes it's usually right (87% precision), but it misses about a third of real overrulings (80 misses vs. 23 false alarms).
- Misses are the costly error for this use case, so the fine-tuning goal is to **raise recall without giving up much precision.**
- 0 Invalid replies: the format instruction in the prompt works for this model.

---

## 6. Error analysis

**False negatives (missed overrulings), observations:**

**1. Overruling without the word "overrule" (model error).**
The model catches the verb "overrule" but misses sentences that judge a prior case instead: "disapprove," "disavow," "unsound," "was incorrect," "no longer controlling," "null and void." Partial overrulings phrased as "to the extent [case] holds..., it is disapproved" are also missed. This is the main target for fine-tuning.

**2. Dataset definition is broader than the prompt.**
Sentences noting a case was previously overruled, while the court declines to reinstate it, are labeled Yes. The model answers No, reading them as history. Its reading is defensible; the labels define overruling more broadly than the prompt does.

**3. Data quality issues.**
- Some rows are annotator notes, not court sentences. They start with `##`, e.g. `##bauguess v. paine is not ovveruled...`, labeled Yes despite saying the opposite.
- Some sentences are cut off before the overruling clause, e.g. `to the extent that rodgers holds that...`. The deciding words are missing.

These rows were kept in the test set, so some baseline errors are label noise, not model mistakes.

**False positives (false alarms), observations:**

**4. Keyword trap.**
Words that suggest undoing or invalidating something push the model toward Yes without it checking what they refer to: "this assignment of error is accordingly overruled" (rejecting an argument, not a case), a contract described as "wholly void," a motion to "modify" or "terminate" alimony. The effect is inconsistent: the same word doesn't always trigger Yes.

**5. Citing or discussing precedent.**
Sentences that cite or explain prior cases, from bare citations like `state v. schaller, 975 s.w.2d 313...` to "see Terry v. Ohio (holding that...)", are sometimes predicted Yes. The model seems to confuse referring to a case with overruling it.

**Overall:** the model reacts to surface signals (specific words, case names, legal-sounding language) rather than to what the sentence actually does to a prior case.

---

## 7. Debugging log

Problems hit during setup and how they were fixed.

| Problem | Cause | Fix |
|---|---|---|
| `Failed building wheel for pyyaml` | Python 3.15 was too new; no prebuilt packages yet | Installed Python 3.12 and rebuilt the venv |
| Venv creation failed with `WinError 2` | Leftover or half-created environment | Recreated the environment cleanly |
| Full eval ran 16+ minutes, GPU near 4% | `localhost` resolved to IPv6 `::1` first; Ollama only listens on IPv4, so every request waited ~2 s before falling back | Changed URL to `127.0.0.1`. Full run now under 1 minute |
| Git push rejected | GitHub repo was created with its own files | Pulled with `--allow-unrelated-histories --no-rebase`, resolved `.gitignore` conflict |

---

## 8. Experiment log

One row per run. Use **val** for comparing options; use **test** only for the baseline and final model.

| Date       | Model        | Change from previous | Split | Acc   | Prec  | Rec   | F1    | Invalid |
|---|---|---|---|---|---|---|---|---|
| 2026-11-08 | qwen2.5:1.5b | Baseline             | test  | 78.5% | 87.2% | 66.1% | 75.2% | 0 |

---

## 9. Next steps

- [ ] Set up WSL2 + Python 3.12 + Unsloth, confirm GPU works inside WSL
- [ ] Convert `train.jsonl` into chat format using the exact prompt above
- [ ] LoRA fine-tune Qwen2.5 1.5B on the train split
- [ ] Export to GGUF, load into Ollama, re-run `evals.py` on test
- [ ] Compare against baseline, especially recall
