# Project Notes

Working log for fine-tuning a small open model on a legal classification task. Newest experiments go at the bottom of the experiment log.

---

## 1. Project overview

**Goal:** Fine-tune a small open-source LLM to decide whether a sentence from a court opinion overrules a previous case, and measure whether fine-tuning improves on the base model.

**Task:** LegalBench `overruling`. Binary classification: Yes (the sentence overrules a prior case) or No.

**Why it matters:** Checking that a case hasn't been overruled ("is it still good law?") is a routine step in legal research. A missed overruling means relying on a dead case, so **recall on the Yes class is the metric that matters most.**

**Data source:** [LegalBench](https://huggingface.co/datasets/nguha/legalbench) (`nguha/legalbench`, file `data/overruling/test.tsv`). Original sentences come from the CaseHOLD dataset (Zheng et al., 2021), licensed CC BY 4.0. Data is downloaded by script and not committed to the repo.

**Headline result:** fine-tuning raised test accuracy from **78.5% to 97.9%** and recall from **66.1% to 97.0%**. Errors dropped from 103 to 10 out of 479. A learning-curve experiment showed most of the gain comes from the first ~400 training examples, and accuracy levels off around ~800.

**Scripts**

| File | Purpose | Runs on |
|---|---|---|
| `prompts.py` | The single shared copy of the prompt (`build_prompt`) | Imported by both sides |
| `prepare_data.py` | Download, clean, and split the data | Windows or WSL |
| `evals.py` | Score a model in Ollama on a split | Windows (Ollama) |
| `format_data.py` | Turn train/val splits into chat-format training files | WSL |
| `train.py` | LoRA fine-tune with Unsloth | WSL (GPU) |
| `export.py` | Merge the adapter and export a 4-bit GGUF for Ollama | WSL |

---

## 2. Environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 5070 (12 GB VRAM), desktop |
| OS | Windows (evaluation) + WSL2 Ubuntu (training) |
| Python | 3.12 in a separate venv on each side (WSL venv created with `uv`) |
| Model server | Ollama on Windows, `http://127.0.0.1:11434` |
| Base model | `qwen2.5:1.5b` in Ollama; `unsloth/Qwen2.5-1.5B-Instruct` for training |
| Training stack | Unsloth 2026.11.4, PyTorch 2.14.1 (CUDA 13.2), Transformers 5.17.0, Triton 3.8.0 |
| Precision | bfloat16 |

xformers is installed but doesn't match this PyTorch version, so Unsloth falls back to PyTorch's built-in attention. Training works normally without it.

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

**Reproducibility:** rerunning gives identical counts, including across operating systems. The split regenerated in WSL (Linux) matched the Windows split exactly.

---

## 4. Evaluation method (`evals.py`)

**Prompt** (must be identical in training data and eval), defined once in `prompts.py` as `build_prompt(sentence)`:
```
Does the sentence:"{sentence}" overrule a previous court case? One word answer: Yes or No
```

**Settings at the top of the script:** `MODEL` picks the Ollama model to score; `SPLIT` picks the data file (`"test"` or `"val"` → `data/overruling/{SPLIT}.jsonl`).

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

## 6. Error analysis: baseline

**False negatives (missed overrulings), observations:**

**1. Overruling without the word "overrule" (model error).**
The model catches the verb "overrule" but misses sentences that judge a prior case instead: "disapprove," "disavow," "unsound," "was incorrect," "no longer controlling," "null and void." Partial overrulings phrased as "to the extent [case] holds..., it is disapproved" are also missed. This is the main target for fine-tuning.

**2. Dataset definition is broader than the prompt.**
Sentences noting a case was previously overruled, while the court declines to reinstate it, are labeled Yes. The model answers No, reading them as history. Its reading is defensible; the labels define overruling more broadly than the prompt does.

**3. Data quality issues.**
- Some rows are annotator notes, not court sentences. They start with `##` or `note:`, e.g. `##bauguess v. paine is not ovveruled...`, labeled Yes despite saying the opposite.
- Some sentences are cut off before the overruling clause, e.g. `to the extent that rodgers holds that...`. The deciding words are missing.

These rows were kept in the test set, so some baseline errors are label noise, not model mistakes.

**False positives (false alarms), observations:**

**4. Keyword trap.**
Words that suggest undoing or invalidating something push the model toward Yes without it checking what they refer to: "this assignment of error is accordingly overruled" (rejecting an argument, not a case), a contract described as "wholly void," a motion to "modify" or "terminate" alimony. The effect is inconsistent: the same word doesn't always trigger Yes.

**5. Citing or discussing precedent.**
Sentences that cite or explain prior cases, from bare citations like `state v. schaller, 975 s.w.2d 313...` to "see Terry v. Ohio (holding that...)", are sometimes predicted Yes. The model seems to confuse referring to a case with overruling it.

**Overall:** the model reacts to surface signals (specific words, case names, legal-sounding language) rather than to what the sentence actually does to a prior case.

---

## 7. Training data format (`format_data.py`)

Each train and val row becomes one conversation: the exact eval prompt as the user message and the correct label as the assistant reply.

```json
{"messages": [{"role": "user", "content": "Does the sentence:\"...\" overrule a previous court case? One word answer: Yes or No"},
              {"role": "assistant", "content": "Yes"}]}
```

| File | Rows |
|---|---|
| `train_chat.jsonl` | 1,675 |
| `val_chat.jsonl` | 240 |

The test split is **not** converted. It's only used for evaluation.

**Chat template:** `train.py` converts each conversation to Qwen's text format with `tokenizer.apply_chat_template`. Qwen's template adds a default system message ("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.") when none is given. Ollama's Qwen template adds the same line, so training and evaluation see the same format.

**Shared prompt:** the prompt used to be written separately in `evals.py` and `format_data.py`, so a change to one copy could silently train the model on one question and test it on another. It now lives in `prompts.py`, and both scripts import `build_prompt`. The refactor was verified with `cmp`: `train_chat.jsonl` regenerated after the change is byte-for-byte identical to the file before it.

---

## 8. Fine-tuning (`train.py`)

**Method:** LoRA (small trainable adapter layers on top of the frozen base model), via Unsloth and TRL's `SFTTrainer`.

**Model loading**

| Setting | Value | Why |
|---|---|---|
| Base model | `unsloth/Qwen2.5-1.5B-Instruct` | Same model as the baseline |
| `max_seq_length` | 512 | Sentences are short |
| `load_in_4bit` | `False` | 1.5B fits in 16-bit on 12 GB, so no compression (LoRA, not QLoRA) |

**LoRA settings**

| Setting | Value |
|---|---|
| `r` | 16 |
| `lora_alpha` | 16 |
| `target_modules` | `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention) + `gate_proj`, `up_proj`, `down_proj` (MLP), in all 28 layers |
| `lora_dropout` | 0 |
| `bias` | `"none"` |
| `random_state` | 42 |
| Trainable parameters | ≈18.5M, about **1.2%** of the model |

**Trainer settings (`SFTConfig`)**

| Setting | Value |
|---|---|
| `per_device_train_batch_size` | 8 |
| `num_train_epochs` | 1 |
| `learning_rate` | 2e-4 (decays toward 0 over training) |
| `max_length` | 512 |
| `eval_strategy` / `eval_steps` | every 50 steps |
| `logging_steps` | 10 |
| `seed` | 42 |
| Gradient accumulation | 2 (set automatically by Unsloth), so each step updates on 8 × 2 = **16 examples** |

**`FRACTION` setting:** the share of the training set to use (`1.0` = all 1,675 rows). `train.py` keeps the first `int(len(train) × FRACTION)` rows with `.select(range(n))`. Since the data was shuffled in `prepare_data.py`, the first rows are a random sample, and each smaller run's data is contained in the bigger run's. Output folders are named after the fraction (`outputs/run_25`, `outputs/lora_25`), and `export.py` uses the same `FRACTION` to read the right adapter and write `outputs/gguf_25`. Validation always uses all 240 rows.

**Loss on answers only:** `train_on_responses_only` masks everything before `<|im_start|>assistant\n`. The model reads the full conversation but is only graded on the "Yes"/"No" answer. Without this, about 97% of the loss would come from predicting the prompt text.

### Training run 1

**Date:** 2026-11-15 · **Runtime:** 1 min 43 s · **Steps:** 105

| Point | Training loss | Validation loss |
|---|---|---|
| Step 10 | 0.285 | |
| Step 50 | ~0.11 | 0.110 |
| Step 100 | ~0.02–0.05 | 0.043 |
| End | (average 0.107) | **0.043** |

**Interpretation**
- Both losses decreased, and validation loss kept improving through the end of the epoch, with no sign of overfitting.
- Expected 210 steps (1,675 ÷ 8) but got 105. Unsloth set gradient accumulation to 2, so each step processes 16 examples (1,675 ÷ 16 ≈ 105). All 1,675 examples were still seen exactly once.
- Low loss is not the result. The real comparison is `evals.py` on the test set.

**Output:** LoRA adapters saved to `outputs/lora` (not in Git).

---

## 9. Export to Ollama (`export.py`)

1. `export.py` loads the adapter from `outputs/lora`, merges it into the base model, and calls `save_pretrained_gguf` with `quantization_method = "q4_k_m"`.
2. Output: the merged 16-bit model in `outputs/gguf/`, and the 4-bit file in **`outputs/gguf_gguf/`** (`Qwen2.5-1.5B-Instruct.Q4_K_M.gguf`).
3. Copied the `.gguf` to Windows as `C:\Users\user\ollama-models\qwen-overruling.Q4_K_M.gguf`.
4. Wrote a `Modelfile` next to it:
   - `FROM ./qwen-overruling.Q4_K_M.gguf`
   - `TEMPLATE`: copied from `ollama show qwen2.5:1.5b --modelfile`, so the chat format matches training
   - `SYSTEM You are Qwen, created by Alibaba Cloud. You are a helpful assistant.`, the same default system message the training text contained
5. Registered it with `ollama create qwen-overruling -f Modelfile`.
6. In `evals.py`, set `MODEL = "qwen-overruling"`.

**Why Q4_K_M:** the baseline `qwen2.5:1.5b` in Ollama is also Q4_K_M. Using the same compression means the before/after difference comes from fine-tuning, not from quantization.

---

## 10. Fine-tuned results

**Date:** 2026-11-21 · **Model:** `qwen-overruling` (training run 1, Q4_K_M) · **Split:** test (479) · **Runtime:** under 1 minute

| Metric    | Baseline | Fine-tuned | Change |
|---|---|---|---|
| Accuracy  | 78.5% | **97.9%** | +19.4 pts |
| Precision | 87.2% | 98.7% | +11.5 pts |
| Recall    | 66.1% | **97.0%** | +30.9 pts |
| F1        | 75.2% | 97.9% | +22.7 pts |
| Invalid   | 0 | 0 | |
| Total errors | 103 | 10 | −93 |

**Confusion matrix (fine-tuned)**

|                | Predicted Yes | Predicted No |
|---|---|---|
| **Actual Yes** | TP = 229 | FN = 7 |
| **Actual No**  | FP = 3   | TN = 240 |

**Interpretation**
- The goal was met: recall went from 66.1% to 97.0% (misses fell from 80 to 7), and precision **also** went up instead of trading off (false alarms fell from 23 to 3).
- Still 0 Invalid replies: the fine-tuned model kept the one-word answer format.
- Caveat: with 479 test sentences, each error is worth about 0.2 points. Future runs should be compared on val, and close results checked with a significance test (McNemar's) before claiming one beats another.

---

## 11. Error analysis: fine-tuned model (test, 10 errors)

| Group | Count |
|---|---|
| Label noise / missing context | 3 |
| Definition mismatch (overruled by legislature) | 1 |
| Genuine misses | 3 |
| Genuine false alarms | 3 |

- 4 of 10 errors (all misses) come from the data, not the model: annotator notes, fragments, or a broader definition of overruling than the prompt states.
- 6 genuine errors remain, following the same two baseline patterns at a much lower rate: unusual overruling phrasing ("departing from," "reversing its position," "disavow this language" with no case named) and overruling-sounding words used for something else ("he asks us to overrule Maxwell," "we reject Wilder's arguments").
- The baseline's main failure, overruling phrased without the word "overrule," is mostly fixed.
- Since the 4 data-caused errors are all misses, true recall is likely a bit above 97.0%.

---

## 12. Learning curve (val)

**Date:** 2026-11-22 · **Question:** how much training data does the improvement actually need?

**Method:** trained two new models on the first 25% and 50% of the training set (`FRACTION = 0.25`, `0.5`), with every other setting identical to run 1. Exported each to Q4_K_M and registered them in Ollama as `qwen-overruling-25` and `qwen-overruling-50` (copies of the run 1 Modelfile with only the `FROM` line changed). Scored all four points on **val** (240), so the test set stays untouched.

| Model | Train examples | Steps | Train time | Val loss | Acc | Prec | Rec | F1 | FN | FP |
|---|---|---|---|---|---|---|---|---|---|---|
| `qwen2.5:1.5b` (base) | 0 | | | | 83.3% | 92.1% | 74.4% | 82.3% | 32 | 8 |
| `qwen-overruling-25` | 418 | 27 | 33 s | 0.0741 | 93.3% | 93.6% | 93.6% | 93.6% | 8 | 8 |
| `qwen-overruling-50` | 837 | 53 | 55 s | 0.0428 | 96.2% | 99.2% | 93.6% | 96.3% | 8 | 1 |
| `qwen-overruling` (100%) | 1,675 | 105 | 1 min 43 s | 0.0428 | 96.7% | 97.6% | 96.0% | 96.8% | 5 | 3 |

All runs: 0 Invalid replies.

**Interpretation**
- **The first 25% gives most of the gain.** Accuracy +10 points, almost entirely from recall (misses 32 → 8). False alarms didn't change.
- **25% → 50% fixes false alarms** (8 → 1) while recall stays the same. The extra examples taught the model what *isn't* an overruling.
- **50% → 100% is flat.** 3 fewer misses, 2 more false alarms, one more correct sentence overall. On 240 sentences each sentence is worth 0.4 points, so this is within noise. Val loss agrees: 0.0428 at both 50% and 100%.
- **Conclusion:** performance levels off around ~800 examples. More data of the same kind is unlikely to help much. Going further would need cleaner labels (section 11: 4 of 10 test errors come from the data) or a different model, not more rows.
- The 100% model still has the best recall (96.0% vs 93.6%), the metric that matters most here, but the gap is 3 sentences and isn't enough to call it clearly better.
- The base model scores higher on val (83.3%) than on test (78.5%). With only 240 rows, val happens to be slightly easier. That's why every point on the curve is compared on the same val set.

---

## 13. Debugging log

Problems hit during setup and how they were fixed.

| Problem | Cause | Fix |
|---|---|---|
| `Failed building wheel for pyyaml` | Python 3.15 was too new; no prebuilt packages yet | Installed Python 3.12 and rebuilt the venv |
| Venv creation failed with `WinError 2` | Leftover or half-created environment | Recreated the environment cleanly |
| Full eval ran 16+ minutes, GPU near 4% | `localhost` resolved to IPv6 `::1` first; Ollama only listens on IPv4, so every request waited ~2 s before falling back | Changed URL to `127.0.0.1`. Full run now under 1 minute |
| Git push rejected | GitHub repo was created with its own files | Pulled with `--allow-unrelated-histories --no-rebase`, resolved `.gitignore` conflict |
| WSL install failed (`HCS_E_HYPERV_NOT_INSTALLED`) | CPU virtualization disabled in BIOS | Enabled virtualization in BIOS, reinstalled Ubuntu |
| `format_data.py` not found in WSL | File was saved in the Windows copy of the repo | Work only in the WSL copy; sync through Git |
| ~30 untracked Unsloth files | Unsloth generates `unsloth_compiled_cache/` on import | Added it (plus `outputs/`, `*.gguf`) to `.gitignore` |
| `git push` from WSL asked for a password | WSL has no GitHub login | `gh auth login` with the GitHub CLI |
| `.gitignore` merge conflict | Same file edited in both the Windows and WSL copies | Kept one version; now editing only in WSL and pulling before each session |
| No `.gguf` in `outputs/gguf` after export | Unsloth writes the quantized file to a sibling folder | Found it in `outputs/gguf_gguf/` |
| `ollama show` / `ollama create`: "timed out waiting for server" | Ollama app wasn't running | Started the Ollama app, reran the command |
| `git pull` on Windows: "untracked working tree files would be overwritten" (`format_data.py`) | An old, never-committed copy of `format_data.py` sat in the Windows folder | Deleted the Windows copy (the real one was on GitHub), pulled again |
| `git pull` in WSL: "divergent branches" | An old local notes commit in WSL had never been pushed, while Windows pushed a newer one | Its content was already in the newer file, so `git reset --hard origin/main`; set `git config pull.rebase false` for the future |
| `ollama create`: "no Modelfile or safetensors files found" | Ran from the repo folder; the Modelfile is in `C:\Users\user\ollama-models` | `cd` to that folder first. `FROM ./...` is also relative to it |

---

## 14. Experiment log

One row per run. Use **val** for comparing options; use **test** only for the baseline and final model.

| Date       | Model           | Change from previous       | Split | Acc   | Prec  | Rec   | F1    | Invalid |
|---|---|---|---|---|---|---|---|---|
| 2026-11-08 | qwen2.5:1.5b    | Baseline                   | test  | 78.5% | 87.2% | 66.1% | 75.2% | 0 |
| 2026-11-21 | qwen-overruling | LoRA r=16, 1 epoch, Q4_K_M | test  | 97.9% | 98.7% | 97.0% | 97.9% | 0 |
| 2026-11-22 | qwen2.5:1.5b       | Baseline on val            | val   | 83.3% | 92.1% | 74.4% | 82.3% | 0 |
| 2026-11-22 | qwen-overruling-25 | 25% of train (418)         | val   | 93.3% | 93.6% | 93.6% | 93.6% | 0 |
| 2026-11-22 | qwen-overruling-50 | 50% of train (837)         | val   | 96.2% | 99.2% | 93.6% | 96.3% | 0 |
| 2026-11-22 | qwen-overruling    | 100% of train (1,675)      | val   | 96.7% | 97.6% | 96.0% | 96.8% | 0 |

---

## 15. Next steps

**Done**
- [x] Set up WSL2 + Python 3.12 + Unsloth, confirm GPU works inside WSL
- [x] Convert `train.jsonl` and `val.jsonl` into chat format using the exact prompt above
- [x] LoRA fine-tune Qwen2.5 1.5B on the train split
- [x] Export to GGUF with 4-bit (Q4_K_M) quantization, matching the baseline Ollama model
- [x] Load into Ollama with Qwen's chat template and re-run `evals.py` on test
- [x] Compare against baseline and re-run error analysis on the fine-tuned model
- [x] Move the prompt into a shared `prompts.py` (verified byte-identical with `cmp`)
- [x] Learning-curve experiment: 0 / 25% / 50% / 100% of train, scored on val
- [x] Update README with the fine-tuned results
