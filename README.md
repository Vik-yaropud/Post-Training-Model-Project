# Legal fine-tuning project

Fine-tune a small open model on a legal classification task, and measure whether it gets better.

**Task:** LegalBench `overruling`. Given a sentence from a court opinion, answer whether it overrules a previous case (Yes/No). About 2,400 labeled sentences, from the CaseHOLD dataset (CC BY 4.0).

**Model:** Qwen2.5 1.5B Instruct, run locally in Ollama on an RTX 5070.

## The plan

| Weekend | Goal | Result |
|---|---|---|
| 1 (this one) | Setup, data split, baseline score | "The base model gets X%" |
| 2 | LoRA fine-tune on the train split, load into Ollama, re-score | "Fine-tuned gets Y%" |
| 3+ | Change one thing at a time and re-score on val | "Here's what moved the number" |

---

## Weekend 1: setup and baseline (about 3 hours)

Do all of this on the desktop with the 5070, in PowerShell.

### 1. Check the GPU
```powershell
nvidia-smi
```
You should see "NVIDIA GeForce RTX 5070" and about 12 GB of memory. If the command isn't found, install the latest Game Ready or Studio driver from nvidia.com first.

### 2. Install Python and Ollama
- **Python 3.12** from python.org. On the first installer screen, tick **"Add python.exe to PATH"**.
- **Ollama for Windows** from ollama.com/download. It runs in the system tray after installing.

Check both:
```powershell
py -3.12 --version
ollama --version
```

### 3. Set up the project folder
Unzip this kit to `C:\Users\<you>\Projects\legal-finetune`, then:
```powershell
cd $HOME\Projects\legal-finetune
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
If `Activate.ps1` is blocked ("running scripts is disabled"), run this once and try again:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
Your prompt should now start with `(.venv)`. Activate it again each time you open a new terminal.

### 4. Download the model
```powershell
ollama pull qwen2.5:1.5b
ollama run qwen2.5:1.5b "Say hi in five words."
```
While that's answering, run `ollama ps` in a second window. The PROCESSOR column should say **100% GPU**. If it says CPU, your driver needs updating.

### 5. Download and split the data
```powershell
python prepare_data.py
```
This downloads the task from Hugging Face and writes three files to `data\overruling\`:
- `train.jsonl` (70%): what the model learns from in Weekend 2
- `val.jsonl` (10%): what you check while adjusting settings
- `test.jsonl` (20%): only for final numbers. Never train or tune on it.

The split is fixed (random seed 42), so it's the same every time you run it.

If the download fails with an access or login error, open huggingface.co/datasets/nguha/legalbench in a browser while signed in, accept the dataset terms if it shows any, then run `hf auth login` (or `huggingface-cli login` on older versions) with a read token from your Hugging Face settings, and try again.

### 6. Get the baseline score
Quick check first, to make sure everything is wired up:
```powershell
python eval.py --model qwen2.5:1.5b --limit 20
```
Then the real baseline on the full test split (a few minutes on the GPU):
```powershell
python eval.py --model qwen2.5:1.5b --split test
```
You'll get accuracy, precision, recall, F1 and a count of invalid replies. **Write the accuracy and F1 down. That's the number to beat.**

Optional comparison point: run the same command with `qwen2.5:3b` (`ollama pull qwen2.5:3b` first). A bigger untrained model is a useful second baseline.

### 7. Read the mistakes
Open the newest file in `results\` in Excel and filter `correct` to FALSE. Write 3 to 5 short notes in a `NOTES.md` file: what kinds of sentences does it get wrong? Citations? Partial overrulings? Answers that aren't Yes or No?

This step is what turns a script into analysis, and it's what you'll talk about in interviews.

### 8. Put it on GitHub
```powershell
git init
git add .
git commit -m "Baseline eval for LegalBench overruling"
```
Then create an empty repo on GitHub and push to it. The `.gitignore` keeps the data and virtual environment out.

### Done when
- [ ] `nvidia-smi` and `ollama ps` show the GPU in use
- [ ] `data\overruling\` has train, val and test files
- [ ] `results\summary.csv` has your baseline run
- [ ] `NOTES.md` has a few observations about the mistakes

---

## Two things to watch for

**If the baseline is already above about 90%,** there's little room for fine-tuning to show improvement. Switch to a harder task and redo steps 5 and 6:
```powershell
python prepare_data.py --task definition_classification
python eval.py --model qwen2.5:1.5b --task definition_classification
```

**If there are a lot of invalid replies,** the model is answering in sentences instead of Yes/No. Note how many. Fine-tuning usually fixes this, and it's a good thing to point out in your results.

---

## What's in the kit

| File | What it does |
|---|---|
| `prompts.py` | The exact prompt wording. Training and eval must use the same one. |
| `prepare_data.py` | Downloads a LegalBench task and makes the fixed train/val/test split |
| `eval.py` | Sends each example to Ollama, scores the answers, saves results |
| `results/summary.csv` | One row per run, so you can compare runs side by side |

## Weekend 2 preview

Training runs in Linux, so you'll install WSL2 (Ubuntu) and Unsloth inside it, fine-tune `unsloth/Qwen2.5-1.5B-Instruct` with LoRA on `train.jsonl`, export it to GGUF, load it with `ollama create`, and rerun `eval.py` with the new model name. Same script, same test set, so the two numbers are directly comparable.
