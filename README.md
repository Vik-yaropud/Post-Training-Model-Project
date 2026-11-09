# Legal Overruling Classifier: Fine-Tuning a Small LLM

Fine-tuning a small open-source language model to recognize when a sentence from a court opinion **overrules a previous case**, and measuring whether fine-tuning beats the base model.

Checking whether a case is still "good law" is a routine step in legal research. A missed overruling means relying on a dead case, so this project focuses on **recall**: how many real overrulings the model catches.

## Status

- [x] Data prep: LegalBench `overruling` task, split 70/10/20 (train/val/test)
- [x] Baseline eval of `qwen2.5:1.5b` on the test set
- [ ] LoRA fine-tuning on the train split
- [ ] Re-evaluation and comparison

**Baseline (test, 479 sentences):** 78.5% accuracy · 87.2% precision · **66.1% recall** · 75.2% F1

The base model is cautious: it rarely raises false alarms but misses about a third of real overrulings, especially ones phrased without the word "overrule" (e.g. "disapprove," "unsound," "was incorrect"). Full results and error analysis are in [NOTES.md](NOTES.md).

## Running it

Requires Python 3.12, [Ollama](https://ollama.com), and an NVIDIA GPU (developed on an RTX 5070).

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull qwen2.5:1.5b

python prepare_data.py   # download and split the data
python evals.py          # score the model on the test set
```

## Data

[LegalBench](https://huggingface.co/datasets/nguha/legalbench) `overruling` task, built from the CaseHOLD dataset (Zheng et al., 2021), licensed CC BY 4.0. Data is downloaded by `prepare_data.py` and not stored in this repo.