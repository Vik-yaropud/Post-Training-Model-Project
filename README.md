# Legal Overruling Classifier: Fine-Tuning a Small LLM

Fine-tuning a small open-source language model to recognize when a sentence from a court opinion **overrules a previous case**, and measuring whether fine-tuning beats the base model.

Checking whether a case is still "good law" is a routine step in legal research. A missed overruling means relying on a dead case, so this project focuses on **recall**: how many real overrulings the model catches.

## Results

Qwen2.5 1.5B Instruct, before and after LoRA fine-tuning, on 479 held-out test sentences:

| Metric | Base model | Fine-tuned |
|---|---|---|
| Accuracy  | 78.5% | **97.9%** |
| Precision | 87.2% | 98.7% |
| Recall    | 66.1% | **97.0%** |
| F1        | 75.2% | 97.9% |
| Errors    | 103 | 10 |

The base model was cautious. It rarely raised false alarms but missed about a third of real overrulings, especially ones phrased without the word "overrule" (e.g. "disapprove," "unsound," "was incorrect"). Fine-tuning cut misses from 80 to 7 and false alarms from 23 to 3.

Of the 10 remaining errors, 4 trace back to the data (annotator notes, cut-off sentences, a broader definition of overruling than the prompt). Full results, training details and error analysis are in [NOTES.md](NOTES.md).

## How it works

1. **Data:** LegalBench `overruling` sentences, split 70/10/20 into train/val/test with a fixed seed.
2. **Baseline:** score the base model in Ollama with a one-word Yes/No prompt.
3. **Fine-tune:** LoRA (r=16, 1 epoch, ~1.2% of parameters trained) with Unsloth, loss on the answer only.
4. **Export:** merge the adapter and quantize to 4-bit GGUF (Q4_K_M), the same compression as the base model in Ollama, so the comparison is fair.
5. **Re-evaluate:** same prompt, same test set, same script.

| Script | Runs on | Purpose |
|---|---|---|
| `prepare_data.py` | Windows or WSL | Download, clean and split the data |
| `evals.py` | Windows | Score a model in Ollama on the test set |
| `format_data.py` | WSL | Turn train/val into chat-format training files |
| `train.py` | WSL | LoRA fine-tune with Unsloth |
| `export.py` | WSL | Merge the adapter and export a 4-bit GGUF |

## Running it

Requires Python 3.12, [Ollama](https://ollama.com), and an NVIDIA GPU (developed on an RTX 5070, 12 GB). Evaluation runs on Windows; training runs in WSL2 Ubuntu because Unsloth targets Linux.

**Evaluate (Windows, PowerShell)**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull qwen2.5:1.5b

python prepare_data.py   # download and split the data
python evals.py          # score the model set in MODEL on the test set
```

**Fine-tune (WSL)**

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install unsloth --torch-backend=auto

python prepare_data.py   # same split as on Windows
python format_data.py    # build chat-format training files
python train.py          # LoRA adapter -> outputs/lora
python export.py         # 4-bit GGUF -> outputs/gguf_gguf/
```

**Load the fine-tuned model into Ollama (Windows)**

Copy the `.gguf` file to Windows, create a `Modelfile` that points to it with Qwen's chat template (`ollama show qwen2.5:1.5b --modelfile`), then:

```powershell
ollama create qwen-overruling -f Modelfile
```

Set `MODEL = "qwen-overruling"` in `evals.py` and run it again.

## Data

[LegalBench](https://huggingface.co/datasets/nguha/legalbench) `overruling` task, built from the CaseHOLD dataset (Zheng et al., 2021), licensed CC BY 4.0. Data is downloaded by `prepare_data.py` and not stored in this repo.
