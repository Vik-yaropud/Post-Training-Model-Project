from unsloth import FastLanguageModel
from datasets import load_dataset

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-1.5B-Instruct",
    max_seq_length = 512,
    load_in_4bit = False
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    random_state = 42
)

ds = load_dataset("json", data_files={"train": "data/overruling/train_chat.jsonl", "validation": "data/overruling/val_chat.jsonl"})

def to_text(row):
    messages = row["messages"]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

ds = ds.map(to_text)
print(ds["train"][0]["text"])