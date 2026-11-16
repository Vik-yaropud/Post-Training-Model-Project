from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

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

trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = ds["train"],
    eval_dataset = ds["validation"],
    args = SFTConfig(
        dataset_text_field = "text",
        max_length = 512,
        per_device_train_batch_size = 8,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        logging_steps = 10,
        eval_strategy = "steps",
        eval_steps = 50,
        output_dir = "outputs",
        seed = 42,
        report_to = "none"
    )
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n",
)

trainer.train()

model.save_pretrained("outputs/lora")
tokenizer.save_pretrained("outputs/lora")