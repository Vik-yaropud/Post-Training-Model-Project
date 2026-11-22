from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "outputs/lora",
    max_seq_length = 512,
    load_in_4bit = False
)

model.save_pretrained_gguf(
    "outputs/gguf",
    tokenizer,
    quantization_method = "q4_k_m",
)