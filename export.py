from unsloth import FastLanguageModel

FRACTION = 0.5

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = f"outputs/lora_{int(FRACTION*100)}",
    max_seq_length = 512,
    load_in_4bit = False
)

model.save_pretrained_gguf(
    f"outputs/gguf_{int(FRACTION*100)}",
    tokenizer,
    quantization_method = "q4_k_m",
)