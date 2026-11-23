import pandas as pd
from prompts import build_prompt

train = pd.read_json("data/overruling/train.jsonl", orient="records", lines=True)
val = pd.read_json("data/overruling/val.jsonl", orient="records", lines=True)

conversations = []

for i in range(len(train)):
    sentence = train.iloc[i]["text"]
    answer = train.iloc[i]["answer"]
    prompt = f"Does the sentence:\"{sentence}\" overrule a previous court case? One word answer: Yes or No"
    conversations.append([{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}])

df = pd.DataFrame({"messages": conversations})

df.to_json("data/overruling/train_chat.jsonl", orient="records", lines=True)

conversations.clear()

for i in range(len(val)):
    sentence = val.iloc[i]["text"]
    answer = val.iloc[i]["answer"]
    prompt = f"Does the sentence:\"{sentence}\" overrule a previous court case? One word answer: Yes or No"
    conversations.append([{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}])

df = pd.DataFrame({"messages": conversations})

df.to_json("data/overruling/val_chat.jsonl", orient="records", lines=True)