from huggingface_hub import hf_hub_download
import pandas as pd
import os

path = hf_hub_download(repo_id="nguha/legalbench", filename="data/overruling/test.tsv", repo_type="dataset")

df = pd.read_csv(path, sep="\t")

df = df[["answer", "text"]]

df = df.dropna()

df = df.drop_duplicates(subset="text")

df = df.sample(frac=1, random_state=42)

train_end = int(len(df)*0.7)
val_end = int(len(df)*0.8)

train_split = df.iloc[:train_end]
val_split = df.iloc[train_end:val_end]
test_split = df.iloc[val_end:]

os.makedirs("data/overruling", exist_ok=True)

train_split.to_json("data/overruling/train.jsonl", orient="records", lines=True)
val_split.to_json("data/overruling/val.jsonl", orient="records", lines=True)
test_split.to_json("data/overruling/test.jsonl", orient="records", lines=True)