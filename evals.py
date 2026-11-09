import requests
import pandas as pd

test = pd.read_json("data/overruling/test.jsonl", orient="records", lines=True)

predictions = []
responses = []
labels = []
count_matches = 0
count_invalid = 0
tp, fp, tn, fn = 0, 0, 0, 0
precision = 0
recall = 0
F1 = 0

def ask(prompt: str) -> str:
    resp = requests.post(
        "http://127.0.0.1:11434/api/chat",
        json={"messages": [{"role": "user", "content": prompt}], "model": "qwen2.5:1.5b", "stream": False, "options": {"temperature": 0, "num_predict": 5}},
        timeout=60)
    data = resp.json()
    answer = data["message"]["content"]
    return answer

def parse_answer(answer: str) -> str:
    answer = answer.strip().lower()
    if answer.startswith("yes"):
        return "Yes"
    elif answer.startswith("no"):
        return "No"
    else:
        return "Invalid"

for i in range(len(test)):
    sentence = test.iloc[i]["text"]
    real_answer = test.iloc[i]["answer"]
    prompt = f"Does the sentence:\"{sentence}\" overrule a previous court case? One word answer: Yes or No"
    answer = ask(prompt)
    labels.append(real_answer)
    responses.append(answer)
    predictions.append(parse_answer(answer))

    if predictions[i] == "Invalid":
        count_invalid += 1
    elif labels[i] == predictions[i]:
        count_matches += 1
        if labels[i] == "Yes":
            tp += 1
        else:
            tn += 1
    else:
        if labels[i] == "Yes":
            fn += 1
        else:
            fp += 1


accuracy = count_matches/len(test)
if tp+fp != 0:
    precision = tp/(tp+fp)
if tp+fn != 0:
    recall = tp/(tp+fn)
if precision+recall != 0:
    F1 = 2*precision*recall/(precision+recall)

for i in range(len(labels)):
    if labels[i] != predictions[i]:
        print(f"#{i+1}\nLabel: {labels[i]}\nPrediction: {predictions[i]}\nRaw response: {responses[i]}\n")

print(f"Accuracy: {accuracy:.1%}")
print(f"Number of invalid responses: {count_invalid}")
print(f"Precision: {precision:.1%}")
print(f"Recall: {recall:.1%}")
print(f"F1: {F1:.1%}")
print(f"TP: {tp}\nFP: {fp}\nFN: {fn}\nTN: {tn}")