# dependencies:
# pytorch, pandas, tensorflow, transformers, datasets

# pip insatall transformers
# pip install datasets

import os
import json
import random
import datasets
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import tensorflow as tf
import pandas as pd
from numpy import savetxt

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

ds = datasets.load_dataset("csv", data_files="data/train.csv")
tokenized_samples = ds.map(preprocess_function, batched=True)

ds_test = datasets.load_dataset("csv", data_files="../../test.csv")
tokenized_samples_test = ds_test.map(preprocess_function, batched=True)

os.environ["WANDB_DISABLED"] = "true"

try:
    os.mkdir("results")
except:
    print('results dir exists')

training_args = TrainingArguments(
    output_dir="results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_samples["train"],
    eval_dataset=tokenized_samples_test["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()


x = [e for e in ds_test["train"]["text"]]
y = [e for e in ds_test["train"]["label"]]

def classify(prompt:str):
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    return float(tf.nn.softmax(logits)[0][1])

a=0
y_received=[]
for index in range(len(x)):
    i=x[index]
    y_received.append(classify(i))
    if int(index/len(x)*100)>a:
        print(index/len(x)*100,flush=True)
        a=int(index/len(x)*100)
savetxt('all_results.txt',y_received)
print("COMPLETED")