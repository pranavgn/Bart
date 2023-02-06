# Dependencies: transformers, pandas, torch

from transformers import pipeline
import pandas as pd

# setting up model
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli", device=0)

# abstracts to classify
data = pd.read_csv("data_for_abstract_classification.csv")

# all labels
candidate_labels = ['biology', 'chemistry', 'computer science', 'physics', 'robotics']

output = []

# classifying all sentences using zeroshot
for i in range(len(data)):
    sequence_to_classify = str(data["abstract"][i])
    temp = classifier(sequence_to_classify, candidate_labels)
    temp["actual_classification"] = data["classification"][i]
    for j in range(len(temp["labels"])):
        temp[temp["labels"][j]]=temp["scores"][j]
    temp.pop("scores")
    temp.pop("labels")
    output.append(temp)
    print(i)

# saving output
pd.DataFrame.from_dict(output).to_csv("zeroshot_abstract_classification.csv")