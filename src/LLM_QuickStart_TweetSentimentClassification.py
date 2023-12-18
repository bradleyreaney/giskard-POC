import os
import numpy as np
import pandas as pd
from scipy.special import softmax
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datetime import datetime

from giskard import Dataset, Model, scan, testing, GiskardClient, Suite

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

DATASET_CONFIG = {"path": "tweet_eval", "name": "sentiment", "split": "validation"}

LABEL_MAPPING = {0: "negative", 1: "neutral", 2: "positive"}

TEXT_COLUMN = "text"
TARGET_COLUMN = "label"

raw_data = load_dataset(**DATASET_CONFIG).to_pandas().iloc[:500]
raw_data = raw_data.replace({"label": LABEL_MAPPING})

giskard_dataset = Dataset(
    df=raw_data,
    # A pandas.DataFrame that contains the raw data (before all the pre-processing steps) and the actual ground truth variable (target).
    target=TARGET_COLUMN,  # Ground truth variable.
    name="Tweets with sentiment",  # Optional.
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


def prediction_function(df: pd.DataFrame) -> np.ndarray:
    encoded_input = tokenizer(list(df[TEXT_COLUMN]), padding=True, return_tensors="pt")
    output = model(**encoded_input)
    return softmax(output["logits"].detach().numpy(), axis=1)


giskard_model = Model(
    model=prediction_function,  # A prediction function that encapsulates all the data pre-processing steps and that
    model_type="classification",  # Either regression, classification or text_generation.
    name="RoBERTa for sentiment classification",  # Optional
    classification_labels=list(LABEL_MAPPING.values()),  # Their order MUST be identical to the prediction_function's
    feature_names=[TEXT_COLUMN],  # Default: all columns of your dataset
)

results = scan(giskard_model, giskard_dataset)

# Generate the output path if it doesn't exist
isExist = os.path.exists("../output")

if not isExist:
    os.makedirs("../output")

now = datetime.now()

dt_string = now.strftime("%d-%m-%Y-%H:%M:%S")

# Save it to a file
# results.to_html("../output/scan_report.html")
results.to_html(f'../output/scan_report_{dt_string}.html')
