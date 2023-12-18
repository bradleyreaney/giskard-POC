import os
import numpy as np
import pandas as pd
from scipy.special import softmax
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datetime import datetime

from giskard import Dataset, Model, scan, testing, GiskardClient, Suite

#  Reference Docs - https://docs.giskard.ai/en/latest/getting_started/quickstart/quickstart_nlp.html

# MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
MODEL_NAME = "mnoukhov/gpt2-imdb-sentiment-classifier"

# DATASET_CONFIG = {"path": "tweet_eval", "name": "sentiment", "split": "validation"}
DATASET_CONFIG = {"path": "imdb", "split": "train"}

# LABEL_MAPPING = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_MAPPING = {0: "neg", 1: "pos"}

TEXT_COLUMN = "text"
TARGET_COLUMN = "label"

NOW = datetime.now()

DT_STRING = NOW.strftime("%d-%m-%Y-%H:%M:%S")

raw_data = load_dataset(**DATASET_CONFIG).to_pandas().iloc[:10]  # This is the sample amount. Can sometimes cause a failure when using the same number.
raw_data = raw_data.replace({"label": LABEL_MAPPING})

giskard_dataset = Dataset(
    df=raw_data,
    # A pandas.DataFrame that contains the raw data (before all the pre-processing steps) and the actual ground truth variable (target).
    target=TARGET_COLUMN,  # Ground truth variable.
    name="IMDB reviews with sentiment",  # Optional.
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
    # name="RoBERTa for sentiment classification",  # Optional
    classification_labels=list(LABEL_MAPPING.values()),  # Their order MUST be identical to the prediction_function's
    feature_names=[TEXT_COLUMN],  # Default: all columns of your dataset
)

results = scan(giskard_model, giskard_dataset)

# Generate the output path if it doesn't exist
isExist = os.path.exists("../output")

if not isExist:
    os.makedirs("../output")

# Save it to a file
results.to_html(f'../output/scan_report_{DT_STRING}.html')
