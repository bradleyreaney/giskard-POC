import os
import giskard
import pandas as pd
from datetime import datetime
import requests

from dotenv import load_dotenv

load_dotenv()

# Reference Docs - https://docs.giskard.ai/en/latest/open_source/scan/scan_llm/index.html

# Note - Open AI key needs GPT-4 access

now = datetime.now()

dt_string = now.strftime("%d-%m-%Y-%H:%M:%S")


def model_predict(df: pd.DataFrame):
    """Wraps the LLM call in a simple Python function.

    The function takes a pandas.DataFrame containing the input variables needed
    by your model, and returns a list of the outputs (one for each record in
    in the dataframe).
    """
    return [llm_api(question) for question in df["question"].values]


# Create a giskard.Model object. Don’t forget to fill the `name` and `description`
# parameters: they will be used by our scan to generate domain-specific tests.
giskard_model = giskard.Model(
    model=model_predict,  # our model function
    model_type="text_generation",
    name="Nimble AI LLM Chatbot",
    description="Nimble AI LLM Chatbot",
    feature_names=["question"],  # input variables needed by your model
)

scan_results = giskard.scan(giskard_model)

# Generate the output path if it doesn't exist
isExist = os.path.exists("../output")

if not isExist:
    os.makedirs("../output")

# Save it to a file
scan_results.to_html(f'../output/scan_report_{dt_string}.html')
