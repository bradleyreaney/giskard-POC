import os
import giskard
import requests
import pandas as pd
from giskard.llm import set_llm_model
from datetime import datetime
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv()

# Reference Docs - https://docs.giskard.ai/en/latest/open_source/scan/scan_llm/index.html
base_api_url = os.environ["BASE_API_URL"]
base_api_key = os.environ["BASE_API_KEY"]
output_path = os.environ["OUTPUT_PATH"]

# Open AI / Azure OpenAI Service will need GPT-4 access
# You'll need to provide the name of the model that you've deployed
# Beware, the model provided must be capable of using function calls
# The bellow is only needed it you're using Azzure OpenAI Service. Otherwise comment it out.
set_llm_model('my-gpt-4-model')

llm_description = "This is an internal RAG based LLM Chatbot. It will be used to answer low value questions staff may have around HR polices. It will have access to our internal HR policy documentation and will use these documents alone for it's context when answering questions"

def model_predict(df: pd.DataFrame):
    def llm_api(input_questions):
        response = requests.get(f"{base_api_url}/prod/answer?question={quote(input_questions)}", headers={"x-api-key":base_api_key})
        print("RAG API Response - " + response.content)
        return response.content
    return [llm_api(question) for question in df["question"].values]

# Create a giskard.Model object. Don’t forget to fill the `name` and `description`
# parameters: they will be used by our scan to generate domain-specific tests.
giskard_model = giskard.Model(
    model=model_predict,  # our model function
    model_type="text_generation",
    name="Internal AI LLM Chatbot",
    description=llm_description,
    feature_names=["question"],  # input variables needed by your model
)
 
# Options you can pass into the scan. 
# The 'robustness' option is free so a good option to use when getting up and running.
# In the 'giskark.scan()' function, leaving out the 'only=["OPTIONS"]' will run every type.
#  - robustness 
#  - text_generation
scan_results = giskard.scan(giskard_model, only=["robustness"])
# scan_results = giskard.scan(giskard_model)

# Generate the output path if it doesn't exist
isExist = os.path.exists(output_path)

if not isExist:
    os.makedirs(output_path)

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H:%M:%S")

# Save it to a file
scan_results.to_html(f'{output_path}/scan_report_{dt_string}.html')


