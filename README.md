# giskard-POC
Proof of concept for Giskard

# Setup

1. Confirm you're using Python 3.11 - `python3 --version`
2. Set up the python virtual directory within the project folder using `python3 -m venv venv`
3. Activate the virtual env from the terminal by run `source venv/bin/activate`
4. Install requirements using the command `pip install -r requirements.txt`
5. Use the `example.env` to create a `.env` file with an Open AI API key

# Things to look out for
- If you have an error about `attr.s`, uninstalling and reinstalling `attrs` fixed it. `pip uninstall attrs` & `pip install attrs`

# What everything does
Note - Most of these examples are still WIP while we test this as a POC. They may require additional changes to get working

#### LLM_Documentation_Q&A
- This will take in a PDF and create the embeddings using Open AIs `https://api.openai.com/v1/embeddings` endpoint and `test-embedding-ada-002` model.
- Gisdard with then generate multiple tests sentenses using Open AIs `https://api.openai.com/v1/completions`and `https://api.openai.com/v1/chat/completions` endpoints using the `text-ada-001` model.
- logs from the first run have been added to the `logExamples` folder. Note - This took $1.52 of Open AI credit to run due to it using GPT-4.

#### LLM_QuickStart_SentimentClassification
- This is an example of Giskard been ran against a sentiment classifier. It includes two Hugging Face datasets and sentiment models.
    - `cardiffnlp/twitter-roberta-base-sentiment` & `tweet_eval`
    - `mnoukhov/gpt2-imdb-sentiment-classifier` & `imdb`

#### LLM_Scan
- This one has had nothing done to it past the example code taken from the Giskard site. Looks like we'll be able to cook up out API and run the Giskard scan against it. Need to look into upgrading our Open AI API key to test this further.

#### Token_From_String_Calc
- This is just some code i've chucked up that might be useful at some point. It calculates how many tokens will be used to create embeddings of a given piece of text. We can then use this to calculate cost of embeddings with 3rd parties such as Open AIs `https://api.openai.com/v1/embeddings` endpoint