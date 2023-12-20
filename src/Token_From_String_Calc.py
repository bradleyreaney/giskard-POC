import tiktoken
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# PDF_LOCATION = "exampleHRDoc/Generic-HR-Policy.pdf"
PDF_LOCATION = "exampleHRDoc/Generic-HR-Policy_Short.pdf"
SAMPLE_TEXT = "If you feel unwell and not able to work, please make sure you contact your line manager via a phone call before 9.00am or as soon as you begin to feel unwell, if this is during the working day."


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = PyPDFLoader(PDF_LOCATION).load_and_split(text_splitter)


# List token count for PDF. Used for pricing against OpenAI
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    print(num_tokens)


num_tokens_from_string(SAMPLE_TEXT, "cl100k_base")