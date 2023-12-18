# giskard-POC
Proof of concept for Giskard

# Setup

1. Confirm you're using Python 3.11 - `python3 --version`
2. Set up python virtual directory with `python3 -m venv venv`
3. Activate the virtual env from the terminal by run `source venv/bin/activate`
4. Install requirements using the command `pip install -r requirements.txt`
5. Use the `example.env` to create a `.env` file with an Open AI API key

# Note
- If you have an error about `attr.s`, uninstalling and reinstalling `attrs` fixed it.