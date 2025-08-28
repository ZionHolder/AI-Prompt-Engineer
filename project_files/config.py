from dotenv import load_dotenv, find_dotenv

#Load API Key
if not (load_dotenv(find_dotenv())):
    raise ValueError("Failed to load .env file")

CREDENTIALS = {
    "openai_url" : "https://api.openai.com/v1"
}

#Model ID
OPEN_AI_MODEL_ID = "gpt-4o"