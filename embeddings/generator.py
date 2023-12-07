import requests
from transformers import AutoTokenizer
from typing import Union, List
from . import api_key

# Constants
GENERATION_SERVER_URL = "https://api.silverarrow.ai/generation"

class Generator:
    # Initializes the Generator
    # TODO: Should make an api call to set to user's chosen model
    def __init__(self):
        self.model_name = "MiniLM"
        self.tokenizers = {}
        self.models_info = {
            "bge-small": {
                "full_name": "BAAI/bge-small-en-v1.5",
                "type": "text",
                "description": "General purpose embedding model",
                "dimensions": 384,
                "token_limit": 512,
                "pricing": "0.0001 per embedding",
            },
            "MiniLM": {
                "full_name": "sentence-transformers/all-MiniLM-L6-v2",
                "type": "text",
                "description": "General purpose embedding model",
                "dimensions": 384,
                "token_limit": 512,
                "pricing": "0.00001 per embedding",
            }
        }
        # Preload tokenizers for all models
        for model in self.models_info:
            full_name = self.models_info[model]["full_name"]
            self.tokenizers[model] = AutoTokenizer.from_pretrained(full_name)

    # List all available embedding models and their details
    def list_models(self):
        return list(self.models_info.values())

    # Set the model to be used for embedding generation
    def set_model(self, model_name: str):
        if model_name not in self.models_info:
            raise ValueError("Model not found. Please choose a valid model.")
        self.model_name = model_name

    # Get the details of the set model
    def get_model_info(self):
        return self.models_info.get(self.model_name, {})

    def embed(self, text: Union[str, List[str]]):
        if api_key is None:
            raise Exception("API key not set. Use embeddings.api_key = API_KEY to set the API key.")

        # Convert text to a list if it's a single string
        texts = [text] if isinstance(text, str) else text

        # Check if the text(s) are within the token limit
        for t in texts:
            is_within_limit, message = self.within_token_limit(t)
            if not is_within_limit:
                raise ValueError(message)

        model_full_name = self.models_info[self.model_name]["full_name"]
        response = requests.post(
            f"{GENERATION_SERVER_URL}/embed_text",
            json={"texts": texts, "model_name": model_full_name},
            headers={"Authorization": f"Bearer {api_key}"}
        )

        if response.status_code != 200:
            raise Exception(f"Error in embedding generation: {response.text}")

        return response.json()["embeddings"]

    
    # Counts the number of tokens in the given text with respect to the set embedding model
    def count_tokens(self, text: str):
        tokenizer = self.tokenizers[self.model_name]
        inputs = tokenizer(text)
        return len(inputs["input_ids"])

    # Checks if the given text is within the token limit of the set embedding model
    def within_token_limit(self, text: str):
        token_count = self.count_tokens(text)
        max_tokens = self.models_info[self.model_name]["token_limit"]
        if token_count > max_tokens:
            return False, f"Text exceeds the token limit of {max_tokens}.\nCurrent token count is {token_count}."
        return True, "Text is within the token limit."
