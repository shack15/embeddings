import requests
from .collection import Collection
from . import api_key

# TODO: Replace with hosted API URL
API_URL = "http://your-api-url"


class Database:
    # Creates a new collection in ChromaDB and returns a Collection object.
    # :param name: Name of the collection to be created
    # :param distance_metric: The distance metric for the collection (default is "cosine").
    # :param model_name: Name of the model to be used for embeddings (optional).
    # :return: Collection object.
    def create_collection(self, name, distance_metric="cosine"):
        if api_key is None:
            raise Exception(
                "API key not set. Use embeddings.api_key = API_KEY to set the API key.")

        # Make API request to create a new collection
        response = requests.post(
            f"{API_URL}/create_collection",
            json={
                "collection_name": name,
                "api_key": api_key,
            }
        )

        if response.status_code == 200:
            return Collection(name)
        else:
            raise Exception(f"Failed to create collection: {response.text}")

    # Connects to an existing collection in ChromaDB and returns a Collection object.
    # :param name: Name of the collection to be loaded.
    # :return: Collection object.
    def load_collection(self, name):
        if api_key is None:
            raise Exception(
                "API key not set. Use embeddings.api_key = API_KEY to set the API key.")

        # Make API request to get the collection
        response = requests.post(
            f"{API_URL}/get_collection",
            json={"collection_name": name, "api_key": api_key}
        )

        if response.status_code == 200:
            return Collection(name)
        elif response.status_code == 404:
            # TODO: Add a 404 if the collection is not found
            raise Exception(f"Collection '{name}' not found.")
        else:
            raise Exception(f"Failed to load collection: {response.text}")

    # Deletes a collection from ChromaDB.
    # :param name: Name of the collection to be deleted.
    def delete_collection(self, name):
        if api_key is None:
            raise Exception(
                "API key not set. Use embeddings.api_key = API_KEY to set the API key.")

        # Making the API request to delete the collection
        response = requests.delete(
            f"{API_URL}/delete_collection",
            json={"collection_name": name, "api_key": api_key}
        )

        if response.status_code != 200:
            raise Exception(f"Failed to delete collection: {response.text}")
