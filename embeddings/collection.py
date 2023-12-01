import requests
import re
from . import api_key

# TODO: Replace with hosted API URL
API_URL = "http://your-api-url"


class Collection:
    def __init__(self, collection_name):
        self.collection_name = collection_name

    # Renames a collection in the db.
    # :param new_collection_name: The name to rename the collection to.
    def rename(self, new_collection_name):
        if api_key is None:
            raise Exception(
                "API key not set. Use embeddings.api_key = API_KEY to set the API key.")

        response = requests.post(
            f"{API_URL}/rename_collection",
            json={"collection_name": self.collection_name, "new_collection_name": new_collection_name},
            headers={"Authorization": f"Bearer {api_key}"}
        )
        return response.json()

    # Returns the number of items in the collection.
    def count(self):
        if api_key is None:
            raise Exception(
                "API key not set. Use embeddings.api_key = API_KEY to set the API key.")

        response = requests.post(
            f"{API_URL}/count_collection",
            json={"collection_name": self.collection_name},
            headers={"Authorization": f"Bearer {api_key}"}
        )
        return response.json()

    # Adds embeddings to the collection.
    # :param documents: List of texts corresponding to the embeddings to be added.
    # :param metadatas: List of dictionaries of metadata corresponding to the embeddings to be added.
    # :param ids: List of ids corresponding to the embeddings to be added.
    # :param embeddings: List of embeddings to be added.
    # TODO: Add check for data types of parameters
    def add(self, embeddings=None, documents=None, metadatas=None, ids=None):
        if api_key is None:
            raise Exception(
                "API key not set. Use embeddings.api_key = API_KEY to set the API key.")

        response = requests.post(
            f"{API_URL}/add_embeddings",
            json={
                "collection_name": self.collection_name,
                "documents": documents,
                "metadatas": metadatas,
                "ids": ids,
                "embeddings": embeddings
            },
            headers={"Authorization": f"Bearer {api_key}"}
        )
        return response.json()

    # Queries the collection for similar embeddings.
    # :param query_embeddings: List of embeddings to be queried.
    # :param n_results: Number of results to return.
    # !!! TODO: CURRENTLY ONLY SUPPORTS ONE DICTIONARY FOR FILTER
    # :param (optional) where: Dictionary of metadata filters to be applied.
    # :param (optional) include_embeddings: Whether to include embeddings in the response.
    def query(self, embedding, n_results=10, where=None, include_embeddings=False):
        if api_key is None:
            raise Exception("API key not set. Use embeddings.api_key = API_KEY to set the API key.")

        include = None
        if include_embeddings:
            include = ["metadatas", "documents", "embeddings"]

        response = requests.post(
            f"{API_URL}/query_collection",
            json={
                "collection_name": self.collection_name,
                "query_embeddings": [embedding],
                "n_results": n_results,
                "where": where,
                "include": include
            },
            headers={"Authorization": f"Bearer {api_key}"}
        )
        return response.json()

    # Gets the embeddings for the given ids, search string or where clause.
    # :param ids: List of ids to get embeddings for.
    # :param (optional) where: Dictionary of metadata filter to be applied.
    # :param (optional) search_string: Search string to be applied.
    # :param (optional) include_embeddings: Whether to include embeddings in the response.
    def get(self, ids=None, where=None, search_string=None, include_embeddings=False):
        if api_key is None:
            raise Exception("API key not set. Use embeddings.api_key = API_KEY to set the API key.")

        include = None
        if include_embeddings:
            include = ["metadatas", "documents", "embeddings"]

        response = requests.post(
            f"{API_URL}/get_collection_items",
            json={
                "collection_name": self.collection_name,
                "ids": ids,
                "where": where,
                "search_string": search_string,
                "include": include
            },
            headers={"Authorization": f"Bearer {api_key}"}
        )
        return response.json()

    # Deletes the given ids, or where clause from the collection.
    # :param ids: List of ids to delete.
    # :param (optional) where: Dictionary of metadata filter to be applied.
    def delete(self, ids=None, where=None):
        response = requests.post(
            f"{API_URL}/delete_from_collection",
            json={"collection_name": self.collection_name, "ids": ids, "where": where},
            headers={"Authorization": f"Bearer {api_key}"}
        )
        return response.json()
