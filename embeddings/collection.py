import requests
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
            json={
                "collection_name": self.collection_name,
                "new_collection_name": new_collection_name,
                "api_key": api_key
            }
        )
        return response.json()

    # Returns the number of items in the collection.
    def count(self):
        if api_key is None:
            raise Exception(
                "API key not set. Use embeddings.api_key = API_KEY to set the API key.")

        response = requests.post(
            f"{API_URL}/count_collection",
            json={"collection_name": self.collection_name, "api_key": api_key}
        )
        return response.json()

    # Adds embeddings to the collection.
    # :param documents: List of texts corresponding to the embeddings to be added.
    # :param metadatas: List of dictionaries of metadata corresponding to the embeddings to be added.
    # :param ids: List of ids corresponding to the embeddings to be added.
    # :param embeddings: List of embeddings to be added.
    def add(self, documents, metadatas=None, ids=None, embeddings=None):
        if api_key is None:
            raise Exception(
                "API key not set. Use embeddings.api_key = API_KEY to set the API key.")

        response = requests.post(
            f"{API_URL}/add_embeddings",
            json={
                "collection_name": self.collection_name,
                "api_key": api_key,
                "documents": documents,
                "metadatas": metadatas,
                "ids": ids,
                "embeddings": embeddings
            }
        )
        return response.json()

    # Queries the collection for similar embeddings.
    # :param query_embeddings: List of embeddings to be queried.
    # :param n_results: Number of results to return.
    # !!! TODO: CURRENTLY ONLY SUPPORTS ONE DICTIONARY FOR FILTER
    # :param (optional) where: Dictionary of metadata filter to be applied.
    # :param (optional) search_string: Search string to be applied.
    # :param (optional) include: List of what should be included in the query, can be equal to any on or multiple of ["metadata", "embedding", "documents"]
    def query(self, query_embeddings, n_results=10, where=None, search_string=None, include=None):
        if api_key is None:
            raise Exception(
                "API key not set. Use embeddings.api_key = API_KEY to set the API key.")

        response = requests.post(
            f"{API_URL}/query_collection",
            json={
                "collection_name": self.collection_name,
                "api_key": api_key,
                "query_embeddings": query_embeddings,
                "n_results": n_results,
                "where": where,
                "search_string": search_string,
                "include": include
            }
        )
        return response.json()

    # Gets the embeddings for the given ids, search string or where clause.
    # :param ids: List of ids to get embeddings for.
    # :param (optional) where: Dictionary of metadata filter to be applied.
    # :param (optional) search_string: Search string to be applied.
    # :param (optional) include: List of what should be included in the query, can be equal to any on or multiple of ["metadata", "embedding", "documents"]
    def get(self, ids, where=None, search_string=None, include=None):
        if api_key is None:
            raise Exception(
                "API key not set. Use embeddings.api_key = API_KEY to set the API key.")

        response = requests.post(
            f"{API_URL}/get_collection_items",
            json={
                "collection_name": self.collection_name,
                "api_key": api_key,
                "ids": ids,
                "where": where,
                "search_string": search_string,
                "include": include
            }
        )
        return response.json()

    # Deletes the given ids, or where clause from the collection.
    # :param ids: List of ids to delete.
    # :param (optional) where: Dictionary of metadata filter to be applied.
    def delete(self, ids=None, where=None):
        response = requests.post(
            f"{API_URL}/delete_from_collection",
            json={
                "collection_name": self.collection_name,
                "api_key": api_key,
                "ids": ids,
                "where": where
            }
        )
        return response.json()
