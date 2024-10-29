import click
from elasticsearch import Elasticsearch, warnings
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Avoid redundant initializations.
class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Logger:
    __metaclass__ = Singleton

    def __init__(self):
        # Doesn't take args yet.
        pass

    def log(self, message):
        click.echo(message)

class Result:
    response = None
    indices = None
    
    def __init__(self, response=None, indices=None):
        self.response = response
        self.indices = indices

class ElasticSearchService:
    __metaclass__ = Singleton

    # Load the tokenizer and model from Hugging Face
    model_name = "distilbert-base-uncased"  # You can choose any model suitable for your task
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    logger = Logger()

    # Initialize the Elasticsearch service
    def __init__(self):
        self.search_server = Elasticsearch("http://localhost:9200")

    # Function to generate embeddings
    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)  # Mean pooling
        return embeddings.squeeze().tolist() 
    

    # Populate our service's index with documents
    def create_index(self, index_name):
        """Create a dummy index with some data."""

        # Define a simple document structure
        documents = [
            {'id': 1, 'name': 'Alice', 'age': 25, 'occupation': 'Engineer'},
            {'id': 2, 'name': 'Bob', 'age': 30, 'occupation': 'Designer'},
            {'id': 3, 'name': 'Charlie', 'age': 35, 'occupation': 'Doctor'},
            {'id': 4, 'name': 'David', 'age': 40, 'occupation': 'Engineer'},
            {'id': 5, 'name': 'Eve', 'age': 45, 'occupation': 'Designer'},
            {'id': 6, 'name': 'Frank', 'age': 50, 'occupation': 'Cook'},
            {'id': 7, 'name': 'Grace', 'age': 55, 'occupation': 'Delivery Driver'},
        ]
        
        # Create the index
        if not self.search_server.indices.exists(index=index_name):
            self.search_server.indices.create(index=index_name)
            self.logger.log(f"Created index: {index_name}")

            # Insert documents with embeddings
            for doc in documents:
                # Generate embedding for the name
                embedding = self.get_embedding(doc['name'])
                doc['embedding'] = embedding  # Add embedding to document
                
                self.search_server.index(index=index_name, id=doc['id'], document=doc)
                self.logger.log(f"Inserted document ID: {doc['id']}")
        else:
            self.logger.log(f"Index {index_name} already exists.")

    # Lexical search
    def lexical_search(self, query, index_name):
        """Search the dummy index for the provided query."""
        if self.search_server.indices.exists(index=index_name):
            # Elasticsearch query to match documents
            response = self.search_server.search(index=index_name, query={"match": {"name": query}})
            result = Result(response=response)
            self.display_results(result, lexical_or_semantic="lexical")
        else:
            self.logger.log(f"Index {index_name} does not exist. Please create it first.")

    # Semantic search
    def semantic_search(self, query, index_name):
        if self.search_server.indices.exists(index=index_name):
            # Generate embedding for the query
            query_embedding = self.get_embedding(query)

            # Fetch all candidates from the index
            response = self.search_server.search(index=index_name, body={
                "query": {
                    "match_all": {}
                }
            })
            
            embeddings = []
            ids = []
            
            for hit in response['hits']['hits']:
                ids.append(hit['_id'])
                embeddings.append(hit['_source']['embedding'])

            # Convert embeddings to numpy array for KNN
            embeddings_array = np.array(embeddings)

            # Use KNN to find the closest embeddings
            nbrs = NearestNeighbors(n_neighbors=3, metric='cosine').fit(embeddings_array)
            distances, indices = nbrs.kneighbors([query_embedding])

            result = Result(response=response, indices=indices)
            self.display_results(result, lexical_or_semantic="semantic")
        else:
            self.logger.log(f"Index {index_name} does not exist. Please create it first.")

    def delete_index(self, index_name):
        if self.search_server.indices.exists(index=index_name):
            self.search_server.indices.delete(index=index_name)
            self.logger.log(f"Deleted index: {index_name}")
        else:
            self.logger.log(f"Index {index_name} does not exist.")

    def display_results(self, result: Result, lexical_or_semantic: str):
        if lexical_or_semantic == "lexical":
            response = result.response
            if response['hits']['hits']:
                self.logger.log(f"Found {len(response['hits']['hits'])} results:")
                for source in response['hits']['hits']:
                    hit = source['_source']
                    self.logger.log(f"Hit name: {hit['name']}, age: {hit['age']}, occupation: {hit['occupation']}")
            else:
                self.logger.log("No results found.")
        
        elif lexical_or_semantic == "semantic":
            indices = result.indices
            response = result.response
            self.logger.log(f"Found {len(indices[0])} closest results:")
            for idx in indices[0]:
                hit = response['hits']['hits'][idx]['_source']
                self.logger.log(f"Hit name: {hit['name']}, age: {hit['age']}, occupation: {hit['occupation']}")


# CLI group for organizing commands
@click.group()
def cli():
    """A CLI tool to interact with Elasticsearch"""
    pass

# Command to create a dummy index
@cli.command()
def create_index():
    index_name = 'dummy_index'
    service = ElasticSearchService()
    service.create_index(index_name)
    click.echo(f"Created index: {index_name}")

# Command to search the dummy index (regular search)
@cli.command()
@click.argument('query')
def lexical_search(query):
    """Search the dummy index for the provided query."""
    index_name = 'dummy_index'
    service = ElasticSearchService()
    service.lexical_search(query, index_name)
    

# Command to perform semantic search based on embeddings
@cli.command()
@click.argument('query')
def semantic_search(query):
    """Perform semantic search using embeddings."""
    index_name = 'dummy_index'
    service = ElasticSearchService()
    service.semantic_search(query, index_name)

# Command to perform semantic search based on embeddings
@cli.command()
@click.argument('query')
def multi_search(query):
    """Perform lexical, then semantic search for a query."""
    index_name = 'dummy_index'
    service = ElasticSearchService()
    service.lexical_search(query, index_name)
    service.semantic_search(query, index_name)


# Command to delete the dummy index
@cli.command()
def delete_index():
    """Delete the dummy index."""
    index_name = 'dummy_index'
    service = ElasticSearchService()
    service.delete_index(index_name)

if __name__ == "__main__":
    # Elastic Search complains that our cluster is unsafe (permissions).
    # But thats fine because this is a demo.
    warnings.filterwarnings("ignore")
    cli()
