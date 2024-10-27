import click
from elasticsearch import Elasticsearch
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Load the tokenizer and model from Hugging Face
model_name = "distilbert-base-uncased"  # You can choose any model suitable for your task
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to generate embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings.squeeze().tolist()  # Convert to list

# Connect to the local Elasticsearch instance
es = Elasticsearch("http://localhost:9200")

# CLI group for organizing commands
@click.group()
def cli():
    """A CLI tool to interact with Elasticsearch"""
    pass

# Command to create a dummy index
@cli.command()
def create_index():
    """Create a dummy index with some data."""
    index_name = 'dummy_index'

    # Define a simple document structure
    documents = [
        {'id': 1, 'name': 'Alice', 'age': 25, 'occupation': 'Engineer'},
        {'id': 2, 'name': 'Bob', 'age': 30, 'occupation': 'Designer'},
        {'id': 3, 'name': 'Charlie', 'age': 35, 'occupation': 'Doctor'},
    ]
    
    # Create the index
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name)
        click.echo(f"Created index: {index_name}")

        # Insert documents with embeddings
        for doc in documents:
            # Generate embedding for the name
            embedding = get_embedding(doc['name'])
            doc['embedding'] = embedding  # Add embedding to document
            
            es.index(index=index_name, id=doc['id'], document=doc)
            click.echo(f"Inserted document ID: {doc['id']}")
    else:
        click.echo(f"Index {index_name} already exists.")

# Command to search the dummy index (regular search)
@cli.command()
@click.argument('query')
def search(query):
    """Search the dummy index for the provided query."""
    index_name = 'dummy_index'
    if es.indices.exists(index=index_name):
        # Elasticsearch query to match documents
        response = es.search(index=index_name, query={"match": {"name": query}})
        
        # Display results
        if response['hits']['hits']:
            click.echo(f"Found {len(response['hits']['hits'])} results:")
            for hit in response['hits']['hits']:
                click.echo(hit['_source'])
        else:
            click.echo("No results found.")
    else:
        click.echo(f"Index {index_name} does not exist. Please create it first.")

# Command to perform semantic search based on embeddings
@cli.command()
@click.argument('query')
def semantic_search(query):
    """Perform semantic search using embeddings."""
    index_name = 'dummy_index'
    
    if es.indices.exists(index=index_name):
        # Generate embedding for the query
        query_embedding = get_embedding(query)

        # Fetch all embeddings from the index
        response = es.search(index=index_name, body={
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

        # Display results
        click.echo(f"Found {len(indices[0])} closest results:")
        for idx in indices[0]:
            hit = response['hits']['hits'][idx]
            click.echo(hit['_source'])
    else:
        click.echo(f"Index {index_name} does not exist. Please create it first.")

# Command to delete the dummy index
@cli.command()
def delete_index():
    """Delete the dummy index."""
    index_name = 'dummy_index'
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        click.echo(f"Deleted index: {index_name}")
    else:
        click.echo(f"Index {index_name} does not exist.")

if __name__ == "__main__":
    cli()
