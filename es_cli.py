import click
from elasticsearch import Elasticsearch

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

        # Insert documents
        for doc in documents:
            es.index(index=index_name, id=doc['id'], document=doc)
            click.echo(f"Inserted document ID: {doc['id']}")
    else:
        click.echo(f"Index {index_name} already exists.")

# Command to search the dummy index
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
