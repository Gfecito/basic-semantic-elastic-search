# Elasticsearch CLI Tool

Mostly for educational purposes on how semantic search can work.
Similar projects already exist, but I wanted a simple CLI executable with minimum overhead, so I made this.

A command-line interface (CLI) tool to interact with Elasticsearch.

This tool allows you to create, search, and manage a dummy index in your Elasticsearch instance, as well as perform semantic searches using embeddings.

## Note

You should know -or find elsewhere- how to set up your own elasticsearch instance.
This resource might suffice: [elasticsearch docs](https://www.elastic.co/guide/en/elasticsearch/reference/7.17/docker.html).

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Commands](#commands)
- [Contributing](#contributing)
- [License](#license)

## Features

- Create a dummy index with sample data.
- Perform basic search queries.
- Implement semantic search using KNN over embeddings.

## Requirements

- Python 3.12 or higher
- Elasticsearch instance running locally (default: http://localhost:9200)

## Installation

1. **Clone the repository**:

```bash
git clone https://github.com/Gfecito/basic-semantic-elastic-search.git
cd your-repo-name
```

2. **Create and activate a virtual environment**:

```
python -m venv env
source env/bin/activate # macOS/Linux
env\Scripts\activate # Windows
```

3. **Install dependencies**:

```
pip install -r requirements.txt
```

# Usage

After setting up the environment, you can run the CLI tool with the following command:

```
python es_cli.py
```

# Commands

Create Index: Create a dummy index with sample data.

```
python es_cli.py create_index
```

Search: Search for documents in the dummy index by name.

```
python es_cli.py search "Alice"
```

Semantic Search: Perform a semantic search based on embeddings.

```
python es_cli.py semantic_search "Alice"
```

Delete Index: Delete the dummy index.

```
python es_cli.py delete_index
```

# Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any improvements, bug fixes, or suggestions.

# License

This project is licensed under the MIT License - see the LICENSE file for details.
