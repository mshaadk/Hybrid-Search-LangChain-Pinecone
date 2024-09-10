# Hybrid Search with LangChain,Pinecone and HuggingFace
This Google Colab Notebook demonstrates how to integrate Pinecone with LangChain and HuggingFace to create a hybrid search system that combines vector embeddings with sparse retrieval techniques. This setup can be used for efficient information retrieval and search tasks.

## Features
- **Pinecone Integration**: Use Pinecone to store and manage vector embeddings for hybrid search.
- **LangChain**: Utilizes LangChain's retrievers to facilitate hybrid search capabilities.
- **HuggingFace Embeddings**: Employs HuggingFace's `all-MiniLM-L6-v2` model for generating vector embeddings.
- **Sparse Retrieval**: Incorporates BM25 encoding for handling sparse data.

## Installation
To get started, follow these instructions:

### 1. Clone the repository:

```bash
git clone https://github.com/mshaadk/Hybrid-Search-LangChain-Pinecone.git
cd Hybrid-Search-LangChain-Pinecone
```

### 2. Install the required packages:

Make sure you have Python installed, then run:

```bash
pip install --upgrade --quiet pinecone-client pinecone-text pinecone-notebooks langchain-community langchain-huggingface
```

### 3. Set Up API Keys:

Ensure you have your Pinecone API key ready. You can set this up using a `.env` file or directly in the notebook as shown:

```python
from google.colab import userdata
api_key = userdata.get('PINECONE_API')
```

Usage
### 1. Initialize Pinecone Client:

Set up the Pinecone client and create an index if it doesn't already exist.

```python
from pinecone import Pinecone, ServerlessSpec

index_name = "hybrid-search-langchain-pinecone"
pc = Pinecone(api_key=api_key)

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="dotproduct",
        spec=ServerlessSpec(cloud='aws', region="us-east-1")
    )

index = pc.Index(index_name)
```

### 2. Generate Embeddings:

Utilize the `HuggingFaceEmbeddings` to create vector embeddings.

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

### 3. Setup Sparse Retrieval with BM25:

Encode text data using BM25 and store the values.

```python
from pinecone_text.sparse import BM25Encoder

bm25_encoder = BM25Encoder().default()
sentences = [
    "In 2023, I visited Paris",
    "In 2022, I visited New York",
    "In 2021, I visited London",
    "In 2020, I visited Rome"
]
bm25_encoder.fit(sentences)
bm25_encoder.dump("bm25_values.json")
```

### 4. Create and Use Hybrid Retriever:

Combine the embeddings and BM25 encoder for hybrid search.

```python
from langchain_community.retrievers import PineconeHybridSearchRetriever

bm25_encoder = BM25Encoder().load("bm25_values.json")
retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index)

retriever.add_texts(sentences)
result = retriever.invoke("What city did I visit in 2022?")
print(result[0])
```

## License
This project is licensed under the [MIT License](LICENSE.txt).

## Contact
For any questions or suggestions, feel free to reach out to [Mohamed Shaad](https://www.linkedin.com/in/mohamedshaad/).
