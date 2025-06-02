
import json
import hashlib
from openai import OpenAI
import chromadb

# Initialize OpenAI client (v1 API)
client = OpenAI()

# Initialize ChromaDB client with modern persistent config
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection(name="portfolio_advice")

def compute_embedding(text: str) -> list:
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def save_to_vector_db(portfolio_data, gpt_summary):
    input_str = json.dumps(portfolio_data, sort_keys=True)
    input_hash = hashlib.sha256(input_str.encode()).hexdigest()
    embedding = compute_embedding(input_str)

    collection.add(
        documents=[input_str],
        embeddings=[embedding],
        metadatas=[{"summary": gpt_summary}],
        ids=[input_hash]
    )

def search_similar_portfolios(portfolio_data):
    input_str = json.dumps(portfolio_data, sort_keys=True)
    embedding = compute_embedding(input_str)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=3
    )
    return results
