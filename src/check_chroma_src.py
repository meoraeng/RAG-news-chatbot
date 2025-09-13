# check_chroma_src.py
import chromadb
client = chromadb.PersistentClient(path="./src/chroma_db")
collection = client.get_collection("articles")
results = collection.get(limit=1, include=['embeddings'])
print("src/chroma_db 임베딩 차원:", len(results['embeddings'][0]))
print("src/chroma_db 문서 수:", collection.count())