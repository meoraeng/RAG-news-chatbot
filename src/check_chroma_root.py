# check_chroma_root.py
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("articles")
results = collection.get(limit=1, include=['embeddings'])
print("루트 chroma_db 임베딩 차원:", len(results['embeddings'][0]))
print("루트 chroma_db 문서 수:", collection.count())