import chromadb
client = chromadb.PersistentClient(path="../chroma_db")
collection = client.get_collection("articles")
print("문서 수:", collection.count())

query = input("검색 키워드를 입력하세요: ")
results = collection.query(query_texts=[query], n_results=5, include=["documents", "metadatas", "distances"])

for i in range(len(results['ids'][0])):
    print(f"\n유사도 점수: {1 - results['distances'][0][i]:.4f}")
    print("제목:", results['metadatas'][0][i]['title'])
    print("본문 일부:", results['documents'][0][i][:100])