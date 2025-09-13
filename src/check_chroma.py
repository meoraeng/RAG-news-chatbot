import os
import chromadb
import numpy as np
from langchain_upstage import UpstageEmbeddings

CHROMA_DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_db"))
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

print("\n=== ChromaDB 컬렉션 목록 ===")
collections = client.list_collections()
for collection in collections:
    print(f"\n컬렉션 이름: {collection.name}")
    print(f"컬렉션 메타데이터: {collection.metadata}")
    print(f"문서 개수: {collection.count()}")
    
    # 각 컬렉션의 내용 확인
    results = collection.get(limit=5, include=['documents', 'metadatas'])
    if results['documents']:
        print("\n문서 샘플:")
        for i, doc in enumerate(results['documents']):
            print(f"\n{i+1}번째 문서:")
            print(f"내용: {doc[:200]}...")  # 처음 200자만 출력
            if results['metadatas']:
                print(f"메타데이터: {results['metadatas'][i]}")
    print("\n" + "="*50)

# 크롤러와 동일한 임베딩 함수 설정
embeddings = UpstageEmbeddings(model="embedding-passage")
collection = client.get_collection("articles")

results = collection.get(limit=1, include=['embeddings', 'metadatas', 'documents'])
print("\n=== 임베딩 정보 ===")
print("실제 임베딩 차원:", len(results['embeddings'][0]))
print("임베딩 shape:", np.array(results['embeddings']).shape)
print("실제 chroma_db 경로:", CHROMA_DB_DIR)
print("컬렉션 내 문서 개수:", collection.count())
print("get() 결과:", results)
print("컬렉션 목록:", client.list_collections())
if results['metadatas']:
    print("메타데이터:", results['metadatas'][0])
if results['documents']:
    print("문서:", results['documents'][0])