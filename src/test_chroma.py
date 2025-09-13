import os
from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings
from langchain_chroma import Chroma
import chromadb

# .env 파일 로드
load_dotenv(override=True)

print("ChromaDB 직접 연결 및 검색 테스트를 시작합니다...")

try:
    # 1. rag_chatbot.py와 동일한 방식으로 ChromaDB 클라이언트 및 벡터스토어를 초기화합니다.
    CHROMA_DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_db"))
    print(f"[INFO] ChromaDB 경로: {CHROMA_DB_DIR}")

    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    embeddings = UpstageEmbeddings(model="embedding-passage")

    vectorstore = Chroma(
        client=client,
        collection_name="articles",
        embedding_function=embeddings
    )
    
    retriever = vectorstore.as_retriever(
        k=3,
        search_type="similarity_score_threshold",
        search_kwargs={'score_threshold': 0.5}
    )
    print("[SUCCESS] Retriever가 성공적으로 초기화되었습니다.")
    print(f"[INFO] Retriever 정보: {retriever}")

    # 2. 가장 단순한 형태로 retriever를 직접 호출해봅니다.
    test_query = "인공지능"
    print(f"\n[TEST] 다음 쿼리로 검색을 시도합니다: '{test_query}'")
    
    # retriever.invoke()는 LangChain 0.1.0 이상에서 표준 방식입니다.
    # 이전 버전에서는 get_relevant_documents()를 사용했습니다.
    retrieved_docs = retriever.invoke(test_query)

    print(f"[SUCCESS] 검색이 성공적으로 완료되었습니다!")
    print(f"'{test_query}' 쿼리에 대해 {len(retrieved_docs)}개의 문서를 찾았습니다.")

    # 3. 검색된 문서의 일부를 출력합니다.
    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- 문서 {i+1} ---")
        print(f"내용: {doc.page_content[:200]}...")
        print(f"메타데이터: {doc.metadata}")

except Exception as e:
    print(f"\n[FATAL] 테스트 중 심각한 오류가 발생했습니다.")
    import traceback
    traceback.print_exc()

finally:
    print("\n테스트를 종료합니다.") 