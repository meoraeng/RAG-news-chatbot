import os
from dotenv import load_dotenv

# load_dotenv()를 rag_chatbot 보다 먼저 호출해야
# rag_chatbot 내부의 UpstageEmbeddings가 API 키를 제대로 인식합니다.
load_dotenv()

from rag_chatbot import vectorstore, retriever

def debug_retriever(query):
    """
    주어진 쿼리에 대해 vectorstore와 retriever가 어떻게 반응하는지 디버깅합니다.
    """
    print(f"--- 쿼리: '{query}' 로 디버깅 시작 ---")
    
    print("\n[1] Vectorstore (유사도 점수 확인 - 임계값 미적용)")
    # retriever의 임계값(threshold)을 적용하지 않고, 순수하게 가장 유사한 문서 5개와 점수를 확인합니다.
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=5)
    
    if not docs_with_scores:
        print(" -> Vectorstore에서 문서를 전혀 찾지 못했습니다.")
    else:
        for i, (doc, score) in enumerate(docs_with_scores):
            print(f"  {i+1}. 점수: {score:.4f} | 제목: {doc.metadata.get('title', 'N/A')}")
            # print(f"     내용: {doc.page_content[:80]}...")
    
    print("\n[2] Retriever (설정된 임계값 적용)")
    # 'rag_chatbot.py'에 설정된 임계값(0.35)이 적용된 결과를 확인합니다.
    retrieved_docs = retriever.invoke(query)

    if not retrieved_docs:
        print(" -> Retriever가 반환한 문서가 없습니다. (점수 임계값 미달)")
    else:
        print(f" -> Retriever가 {len(retrieved_docs)}개의 문서를 반환했습니다.")
        for i, doc in enumerate(retrieved_docs):
            print(f"  {i+1}. 제목: {doc.metadata.get('title', 'N/A')}")

    print("\n--- 디버깅 종료 ---")


if __name__ == "__main__":
    # 테스트할 질문
    test_query_1 = "SKT 유심 사태 관련 소식은 무엇이 있나요"
    test_query_2 = "가장 인기있는 반려동물은 무엇인가요"
    
    debug_retriever(test_query_1)
    print("\n" + "="*50 + "\n")
    debug_retriever(test_query_2) 