from dotenv import load_dotenv
import os
import chromadb
from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings
import tiktoken
import numpy as np
from tabulate import tabulate

# .env 파일 로드
load_dotenv(override=True)

# --- 설정 ---
CHROMA_DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_db"))

def get_database_metrics():
    """ChromaDB에 저장된 데이터의 정량적 지표를 계산하고 출력합니다."""

    print("="*80)
    print("           벡터 데이터베이스(Vector DB) 지표 분석 시작")
    print("="*80)

    # 1. DB 및 임베딩 모델 초기화
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        # 임베딩 모델은 지표 계산에 직접 사용되진 않지만, 정보 명시를 위해 로드합니다.
        embeddings = UpstageEmbeddings(model="embedding-passage")
        vectorstore = Chroma(
            client=client,
            collection_name="articles",
            embedding_function=embeddings
        )
        print("[1/3] DB 및 관련 모듈 초기화 완료.")
    except Exception as e:
        print(f"[오류] DB 초기화 중 오류 발생: {e}")
        return

    # 2. DB에서 모든 데이터 가져오기
    try:
        print("[2/3] DB에서 전체 데이터 로드 중...")
        # .get() 메서드는 collection의 모든 아이템을 가져옵니다.
        db_contents = vectorstore._collection.get(include=["metadatas", "documents"])
        metadatas = db_contents.get('metadatas', [])
        documents = db_contents.get('documents', [])
        print(f"  - 총 {len(documents)}개의 청크(Chunk) 데이터를 성공적으로 로드했습니다.")
    except Exception as e:
        print(f"[오류] DB 데이터 로드 실패: {e}")
        return

    # 3. 지표 계산
    print("[3/3] 정량 지표 계산 중...")

    # a. 원본 문서 수: metadata의 'url' 필드의 고유(unique)한 개수를 셉니다.
    num_original_docs = "계산 불가"
    if metadatas:
        # 'url'이 있는 메타데이터만 필터링하여 고유한 url의 개수를 셉니다.
        unique_urls = set(m['url'] for m in metadatas if m and 'url' in m)
        num_original_docs = len(unique_urls)
    
    # b. 총 청크 수
    total_chunks = len(documents)

    # c. 임베딩 모델 정보
    embedding_model_name = "Upstage Solar (embedding-passage)"
    embedding_dim = 1024 # Upstage/embedding-passage 모델의 공식 차원

    # d. 평균 청크 당 토큰 수
    avg_tokens_per_chunk = "계산 불가"
    if documents:
        # 'cl100k_base'는 OpenAI의 gpt-3.5-turbo, gpt-4 등에서 사용하는 표준 토크나이저입니다.
        tokenizer = tiktoken.get_encoding("cl100k_base")
        token_counts = [len(tokenizer.encode(doc, disallowed_special=())) for doc in documents]
        avg_tokens_per_chunk = f"{np.mean(token_counts):.0f} 토큰" if token_counts else 0

    # e. 유사도 측정 방식
    similarity_metric = "코사인 유사도 (Cosine Similarity)"

    print("  - 지표 계산 완료!")

    # 4. 결과 출력
    print("\n" + "="*80)
    print("                     [최종 분석 결과]")
    print("="*80)

    headers = ["지표 (Metric)", "값 (Value)", "설명"]
    table_data = [
        ["원본 문서 수", f"{num_original_docs} 개", "수집한 전체 IT 기술 뉴스 기사의 수"],
        ["총 청크(Chunk) 수", f"{total_chunks} 개", "원본 문서를 의미 단위로 분할한 결과"],
        ["임베딩 모델", embedding_model_name, "한국어 특화 임베딩 모델"],
        ["임베딩 벡터 차원", f"{embedding_dim}", "각 청크를 표현하는 벡터의 크기"],
        ["평균 청크 당 토큰 수", avg_tokens_per_chunk, "LLM의 컨텍스트 길이를 고려한 단위"],
        ["유사도 측정 방식", similarity_metric, "두 벡터 간의 방향 유사성 측정"],
    ]
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("="*80)


if __name__ == "__main__":
    get_database_metrics() 