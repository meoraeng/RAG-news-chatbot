import os
from dotenv import load_dotenv
load_dotenv()

import chromadb
import pandas as pd
from tabulate import tabulate
from langchain_upstage import UpstageEmbeddings

# --- 설정 ---
CHROMA_DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_db"))
COLLECTION_NAME = "articles"
EMBEDDING_MODEL = "embedding-passage"

# --- ChromaDB 클라이언트 초기화 ---
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

def get_db_statistics():
    """
    ChromaDB에서 데이터를 가져와 임베딩 및 문서 통계를 계산하고 테이블 형식으로 출력합니다.
    """
    try:
        print(f"Connecting to ChromaDB at: {CHROMA_DB_DIR}")
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"Successfully loaded '{COLLECTION_NAME}' collection.")
    except Exception as e:
        print(f"Error loading collection: {e}")
        print("Please ensure the crawler has been run and the database is populated.")
        return

    total_chunks = collection.count()
    if total_chunks == 0:
        print("The collection is empty. No statistics to generate.")
        return

    print(f"Fetching {total_chunks} document chunks from the database in batches...")
    
    batch_size = 500  # 한 번에 가져올 데이터 수
    all_metadatas = []
    all_documents = []
    all_ids = []

    print("\n--- Step 1: Fetching IDs and Metadatas in batches (using offset) ---")
    for offset in range(0, total_chunks, batch_size):
        print(f"  - Fetching metadata batch from offset {offset}...")
        batch_data = collection.get(
            limit=min(batch_size, total_chunks - offset),
            offset=offset,
            include=["metadatas"]
        )
        all_ids.extend(batch_data['ids'])
        all_metadatas.extend(batch_data['metadatas'])

    print("\n--- Step 2: Fetching Documents in batches (using IDs) ---")
    for i in range(0, total_chunks, batch_size):
        batch_ids = all_ids[i:i + batch_size]
        print(f"  - Fetching documents for {len(batch_ids)} IDs...")
        batch_docs = collection.get(
            ids=batch_ids,
            include=["documents"]
        )
        # ID로 조회 시 순서가 보장되므로 그대로 추가합니다.
        all_documents.extend(batch_docs['documents'])

    print("\nAll data has been successfully fetched.")
    
    # --- 1. 기본 통계 정보 ---
    print("\n" + "="*50)
    print("                DB 및 임베딩 기본 정보")
    print("="*50)

    # Upstage의 'embedding-passage' 모델은 공식적으로 1024차원입니다.
    # ChromaDB에 저장된 벡터의 바이트 길이(4096 bytes)가 아닌 실제 차원 수를 명시합니다.
    embedding_dim = 1024
    
    basic_stats_data = [
        ["DB 경로", CHROMA_DB_DIR],
        ["컬렉션 이름", COLLECTION_NAME],
        ["총 문서 조각(Chunk) 수", f"{total_chunks} 개"],
        ["임베딩 모델", EMBEDDING_MODEL],
        ["임베딩 벡터 차원", f"{embedding_dim} 차원"],
    ]
    print(tabulate(basic_stats_data, headers=["항목", "정보"], tablefmt="pretty"))


    # Pandas DataFrame으로 데이터 변환
    df = pd.DataFrame({
        'id': all_ids,
        'document': all_documents,
        'platform': [m.get('platform', 'N/A') for m in all_metadatas],
        'url': [m.get('url', 'N/A') for m in all_metadatas],
        'category': [m.get('category', 'N/A') for m in all_metadatas],
    })
    df['char_count'] = df['document'].str.len()
    
    # --- 2. 문서 조각(Chunk) 통계 ---
    print("\n" + "="*50)
    print("                 문서 조각(Chunk) 통계")
    print("="*50)
    chunk_stats_data = [
        ["평균 글자 수", f"{df['char_count'].mean():.2f} 자"],
        ["최소 글자 수", f"{df['char_count'].min()} 자"],
        ["최대 글자 수", f"{df['char_count'].max()} 자"],
    ]
    print(tabulate(chunk_stats_data, headers=["항목", "값"], tablefmt="pretty"))


    # --- 3. 데이터 소스별 통계 ---
    print("\n" + "="*50)
    print("                  데이터 소스별 통계")
    print("="*50)
    
    # 소스별 고유 기사 수 계산
    unique_articles_per_platform = df.groupby('platform')['url'].nunique().reset_index()
    unique_articles_per_platform.columns = ['플랫폼', '고유 기사 수']

    # 소스별 Chunk 수 계산
    chunks_per_platform = df.groupby('platform').size().reset_index(name='Chunk 수')
    chunks_per_platform.columns = ['플랫폼', 'Chunk 수']

    # 두 데이터프레임 병합
    platform_stats = pd.merge(unique_articles_per_platform, chunks_per_platform, on='플랫폼')
    
    # 총계 추가
    total_unique_articles = df['url'].nunique()
    total_chunks_sum = platform_stats['Chunk 수'].sum()
    total_row = pd.DataFrame([{'플랫폼': '총계', '고유 기사 수': total_unique_articles, 'Chunk 수': total_chunks_sum}])
    platform_stats = pd.concat([platform_stats, total_row], ignore_index=True)
    
    print(tabulate(platform_stats, headers='keys', tablefmt='psql', showindex=False))
    
    # --- 4. 플랫폼 및 카테고리별 통계 ---
    print("\n" + "="*50)
    print("              플랫폼 및 카테고리별 상세 통계")
    print("="*50)

    # 플랫폼과 카테고리별로 고유 기사 수와 Chunk 수 계산
    detailed_stats = df.groupby(['platform', 'category']).agg(
        unique_articles=('url', 'nunique'),
        chunk_count=('id', 'count')
    ).reset_index()
    
    detailed_stats.columns = ['플랫폼', '카테고리', '고유 기사 수', 'Chunk 수']
    
    print(tabulate(detailed_stats.sort_values(by=['플랫폼', '카테고리']), headers='keys', tablefmt='psql', showindex=False))

if __name__ == "__main__":
    get_db_statistics() 