import os
import chromadb
from bs4 import BeautifulSoup
import requests
from typing import List
import time
import random

# --- langchain_crawler.py에서 가져온 설정 및 함수 ---

# ChromaDB 설정
CHROMA_DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_db"))
print(f"DEBUG: 데이터베이스 경로 확인: {CHROMA_DB_DIR}")

# 카테고리별 페이지 설정
CATEGORIES = {
    "cloud_computing": {"url": "https://www.itworld.co.kr/cloud-computing/page/", "pages": range(1, 61)},
    "artificial_intelligence": {"url": "https://www.itworld.co.kr/artificial-intelligence/page/", "pages": range(1, 61)},
    "software_development": {"url": "https://www.itworld.co.kr/software-development/page/", "pages": range(1, 19)},
    "internet": {"url": "https://zdnet.co.kr/news/?lstcode=0040", "pages": range(1, 61, 2)},
    "computing": {"url": "https://zdnet.co.kr/news/?lstcode=0020", "pages": range(1, 196, 5)},
    "broadcasting": {"url": "https://zdnet.co.kr/news/?lstcode=0010", "pages": range(1, 72, 2)},
}

# User-Agent 설정
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
]

def get_random_user_agent():
    return random.choice(USER_AGENTS)

def create_session():
    session = requests.Session()
    session.headers.update({'User-Agent': get_random_user_agent()})
    return session

def get_article_links(url: str, max_retries: int = 3) -> List[str]:
    session = create_session()
    links = []
    for attempt in range(max_retries):
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            if "zdnet.co.kr" in url:
                articles = soup.select('div.news_box div.newsPost')
                for article in articles:
                    link_elem = article.select_one('div.assetText a')
                    if link_elem and link_elem.get('href'):
                        href = link_elem['href']
                        full_url = f"https://zdnet.co.kr{href}" if href.startswith('/') else href
                        links.append(full_url)
            elif "itworld.co.kr" in url:
                articles = soup.select('div.content-listing-various__row a.content-row-article')
                for article in articles:
                    if article.get('href'):
                        links.append(article['href'])
            
            if links: break
        except Exception as e:
            print(f"링크 수집 시도 {attempt + 1}/{max_retries} 실패: {str(e)}")
            if attempt < max_retries - 1: time.sleep(2 ** attempt)
            continue
    return links

# --- 메타데이터 업데이트 주 로직 ---

def update_categories_in_db():
    """DB의 모든 문서에 대해 대분류 카테고리를 업데이트합니다."""
    
    print("ChromaDB에 연결 중...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    try:
        collection = client.get_collection(name="articles")
        print(f"'{collection.name}' 컬렉션을 성공적으로 불러왔습니다. 총 문서 수: {collection.count()}")
    except Exception as e:
        print(f"컬렉션 로드 실패: {e}")
        return

    for category_name, config in CATEGORIES.items():
        print(f"\n{'='*50}")
        print(f"대분류 카테고리 '{category_name}' 처리 시작...")
        
        all_links_for_category = []
        pages = list(config["pages"])
        
        for i, page in enumerate(pages):
            if "itworld.co.kr" in config["url"]:
                page_url = f"{config['url']}{page}/"
            else:
                page_url = f"{config['url']}&page={page}"
            
            print(f"  - 페이지 링크 수집 중... ({i+1}/{len(pages)})", end='\r')
            article_links = get_article_links(page_url)
            all_links_for_category.extend(article_links)
            time.sleep(random.uniform(0.5, 1.5)) # 서버 부하 감소
        
        unique_links = sorted(list(set(all_links_for_category)))
        print(f"\n'{category_name}' 카테고리에서 {len(unique_links)}개의 고유 기사 링크를 수집했습니다.")

        if not unique_links:
            print("처리할 링크가 없으므로 다음 카테고리로 넘어갑니다.")
            continue

        try:
            # 해당 URL을 가진 모든 문서 조각(chunk)을 DB에서 가져옴
            results = collection.get(where={"url": {"$in": unique_links}}, include=["metadatas"])
            
            ids_to_update = results['ids']
            metadatas_to_update = results['metadatas']

            if not ids_to_update:
                print("DB에서 일치하는 문서를 찾지 못했습니다. 다음으로 넘어갑니다.")
                continue

            print(f"{len(ids_to_update)}개의 문서 조각에 대한 메타데이터를 업데이트합니다.")
            
            updated_metadatas = []
            for metadata in metadatas_to_update:
                new_metadata = metadata.copy()
                # 기존 category를 sub_category로 이동하고, 새로운 대분류 category 추가
                new_metadata['sub_category'] = new_metadata.get('category', 'N/A')
                new_metadata['category'] = category_name
                updated_metadatas.append(new_metadata)

            # DB에 메타데이터 일괄 업데이트
            collection.update(ids=ids_to_update, metadatas=updated_metadatas)
            print(f"'{category_name}' 카테고리 메타데이터 업데이트 완료!")

        except Exception as e:
            print(f"'{category_name}' 처리 중 DB 오류 발생: {e}")

    print(f"\n{'='*50}")
    print("모든 카테고리의 메타데이터 업데이트가 완료되었습니다.")

if __name__ == "__main__":
    update_categories_in_db() 