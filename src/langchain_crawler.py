import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
import requests
from typing import List, Dict, Optional
import time
import random
from datetime import datetime
import chromadb
from chromadb.config import Settings
from langchain_upstage import UpstageEmbeddings
import tiktoken
import json
import re

# 로깅 설정
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "crawler.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 핸들러 중복 추가 방지
if not logger.handlers:
    # 파일 핸들러 (UTF-8 인코딩, 매 실행 시 새로 작성)
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 포매터
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# 환경 변수 로드
load_dotenv()

# ChromaDB 설정
CHROMA_DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_db"))
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

# 진행 상태 저장 파일
PROGRESS_FILE = "crawler_progress.json"

def load_progress():
    """이전 크롤링 진행 상태를 불러온다"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"processed_urls": [], "current_category": None, "current_page": None}

def save_progress(progress):
    """크롤링 진행 상태 저장"""
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

# ChromaDB 클라이언트 초기화
client = chromadb.PersistentClient(
    path=CHROMA_DB_DIR,
    settings=Settings(
        anonymized_telemetry=False,
        is_persistent=True
    )
)

# Upstage 임베딩 모델 래퍼 클래스 (ChromaDB용)
class UpstageEmbeddingFunction:
    def __init__(self):
        self.model = UpstageEmbeddings(model="embedding-passage")
    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.model.embed_documents(input)
    def name(self):
        return "upstage-embedding-passage"

embedding_function = UpstageEmbeddingFunction()

# ChromaDB 컬렉션 생성
try:
    client.delete_collection("articles")
except:
    pass
collection = client.get_or_create_collection(
    name="articles",
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"}
)

# 카테고리별 페이지 설정
CATEGORIES = {
    # itworld 플랫폼
    "cloud_computing": {
        "url": "https://www.itworld.co.kr/cloud-computing/page/",
        "pages": range(1, 61)  # 1-60페이지
        # "pages": [1]  # 테스트: 첫 페이지만
    },
    "artificial_intelligence": {
        "url": "https://www.itworld.co.kr/artificial-intelligence/page/",
        "pages": range(1, 61)  # 1-60페이지
        # "pages": [1]  # 테스트: 첫 페이지만
    },
    "software_development": {
        "url": "https://www.itworld.co.kr/software-development/page/",
        "pages": range(1, 19)  # 1-18페이지
        # "pages": [1]  # 테스트: 첫 페이지만
    },
    # zdnet 플랫폼
    "internet": {
        "url": "https://zdnet.co.kr/news/?lstcode=0040",
        "pages": range(1, 61, 2)  # 1-60페이지, 2페이지 간격
        # "pages": [1]  # 테스트: 첫 페이지만
    },
    "computing": {
        "url": "https://zdnet.co.kr/news/?lstcode=0020",
        "pages": range(1, 196, 5)  # 1-195페이지, 5페이지 간격
        # "pages": [1]  # 테스트: 첫 페이지만
    },
    "broadcasting": {
        "url": "https://zdnet.co.kr/news/?lstcode=0010",
        "pages": range(1, 72, 2)  # 1-71페이지, 2페이지 간격
        # "pages": [1]  # 테스트: 첫 페이지만
    }
}

# User-Agent 설정
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
]

def get_random_user_agent():
    return random.choice(USER_AGENTS)

def create_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': get_random_user_agent(),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    })
    return session

def get_article_links(url: str, max_retries: int = 3) -> List[str]:
    """ZDNet과 ITWorld에서 기사 링크 수집"""
    session = create_session()
    links = []
    
    for attempt in range(max_retries):
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # ZDNet 기사 링크 추출
            if "zdnet.co.kr" in url:
                articles = soup.select('div.news_box div.newsPost')
                for article in articles:
                    try:
                        link_elem = article.select_one('div.assetText a')
                        if link_elem and link_elem.get('href'):
                            href = link_elem['href']
                            full_url = f"https://zdnet.co.kr{href}" if href.startswith('/') else href
                            links.append(full_url)
                    except Exception as e:
                        logger.warning(f"ZDNet 링크 추출 중 오류: {str(e)}")
                        continue
            
            # ITWorld 기사 링크 추출
            elif "itworld.co.kr" in url:
                articles = soup.select('div.content-listing-various__row a.content-row-article')
                for article in articles:
                    try:
                        if article.get('href'):
                            href = article['href']
                            links.append(href)
                    except Exception as e:
                        logger.warning(f"ITWorld 링크 추출 중 오류: {str(e)}")
                        continue
            
            if links:
                logger.info(f"페이지 {url}에서 발견된 기사 수: {len(links)}개")
                break
                
        except Exception as e:
            logger.warning(f"링크 수집 시도 {attempt + 1}/{max_retries} 실패: {url}, 오류: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 지수 백오프
            continue
            
    return links

def split_text_into_sentences(text: str) -> List[str]:
    """텍스트를 문장 단위로 분리"""
    # 문장 종결 부호로 분리
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def split_text_with_overlap(text: str, max_tokens: int = 1000, overlap: int = 100) -> List[Dict]:
    """텍스트를 오버랩을 포함하여 분할"""
    encoding = tiktoken.get_encoding("cl100k_base")
    sentences = split_text_into_sentences(text)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = len(encoding.encode(sentence))
        
        if current_tokens + sentence_tokens > max_tokens:
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "tokens": current_tokens
                })
                # 오버랩을 위해 마지막 문장들을 유지
                overlap_tokens = 0
                current_chunk = []
                for s in reversed(current_chunk):
                    s_tokens = len(encoding.encode(s))
                    if overlap_tokens + s_tokens <= overlap:
                        current_chunk.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break
                current_tokens = overlap_tokens
            else:
                # 문장이 max_tokens보다 큰 경우
                chunks.append({
                    "text": sentence,
                    "tokens": sentence_tokens
                })
                current_chunk = []
                current_tokens = 0
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
    
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append({
            "text": chunk_text,
            "tokens": current_tokens
        })
    
    return chunks

def extract_article_content(url: str, max_retries: int = 3) -> Optional[Dict]:
    """기사 내용 추출"""
    session = create_session()
    
    for attempt in range(max_retries):
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 플랫폼 정보 설정
            platform = "ZDNet" if "zdnet.co.kr" in url else "ITWorld"
            
            # ZDNet 기사 내용 추출
            if "zdnet.co.kr" in url:
                title = soup.select_one('div.news_head h1')
                title = title.text.strip() if title else "제목 없음"
                
                summary = soup.select_one('div.assetText p')
                summary = summary.text.strip() if summary else ""
                
                content_elements = soup.select('div.view_cont p')
                content = "\n".join([elem.text.strip() for elem in content_elements if elem.text.strip()])
                
                meta = soup.select_one('div.news_head p.meta')
                date = meta.select_one('span').text.strip() if meta else datetime.now().strftime("%Y-%m-%d")
                author = soup.select_one('div.reporter_list strong')
                author = author.text.strip() if author else "작자미상"
                
                category = meta.select_one('a')
                category = category.text.strip() if category else "카테고리 없음"
            
            # ITWorld 기사 내용 추출
            elif "itworld.co.kr" in url:
                title = soup.select_one('h1.article-hero__title')
                title = title.text.strip() if title else "제목 없음"
                
                summary = soup.select_one('h2.content-subheadline')
                summary = summary.text.strip() if summary else ""
                
                content_elements = soup.select('div.article-column__content p')
                content = "\n".join([elem.text.strip() for elem in content_elements if elem.text.strip()])
                
                date = soup.select_one('div.card__info--light span')
                date = date.text.strip() if date else datetime.now().strftime("%Y-%m-%d")
                
                author = soup.select_one('div.author__name a')
                author = author.text.strip() if author else "작자미상"
                
                category = soup.select_one('div.card__tags span.tag')
                category = category.text.strip() if category else "카테고리 없음"
            
            # 텍스트 길이 제한 (10,000자)
            if len(content) > 10000:
                content = content[:10000]
            
            return {
                "title": title,
                "summary": summary,
                "date": date,
                "author": author,
                "category": category,
                "content": content,
                "url": url,
                "platform": platform  # 플랫폼 정보 추가
            }
            
        except Exception as e:
            logger.error(f"기사 내용 추출 시도 {attempt + 1}/{max_retries} 실패: {url}, 오류: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            continue
    
    return None

def process_articles():
    """기사 처리 및 저장"""
    total_processed = 0
    total_success = 0
    total_failed = 0
    
    # 진행 상태 불러오기
    progress = load_progress()
    processed_urls = set(progress["processed_urls"])
    
    logger.info(f"현재까지 처리된 URL 수: {len(processed_urls)}개")
    
    # 이전에 처리하던 카테고리와 페이지부터 시작
    start_category = progress.get("current_category")
    start_page = progress.get("current_page")
    
    if start_category and start_page:
        logger.info(f"이전 진행 상태에서 재시작: {start_category} 카테고리, {start_page} 페이지")
    
    categories = list(CATEGORIES.items())
    if start_category:
        # 이전 카테고리부터 시작
        start_idx = next((i for i, (cat, _) in enumerate(categories) if cat == start_category), 0)
        categories = categories[start_idx:]
    
    for category, config in categories:
        logger.info(f"\n{'='*50}")
        logger.info(f"{category} 카테고리 처리 중...")
        logger.info(f"{'='*50}")
        progress["current_category"] = category
        
        pages = list(config["pages"])
        if start_page and category == start_category:
            # 이전 페이지부터 시작
            start_idx = next((i for i, p in enumerate(pages) if p == start_page), 0)
            pages = pages[start_idx:]
        
        category_success = 0
        category_failed = 0
        
        for page in pages:
            logger.info(f"\n{'='*30}")
            logger.info(f"페이지 {page} 처리 중...")
            logger.info(f"{'='*30}")
            progress["current_page"] = page
            save_progress(progress)  # 진행 상태 저장
            
            # ITWorld와 ZDNet의 URL 구조에 따라 다르게 처리
            if "itworld.co.kr" in config["url"]:
                page_url = f"{config['url']}{page}/"
            else:
                page_url = f"{config['url']}&page={page}"
            
            try:
                # 기사 링크 수집
                article_links = get_article_links(page_url)
                if not article_links:
                    logger.info(f"페이지 {page_url}에서 기사를 찾을 수 없습니다.")
                    continue
                
                logger.info(f"발견된 기사 수: {len(article_links)}개")
                
                # 각 기사 처리
                for url in article_links:
                    if url in processed_urls:
                        logger.info(f"이미 처리된 기사 건너뛰기: {url}")
                        continue
                        
                    logger.info(f"기사 처리 중: {url}")
                    total_processed += 1
                    
                    try:
                        # 기사 내용 추출
                        article_data = extract_article_content(url)
                        if not article_data:
                            logger.warning(f"기사 내용을 추출할 수 없습니다: {url}")
                            total_failed += 1
                            category_failed += 1
                            continue
                        
                        # 텍스트 분할
                        chunks = split_text_with_overlap(article_data["content"])
                        
                        # 각 청크에 메타데이터 추가
                        for i, chunk in enumerate(chunks):
                            chunk["metadata"] = {
                                "title": article_data["title"],
                                "summary": article_data["summary"],
                                "date": article_data["date"],
                                "author": article_data["author"],
                                "category": category,  # 대분류 카테고리 저장
                                "sub_category": article_data["category"],  # 기존 세부 카테고리는 sub_category로 저장
                                "url": article_data["url"],
                                "platform": article_data["platform"],
                                "chunk_index": i,
                                "total_chunks": len(chunks)
                            }
                        
                        # ChromaDB에 저장
                        for chunk in chunks:
                            collection.add(
                                documents=[chunk["text"]],
                                metadatas=[chunk["metadata"]],
                                ids=[f"{article_data['url']}_{chunk['metadata']['chunk_index']}"]
                            )
                        
                        logger.info(f"기사 처리 완료: {article_data['title']} ({article_data['platform']})")
                        total_success += 1
                        category_success += 1
                        processed_urls.add(url)
                        progress["processed_urls"] = list(processed_urls)
                        save_progress(progress)  # 진행 상태 저장
                        
                    except Exception as e:
                        logger.error(f"기사 처리 중 오류 발생: {url}, 오류: {str(e)}")
                        total_failed += 1
                        category_failed += 1
                        continue
                    
                    # 요청 간 딜레이
                    time.sleep(random.uniform(1, 3))
                
                logger.info(f"페이지 {page} 처리 완료")
                logger.info(f"현재까지 성공: {category_success}개, 실패: {category_failed}개")
                
            except Exception as e:
                logger.error(f"페이지 {page_url} 처리 중 오류 발생: {str(e)}")
                continue
            
            # 페이지 간 딜레이
            time.sleep(random.uniform(2, 5))
        
        logger.info(f"\n{category} 카테고리 처리 완료:")
        logger.info(f"성공: {category_success}개")
        logger.info(f"실패: {category_failed}개")
    
    logger.info(f"\n전체 크롤링 완료:")
    logger.info(f"총 처리된 기사: {total_processed}개")
    logger.info(f"성공한 기사: {total_success}개")
    logger.info(f"실패한 기사: {total_failed}개")
    
    # 크롤링 완료 후 진행 상태 파일 삭제
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

    # 크롤링 후 임베딩 차원 확인
    test_emb = embedding_function([f"테스트 문장입니다."])[0]
    logger.info(f"임베딩 차원: {len(test_emb)}")
    logger.info(f"컬렉션 내 문서 개수: {collection.count()}")
    logger.info(f"컬렉션 메타데이터: {collection.metadata}")
    logger.info(f"실제 chroma_db 경로: {CHROMA_DB_DIR}")
    if "dimension" in collection.metadata:
        logger.info(f"컬렉션 임베딩 차원: {collection.metadata['dimension']}")

if __name__ == "__main__":
    process_articles()
    # 크롤링 후 컬렉션에 저장된 문서 개수 출력
    logger.info(f"최종 컬렉션 내 문서 개수: {collection.count()}") 