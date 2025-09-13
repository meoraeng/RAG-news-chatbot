import requests
from bs4 import BeautifulSoup
import time
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import random
import tiktoken
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import gc
import numpy as np
from datetime import datetime
import chromadb
from chromadb.config import Settings

# .env 파일 로드
load_dotenv()

# OpenAI API 키 설정
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class OpenAIEmbeddingFunction:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def __call__(self, input: str) -> list:
        """텍스트를 임베딩 벡터로 변환합니다."""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=input
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"임베딩 생성 중 오류 발생: {str(e)}")
            return None

# ChromaDB 클라이언트 초기화
chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(
        anonymized_telemetry=False,
        is_persistent=True
    )
)

# 컬렉션 생성 또는 가져오기
collection = chroma_client.get_or_create_collection(
    name="zdnet_articles",
    metadata={"hnsw:space": "cosine"},
    embedding_function=OpenAIEmbeddingFunction()
)

def split_into_sentences(text):
    """텍스트를 문장 단위로 분리"""
    pattern = r'[.!?。]\s+|[.!?。](?=[A-Z])'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]

def split_text_with_overlap(text, max_tokens=3000, overlap_tokens=500):
    """텍스트를 오버랩핑하여 나누기"""
    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_tokens = encoding.encode(sentence)
        sentence_length = len(sentence_tokens)
        
        if sentence_length > max_tokens:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            while sentence_tokens:
                chunk_tokens = sentence_tokens[:max_tokens]
                chunk_text = encoding.decode(chunk_tokens)
                chunks.append(chunk_text)
                sentence_tokens = sentence_tokens[max_tokens-overlap_tokens:]
            continue
        
        if current_length + sentence_length > max_tokens:
            chunks.append(' '.join(current_chunk))
            
            overlap_start = -1
            for i in range(len(current_chunk)-1, -1, -1):
                overlap_text = ' '.join(current_chunk[i:])
                if len(encoding.encode(overlap_text)) <= overlap_tokens:
                    overlap_start = i
                    break
            
            if overlap_start != -1:
                current_chunk = current_chunk[overlap_start:]
            else:
                current_chunk = []
            current_length = sum(len(encoding.encode(s)) for s in current_chunk)
        
        current_chunk.append(sentence)
        current_length += sentence_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def get_embeddings_with_overlap(text):
    """오버랩핑된 텍스트에 대한 임베딩 생성"""
    try:
        print("텍스트를 청크로 나누는 중...")
        chunks = split_text_with_overlap(text)
        print(f"총 {len(chunks)}개의 청크로 나눔")
        embeddings = []
        
        for i, chunk in enumerate(chunks, 1):
            print(f"청크 {i}/{len(chunks)} 임베딩 생성 중...")
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=chunk
            )
            embeddings.append({
                "text": chunk,
                "embedding": response.data[0].embedding
            })
            print(f"청크 {i} 임베딩 생성 완료")
        
        return embeddings
    except Exception as e:
        print(f"임베딩 생성 실패: {str(e)}")
        return None

def create_webdriver():
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920x1080')
    chrome_options.add_argument('--disable-notifications')
    chrome_options.add_argument('--disable-extensions')
    chrome_options.add_argument('--disable-infobars')
    chrome_options.add_argument('--disable-web-security')
    chrome_options.add_argument('--disable-features=IsolateOrigins,site-per-process')
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.set_page_load_timeout(30)  # 페이지 로드 타임아웃을 30초로 설정
    driver.implicitly_wait(10)  # 암시적 대기 시간을 10초로 설정
    return driver

def get_article_links(driver, url):
    """기사 링크를 수집합니다."""
    try:
        driver.get(url)
        
        # 페이지 로딩을 위한 명시적 대기
        wait = WebDriverWait(driver, 30)  # 30초로 증가
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.news_box")))
        
        # 추가 대기 시간
        time.sleep(5)
        
        articles = driver.find_elements(By.CSS_SELECTOR, 'div.news_box div.newsPost')
        print(f"발견된 기사 링크 수: {len(articles)}")
        
        links = []
        for article in articles:
            try:
                link = article.find_element(By.CSS_SELECTOR, 'div.assetText a').get_attribute('href')
                links.append(link)
            except Exception as e:
                print(f"링크 추출 중 오류: {str(e)}")
                continue
                
        return links
    except Exception as e:
        print(f"링크 수집 중 오류 발생: {str(e)}")
        return []

def extract_article_content(driver, url, max_retries=3):
    """기사 내용을 추출합니다."""
    for attempt in range(max_retries):
        try:
            driver.get(url)
            time.sleep(3)  # 페이지 로딩을 위한 대기 시간
            
            # 제목 추출
            title = driver.find_element(By.CSS_SELECTOR, "div.news_head h1").text.strip()
            
            # 요약 추출 (기사 목록에서)
            try:
                summary = driver.find_element(By.CSS_SELECTOR, "div.assetText p").text.strip()
            except:
                summary = ""
            
            # 내용 추출
            content_elements = driver.find_elements(By.CSS_SELECTOR, "div.view_cont p")
            content = "\n".join([elem.text.strip() for elem in content_elements if elem.text.strip()])
            
            # 날짜와 작성자 추출
            meta = driver.find_element(By.CSS_SELECTOR, "div.news_head p.meta")
            date = meta.find_element(By.CSS_SELECTOR, "span").text.strip()
            author = driver.find_element(By.CSS_SELECTOR, "div.reporter_list strong").text.strip()
            
            # 카테고리 추출
            category = meta.find_element(By.CSS_SELECTOR, "a").text.strip()
            
            # 텍스트 길이 제한 (10,000자)
            if len(content) > 10000:
                content = content[:10000]
                
            return {
                "title": title,
                "summary": summary,
                "content": content,
                "date": date,
                "author": author,
                "category": category,
                "url": url
            }
        except Exception as e:
            print(f"시도 {attempt + 1}/{max_retries} 실패: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(5)  # 재시도 전 대기
                continue
            else:
                print(f"기사 정보 추출 중 오류 발생: {str(e)}")
                return None

def save_to_chroma(article_data, embeddings):
    """기사 데이터를 ChromaDB에 저장"""
    try:
        # 메타데이터 준비
        metadata = {
            "title": article_data['title'],
            "url": article_data['url'],
            "date": article_data['date'],
            "author": article_data['author'],
            "category": article_data['category'],
            "source": "zdnet",
            "created_at": datetime.now().isoformat()
        }
        
        # 각 청크별로 ChromaDB에 저장
        for i, chunk_data in enumerate(embeddings):
            collection.add(
                embeddings=[chunk_data['embedding']],
                documents=[chunk_data['text']],
                metadatas=[metadata],
                ids=[f"{article_data['url']}_{i}"]
            )
        
        print(f"ChromaDB 저장 완료: {article_data['title']}")
        return True
        
    except Exception as e:
        print(f"ChromaDB 저장 중 오류 발생: {str(e)}")
        return False

def generate_embedding_for_article(article):
    """기사에 대한 임베딩을 생성하고 ChromaDB에 저장"""
    try:
        # 텍스트 준비
        text = f"{article['title']}\n{article['content']}"
        
        # 오버랩핑된 임베딩 생성
        embeddings = get_embeddings_with_overlap(text)
        if not embeddings:
            return None
        
        # ChromaDB에 저장
        if save_to_chroma(article, embeddings):
            return True
        return False
        
    except Exception as e:
        print(f"임베딩 생성 중 오류 발생: {str(e)}")
        return None

def scrape_zdnet_articles():
    """ZDNet 기사를 스크래핑합니다."""
    # 테스트용: 1페이지만 크롤링
    test_category = {
        "name": "인터넷",
        "base_url": "https://zdnet.co.kr/news/?lstcode=0040&page={}",
        "start_page": 1,
        "end_page": 1,  # 1페이지만
        "interval": 1
    }
    
    print(f"\n{test_category['name']} 카테고리 처리 중...")
    
    # 각 페이지마다 새로운 WebDriver 인스턴스 생성
    driver = None
    try:
        driver = create_webdriver()
        url = test_category['base_url'].format(test_category['start_page'])
        links = get_article_links(driver, url)
        
        success_count = 0
        fail_count = 0
        
        for link in links:
            try:
                article_data = extract_article_content(driver, link)
                if article_data:
                    article_with_embedding = generate_embedding_for_article(article_data)
                    if article_with_embedding:
                        success_count += 1
                    else:
                        fail_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(f"기사 처리 중 오류: {str(e)}")
                fail_count += 1
                continue
        
        print(f"페이지 {test_category['start_page']} 처리 완료: 성공 {success_count}개, 실패 {fail_count}개")
        
    except Exception as e:
        print(f"페이지 {test_category['start_page']} 처리 중 오류: {str(e)}")
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass

# 기존 코드 주석 처리
"""
def scrape_zdnet_articles():
    categories = [
        {
            "name": "인터넷",
            "base_url": "https://zdnet.co.kr/news/?lstcode=0040&page={}",
            "start_page": 1,
            "end_page": 60,
            "interval": 2
        },
        {
            "name": "컴퓨팅",
            "base_url": "https://zdnet.co.kr/news/?lstcode=0020&page={}",
            "start_page": 1,
            "end_page": 195,
            "interval": 5
        },
        {
            "name": "방송/통신",
            "base_url": "https://zdnet.co.kr/news/?lstcode=0010&page={}",
            "start_page": 1,
            "end_page": 71,
            "interval": 2
        }
    ]
    
    total_processed = 0
    total_failed = 0
    
    for category in categories:
        print(f"\n{category['name']} 카테고리 처리 중...")
        
        for page in range(category['start_page'], category['end_page'] + 1, category['interval']):
            print(f"\n페이지 {page} 처리 중...")
            
            # 각 페이지마다 새로운 WebDriver 인스턴스 생성
            driver = None
            try:
                driver = create_webdriver()
                url = category['base_url'].format(page)
                links = get_article_links(driver, url)
                
                success_count = 0
                fail_count = 0
                
                for link in links:
                    try:
                        article_data = extract_article_content(driver, link)
                        if article_data:
                            article_with_embedding = generate_embedding_for_article(article_data)
                            if article_with_embedding:
                                success_count += 1
                                total_processed += 1
                            else:
                                fail_count += 1
                                total_failed += 1
                        else:
                            fail_count += 1
                            total_failed += 1
                    except Exception as e:
                        print(f"기사 처리 중 오류: {str(e)}")
                        fail_count += 1
                        total_failed += 1
                        continue
                
                print(f"페이지 {page} 처리 완료: 성공 {success_count}개, 실패 {fail_count}개")
                
            except Exception as e:
                print(f"페이지 {page} 처리 중 오류: {str(e)}")
                total_failed += 1
            finally:
                if driver:
                    try:
                        driver.quit()
                    except:
                        pass
                time.sleep(3)  # 페이지 간 딜레이
    
    print(f"\n총 처리된 기사: {total_processed}개")
    print(f"총 실패한 기사: {total_failed}개")
"""

if __name__ == "__main__":
    scrape_zdnet_articles() 