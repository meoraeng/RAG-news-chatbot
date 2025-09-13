import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
import gc
import numpy as np
from datetime import datetime

load_dotenv()

def split_text_into_sentences(text, max_length=1000):
    """텍스트를 문장 단위로 분할합니다."""
    sentences = []
    current_sentence = ""
    
    for char in text:
        current_sentence += char
        if len(current_sentence) >= max_length and char in ['.', '!', '?', '。', '！', '？']:
            sentences.append(current_sentence.strip())
            current_sentence = ""
    
    if current_sentence:
        sentences.append(current_sentence.strip())
    
    return sentences

def handle_overlapping_text_chunks(text, chunk_size=1000, overlap=200):
    """텍스트를 겹치는 청크로 분할합니다."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        if end > len(text):
            end = len(text)
        chunks.append(text[start:end])
        start = end - overlap
    
    return chunks

def generate_embedding(text):
    """텍스트에 대한 임베딩을 생성합니다."""
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    try:
        # 텍스트를 더 작은 청크로 분할
        chunks = handle_overlapping_text_chunks(text, chunk_size=500, overlap=25)
        
        # 임베딩 벡터의 차원 수를 가져옴
        sample_response = client.embeddings.create(
            model="text-embedding-3-small",
            input="sample"
        )
        embedding_dim = len(sample_response.data[0].embedding)
        del sample_response
        
        # 결과를 저장할 배열 초기화
        embeddings_sum = [0.0] * embedding_dim
        chunk_count = 0
        
        # 청크를 순차적으로 처리
        for chunk in chunks:
            try:
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=chunk
                )
                embedding = response.data[0].embedding
                
                # 임베딩 합산
                for i in range(embedding_dim):
                    embeddings_sum[i] += embedding[i]
                chunk_count += 1
                
                # 메모리 해제
                del response
                del embedding
                gc.collect()
                
            except Exception as e:
                print(f"청크 처리 중 오류 발생: {str(e)}")
                continue
        
        # 평균 계산
        if chunk_count > 0:
            avg_embedding = [x / chunk_count for x in embeddings_sum]
            return avg_embedding
        return None
        
    except Exception as e:
        print(f"임베딩 생성 중 오류 발생: {str(e)}")
        return None

def create_webdriver():
    """WebDriver 인스턴스를 생성합니다."""
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-notifications')
    options.add_argument('--disable-web-security')
    options.add_argument('--disable-features=IsolateOrigins,site-per-process')
    options.add_argument('--disable-site-isolation-trials')
    options.add_argument('--window-size=1920x1080')
    options.add_argument('--disable-software-rasterizer')
    options.add_argument('--disable-infobars')
    options.add_argument('--disable-popup-blocking')
    options.add_argument('--memory-pressure-off')
    options.add_argument('--single-process')
    options.add_argument('--aggressive-cache-discard')
    
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(180)
    driver.implicitly_wait(20)
    return driver

def get_article_links(driver, url):
    """기사 링크를 수집합니다."""
    try:
        driver.get(url)
        time.sleep(5)
        
        links = []
        elements = driver.find_elements(By.CSS_SELECTOR, 'a[href*="/helloworld/"], a[href*="/news/"]')
        
        for element in elements:
            href = element.get_attribute('href')
            if href and ('/helloworld/' in href or '/news/' in href):
                if href not in links:
                    links.append(href)
        
        return links
    except Exception as e:
        print(f"링크 수집 중 오류 발생: {str(e)}")
        return []

def extract_article_content(driver, url):
    """기사 내용을 추출합니다."""
    try:
        driver.get(url)
        # 페이지 로딩 대기 시간 증가
        time.sleep(10)
        
        # 여러 가능한 제목 선택자 시도
        title_selectors = [
            'h1.posting_tit',
            'h1.title',
            'div.title h1',
            'div.content_title h1',
            'h1'
        ]
        
        title = None
        for selector in title_selectors:
            try:
                title_element = driver.find_element(By.CSS_SELECTOR, selector)
                title = title_element.text.strip()
                if title:
                    break
            except:
                continue
        
        if not title:
            print(f"제목을 찾을 수 없습니다: {url}")
            return None
        
        # 내용 추출 시도
        content_selectors = [
            'div.con_view p, div.con_view h4, div.con_view ol li',
            'div.content p, div.content h4, div.content ol li',
            'div.article_content p, div.article_content h4, div.article_content ol li'
        ]
        
        content = ""
        for selector in content_selectors:
            try:
                content_elements = driver.find_elements(By.CSS_SELECTOR, selector)
                content = '\n'.join([elem.text.strip() for elem in content_elements if elem.text.strip()])
                if content:
                    break
            except:
                continue
        
        if not content:
            print(f"내용을 찾을 수 없습니다: {url}")
            return None
        
        # 현재 날짜를 기본값으로 사용
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        return {
            "title": title,
            "content": content,
            "date": current_date,
            "author": "D2",
            "tags": [],
            "url": url
        }
        
    except Exception as e:
        print(f"내용 추출 중 오류 발생: {str(e)}")
        return None

def generate_embedding_for_article(article):
    """기사에 대한 임베딩을 생성합니다."""
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    try:
        # 텍스트 준비
        text = f"{article['title']}\n{article['content']}"
        
        # 텍스트가 너무 길 경우 처리
        max_text_length = 8000  # OpenAI의 토큰 제한을 고려
        if len(text) > max_text_length:
            text = text[:max_text_length]
        
        # 임베딩 생성
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        
        # 임베딩 벡터 추출
        embedding = response.data[0].embedding
        
        # 결과 데이터 구성
        result = {
            "metadata": {
                "title": article['title'],
                "url": article['url'],
                "date": article['date'],
                "author": article['author'],
                "tags": article['tags'],
                "source": "d2",
                "created_at": datetime.now().isoformat()
            },
            "content": {
                "text": text,
                "embedding": embedding
            }
        }
        
        return result
        
    except Exception as e:
        print(f"임베딩 생성 중 오류 발생: {str(e)}")
        return None

def save_article_data(article_data, output_dir="data"):
    """기사 데이터를 JSON 파일로 저장합니다."""
    try:
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 파일명 생성 (URL의 마지막 부분 사용)
        filename = article_data['metadata']['url'].split('/')[-1]
        if not filename:
            filename = f"article_{int(time.time())}"
        
        # 파일 경로
        filepath = os.path.join(output_dir, f"{filename}.json")
        
        # JSON 파일로 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(article_data, f, ensure_ascii=False, indent=2)
            
        print(f"데이터 저장 완료: {filepath}")
        return True
        
    except Exception as e:
        print(f"데이터 저장 중 오류 발생: {str(e)}")
        return False

def scrape_d2_articles():
    """D2 블로그 기사를 스크래핑합니다."""
    try:
        driver = create_webdriver()
        base_url = "https://d2.naver.com/home"
        
        # 기사 링크 수집
        article_links = get_article_links(driver, base_url)
        print(f"발견된 기사 링크 수: {len(article_links)}")
        
        # 기사 정보 저장
        processed_count = 0
        error_count = 0
        
        for i, url in enumerate(article_links):
            print(f"\n기사 처리 중 ({i+1}/{len(article_links)}): {url}")
            
            try:
                # 기사 내용 추출
                article_data = extract_article_content(driver, url)
                if not article_data:
                    print("기사 내용 추출 실패")
                    error_count += 1
                    continue
                
                # 임베딩 생성
                article_with_embedding = generate_embedding_for_article(article_data)
                if not article_with_embedding:
                    print("임베딩 생성 실패")
                    error_count += 1
                    continue
                
                # 데이터 저장
                if save_article_data(article_with_embedding):
                    processed_count += 1
                
                # API 호출 제한을 위한 딜레이
                time.sleep(1)
                
            except Exception as e:
                print(f"기사 처리 중 오류 발생: {str(e)}")
                error_count += 1
                continue
        
        print(f"\n처리 완료: 성공 {processed_count}개, 실패 {error_count}개")
        
    except Exception as e:
        print(f"스크래핑 중 오류 발생: {str(e)}")
    finally:
        if 'driver' in locals():
            driver.quit()

if __name__ == "__main__":
    scrape_d2_articles() 