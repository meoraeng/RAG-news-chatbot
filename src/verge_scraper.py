# 봇 차단 문제로 인해 임시 주석 처리

# import requests
# from bs4 import BeautifulSoup
# import json
# import time
# from openai import OpenAI
# import os
# from dotenv import load_dotenv
# from playwright.sync_api import sync_playwright

# # .env 파일에서 환경 변수 로드
# load_dotenv()

# # OpenAI API 키 설정
# client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# def split_text(text, max_length=4000):
#     """텍스트를 최대 길이로 분할"""
#     if len(text) <= max_length:
#         return [text]
    
#     chunks = []
#     current_chunk = ""
    
#     for sentence in text.split('.'):
#         if len(current_chunk) + len(sentence) + 1 <= max_length:
#             current_chunk += sentence + '.'
#         else:
#             chunks.append(current_chunk)
#             current_chunk = sentence + '.'
    
#     if current_chunk:
#         chunks.append(current_chunk)
    
#     return chunks

# def get_embedding(text):
#     """텍스트를 임베딩으로 변환"""
#     try:
#         response = client.embeddings.create(
#             model="text-embedding-3-small",
#             input=text
#         )
#         return response.data[0].embedding
#     except Exception as e:
#         print(f"임베딩 생성 중 오류 발생: {str(e)}")
#         return None

# def extract_article_content(page, article_url):
#     """기사 내용 추출"""
#     try:
#         page.goto(article_url, wait_until='load', timeout=60000)
#         page.wait_for_selector('article', timeout=30000)
        
#         # JavaScript 실행이 완료될 때까지 대기
#         time.sleep(2)
        
#         content = page.query_selector('article')
#         if content:
#             html_content = content.inner_html()
#             soup = BeautifulSoup(html_content, 'html.parser')
            
#             # 불필요한 요소 제거
#             for element in soup.select('script, style, iframe, noscript, .related-posts, .social-share, .comments-section'):
#                 element.decompose()
            
#             return soup.get_text(strip=True)
#         return None
#     except Exception as e:
#         print(f"기사 내용 추출 중 오류 발생: {str(e)}")
#         return None

# def scrape_verge_articles(max_articles=30):
#     """The Verge 기사 스크래핑"""
#     articles = []
#     base_url = "https://www.theverge.com/"
    
#     with sync_playwright() as p:
#         try:
#             browser = p.chromium.launch(
#                 headless=True,
#                 args=[
#                     '--disable-gpu',
#                     '--disable-dev-shm-usage',
#                     '--disable-setuid-sandbox',
#                     '--no-sandbox'
#                 ]
#             )
#             context = browser.new_context(
#                 user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
#                 viewport={'width': 1920, 'height': 1080}
#             )
#             page = context.new_page()
            
#             # 타임아웃 설정
#             page.set_default_timeout(60000)
#             page.set_default_navigation_timeout(60000)
            
#             print("브라우저 초기화 완료")
            
#             page.goto(base_url, wait_until='load', timeout=60000)
#             print("The Verge 웹사이트 로드 중...")
            
#             # 기사 목록이 로드될 때까지 대기
#             page.wait_for_selector('article', timeout=30000)
#             print("기사 목록 로드 완료")
            
#             # JavaScript 실행이 완료될 때까지 추가 대기
#             time.sleep(2)
            
#             # 페이지 소스 가져오기
#             html_content = page.content()
#             soup = BeautifulSoup(html_content, 'html.parser')
#             article_elements = soup.select('article')
            
#             print(f"발견된 기사 수: {len(article_elements)}")
            
#             for idx, article in enumerate(article_elements[:max_articles], 1):
#                 try:
#                     title_element = article.select_one('h2 a')
#                     date_element = article.select_one('time')
                    
#                     if not title_element or not date_element:
#                         continue
                    
#                     title = title_element.get_text(strip=True)
#                     date = date_element.get('datetime', '')
#                     article_url = title_element.get('href', '')
                    
#                     if not article_url:
#                         continue
                    
#                     if not article_url.startswith('http'):
#                         article_url = 'https://www.theverge.com' + article_url
                    
#                     print(f"기사 {idx}/{min(max_articles, len(article_elements))} 처리 중: {title}")
                    
#                     # 기사 내용 추출
#                     content = extract_article_content(page, article_url)
#                     if not content:
#                         print(f"기사 내용을 추출할 수 없습니다: {title}")
#                         continue
                    
#                     print(f"기사 내용 길이: {len(content)} 문자")
                    
#                     # 임베딩 생성
#                     embedding = get_embedding(title + " " + content)
#                     if not embedding:
#                         print(f"임베딩을 생성할 수 없습니다: {title}")
#                         continue
                    
#                     articles.append({
#                         'title': title,
#                         'date': date,
#                         'content': content,
#                         'embedding': embedding,
#                         'url': article_url
#                     })
                    
#                     print(f"기사 처리 완료: {title}")
                    
#                     # 서버 부하 방지를 위한 대기
#                     time.sleep(2)
                    
#                 except Exception as e:
#                     print(f"기사 처리 중 오류 발생: {str(e)}")
#                     continue
            
#             return articles
        
#         except Exception as e:
#             print(f"크롤링 중 오류 발생: {str(e)}")
#             return []
        
#         finally:
#             if 'browser' in locals():
#                 browser.close()

# def save_results(articles_data):
#     """결과를 JSON 파일로 저장"""
#     with open('verge_articles_with_embeddings.json', 'w', encoding='utf-8') as f:
#         json.dump({'articles': articles_data}, f, ensure_ascii=False, indent=2)

# if __name__ == "__main__":
#     print("The Verge 크롤링 및 임베딩 시작...")
#     articles = scrape_verge_articles()
#     print(f"총 {len(articles)}개의 기사가 처리되었습니다.")
#     save_results(articles)
#     print("결과가 verge_articles_with_embeddings.json 파일에 저장되었습니다.") 

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
    driver.set_page_load_timeout(180)
    driver.implicitly_wait(20)
    return driver

def get_article_links(driver, url):
    """기사 링크를 수집합니다."""
    try:
        driver.get(url)
        time.sleep(5)
        
        links = []
        # 기사 목록 컨테이너에서 모든 기사 링크 수집
        articles = driver.find_elements(By.CSS_SELECTOR, 'div.article_list > div.article_item')
        
        for article in articles:
            try:
                href = article.find_element(By.CSS_SELECTOR, 'a').get_attribute('href')
                if href:
                    # 제목
                    title = article.find_element(By.CSS_SELECTOR, 'h3').text.strip()
                    
                    # 작성자
                    author = article.find_element(By.CSS_SELECTOR, 'span.author').text.strip()
                    
                    # 날짜
                    date = article.find_element(By.CSS_SELECTOR, 'span.date').text.strip()
                    
                    # 카테고리
                    categories = [cat.text.strip() for cat in article.find_elements(By.CSS_SELECTOR, 'span.category')]
                    
                    links.append({
                        'url': href,
                        'title': title,
                        'author': author,
                        'date': date,
                        'category': categories
                    })
            except Exception as e:
                print(f"기사 정보 추출 중 오류 발생: {str(e)}")
                continue
        
        return links
    except Exception as e:
        print(f"링크 수집 중 오류 발생: {str(e)}")
        return []

def extract_article_content(driver, url):
    """기사 내용을 추출합니다."""
    try:
        driver.get(url)
        time.sleep(5)
        
        # 제목 추출
        title = driver.find_element(By.CSS_SELECTOR, "h1.article_title").text.strip()
        
        # 내용 추출
        content_element = driver.find_element(By.CSS_SELECTOR, "div.article_content")
        content = content_element.text.strip()
        
        # 날짜 추출
        date = driver.find_element(By.CSS_SELECTOR, "span.date").text.strip()
        
        # 작성자 추출
        author = driver.find_element(By.CSS_SELECTOR, "span.author").text.strip()
        
        # 태그 추출
        tags = [tag.text.strip() for tag in driver.find_elements(By.CSS_SELECTOR, "div.tags > a")]
        
        # 텍스트 길이 제한 (10,000자)
        if len(content) > 10000:
            content = content[:10000]
            
        return {
            "title": title,
            "content": content,
            "date": date,
            "author": author,
            "tags": tags,
            "url": url
        }
    except Exception as e:
        print(f"기사 정보 추출 중 오류 발생: {str(e)}")
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
                "source": "verge",
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

def scrape_verge_articles():
    """The Verge 기사를 스크래핑합니다."""
    try:
        driver = create_webdriver()
        base_url = "https://www.theverge.com/"
        
        # 기사 링크 수집
        article_links = get_article_links(driver, base_url)
        print(f"발견된 기사 링크 수: {len(article_links)}")
        
        # 기사 정보 저장
        processed_count = 0
        error_count = 0
        
        for i, article_info in enumerate(article_links):
            print(f"\n기사 처리 중 ({i+1}/{len(article_links)}): {article_info['url']}")
            
            try:
                # 기사 내용 추출
                article_data = extract_article_content(driver, article_info['url'])
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
    scrape_verge_articles() 