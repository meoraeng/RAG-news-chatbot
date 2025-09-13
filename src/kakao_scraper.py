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
        articles = driver.find_elements(By.CSS_SELECTOR, 'ul.list_post > li > a.link_post')
        
        for article in articles:
            try:
                href = article.get_attribute('href')
                if href and '/posts/' in href:
                    # 제목
                    title = article.find_element(By.CSS_SELECTOR, 'h4.tit_post').text.strip()
                    
                    # 작성자
                    author = article.find_element(By.CSS_SELECTOR, 'dd.txt_authors').text.strip()
                    
                    # 날짜
                    date = article.find_element(By.CSS_SELECTOR, 'dd.txt_date').text.strip()
                    
                    # 카테고리
                    categories = [cat.text.strip() for cat in article.find_elements(By.CSS_SELECTOR, 'span.txt_cate')]
                    
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

def extract_article_content(url):
    try:
        driver = create_webdriver()
        driver.get(url)
        time.sleep(5)
        
        # 제목 추출
        title = driver.find_element(By.CSS_SELECTOR, "h1.tit_post").text.strip()
        
        # 내용 추출 (daum-wm-content 클래스 내의 모든 텍스트)
        content_element = driver.find_element(By.CSS_SELECTOR, "div.daum-wm-content.preview")
        content = content_element.text.strip()
        
        # 날짜 추출 (여러 선택자 시도)
        date = None
        date_selectors = [
            "div.info_post > span:nth-child(2)",
            "div.info_post > span.daum-wm-datetime",
            "div.info_post > span:not(.screen_out)"
        ]
        
        for selector in date_selectors:
            try:
                date_element = driver.find_element(By.CSS_SELECTOR, selector)
                date = date_element.text.strip() or date_element.get_attribute("aria-hidden")
                if date:
                    break
            except:
                continue
        
        if not date:
            print("날짜를 찾을 수 없습니다.")
            return None
        
        # 작성자 추출
        author = driver.find_element(By.CSS_SELECTOR, "div.info_post > a").text.strip()
        
        # 태그 추출
        tags = [tag.text.strip() for tag in driver.find_elements(By.CSS_SELECTOR, "ul.list_tag > li > a.link_tag")]
        
        driver.quit()
        
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
        if 'driver' in locals():
            driver.quit()
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
                "source": "kakao_tech",
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

def scrape_kakao_articles():
    """카카오 테크 블로그 기사를 스크래핑합니다."""
    try:
        driver = create_webdriver()
        base_url = "https://tech.kakao.com/blog/"
        
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
                article_data = extract_article_content(article_info['url'])
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
    scrape_kakao_articles() 