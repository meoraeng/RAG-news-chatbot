import os
import random
import chromadb
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_upstage import ChatUpstage

def generate_questions_from_db():
    """
    ChromaDB에서 무작위로 문서를 하나 선택하고,
    해당 문서의 내용을 바탕으로 LLM을 사용하여 평가용 질문을 생성합니다.
    """
    load_dotenv()

    # --- ChromaDB 연결 ---
    try:
        db_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_db"))
        client = chromadb.PersistentClient(path=db_dir)
        collection = client.get_collection(name="articles")
        print("[INFO] ChromaDB에 성공적으로 연결되었습니다.")
    except Exception as e:
        print(f"[ERROR] ChromaDB 연결에 실패했습니다: {e}")
        return

    # --- 무작위 문서 선택 ---
    # DB의 모든 데이터를 가져옵니다. (메타데이터와 문서 내용 포함)
    try:
        all_docs = collection.get(include=["metadatas", "documents"])
        if not all_docs or not all_docs['ids']:
            print("[ERROR] DB에서 문서를 가져올 수 없습니다. DB가 비어있을 수 있습니다.")
            return
        
        # 무작위로 문서 하나를 선택합니다.
        random_index = random.randint(0, len(all_docs['ids']) - 1)
        doc_id = all_docs['ids'][random_index]
        doc_content = all_docs['documents'][random_index]
        doc_metadata = all_docs['metadatas'][random_index]
        
        category = doc_metadata.get('platform', 'N/A')
        title = doc_metadata.get('title', '제목 없음')

    except Exception as e:
        print(f"[ERROR] DB에서 문서를 선택하는 중 오류가 발생했습니다: {e}")
        return

    # --- LLM을 이용한 질문 생성 ---
    print("\n[INFO] 선택된 문서를 바탕으로 AI가 질문을 생성합니다...")
    try:
        chat = ChatUpstage()
        
        prompt_template = """
당신은 IT 기술 블로그 아티클을 평가하기 위한 질문을 만드는 전문 출제위원입니다.
당신의 임무는 오직 아래에 주어진 "아티클 본문" 내용에만 근거하여, 이 글의 핵심 내용을 묻는 좋은 질문 3개를 만드는 것입니다.

**[규칙]**
1. 질문은 반드시 "아티클 본문"의 내용만으로 답변할 수 있어야 합니다.
2. 외부 지식이나 상식을 요구하는 질문을 만들어서는 안 됩니다.
3. 질문은 명확하고 간결해야 합니다.
4. 각 질문은 줄바꿈으로 구분하여, 숫자 없이 질문 내용만 나열해주세요.

---
[아티클 본문]
{context}
---

[생성된 질문]
"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | chat
        
        response = chain.invoke({"context": doc_content})
        generated_questions = response.content.strip().split('\n')

    except Exception as e:
        print(f"[ERROR] LLM을 통해 질문을 생성하는 중 오류가 발생했습니다: {e}")
        return

    # --- 결과 출력 ---
    print("\n" + "="*60)
    print(f"질문 생성을 위한 원본 아티클 (카테고리: {category})")
    print(f"제목: {title}")
    print("="*60)
    print(doc_content)
    print("\n" + "="*60)
    print("AI가 생성한 질문 목록 (이 질문들을 바탕으로 정답지를 만들어보세요)")
    print("="*60)
    for i, q in enumerate(generated_questions):
        print(f"질문 {i+1}: {q.strip()}")
    print("="*60)
    print("\n다른 문서로 새로운 질문을 생성하려면 스크립트를 다시 실행하세요.")


if __name__ == "__main__":
    generate_questions_from_db() 