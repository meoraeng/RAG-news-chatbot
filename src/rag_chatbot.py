import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_upstage.chat_models import ChatUpstage
from langchain_upstage.embeddings import UpstageEmbeddings
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv(override=True)

# --- 상수 정의 ---
PROMPT_TEMPLATES = {
    "Basic": """
        당신은 사용자의 질문에 대해 주어진 문서를 참고하여 답변하는 AI 챗봇입니다. 
        문서의 내용을 벗어난 질문에는 답변하지 마세요.
        답변은 항상 한국어로 작성해주세요.
        
        [검색된 문서]:
        {context}
        
        [질문]: 
        {input}
    """,
    "Few-shot": """
        당신은 사용자의 질문에 대해 주어진 문서를 참고하여 답변하는 AI 챗봇입니다. 
        아래 예시와 같이, 주어진 문서의 내용을 바탕으로 상세하고 친절하게 답변해주세요.
        문서의 내용을 벗어난 질문에는 답변하지 마세요.
        답변은 항상 한국어로 작성해주세요.

        [예시 1]
        질문: RAG가 무엇인가요?
        답변: RAG는 Retrieval-Augmented Generation의 약자로, 대규모 언어 모델(LLM)이 외부 지식 베이스의 정보를 활용하여 더 정확하고 신뢰성 있는 답변을 생성하도록 하는 기술입니다. 검색(Retrieval)을 통해 관련 문서를 찾고, 이를 바탕으로 답변을 생성(Generation)하는 두 단계로 구성됩니다.

        [예시 2]
        질문: LangChain의 장점은?
        답변: LangChain의 주요 장점은 LLM 애플리케이션 개발을 모듈화하고 간소화할 수 있다는 점입니다. 다양한 LLM 모델, 외부 데이터 소스, API 등을 '체인'이라는 형태로 쉽게 결합할 수 있어 복잡한 워크플로우를 효율적으로 구축할 수 있으며, 개발 과정을 표준화하여 생산성을 높여줍니다.
        
        [검색된 문서]:
        {context}
        
        [질문]: 
        {input}
    """,
    "Self-Correction": """
        **초안 생성 프롬프트 (내부용)**
        You are an AI assistant. Use the retrieved documents to provide a draft answer to the user's question.
        
        [검색된 문서]:
        {context}
        
        [질문]: 
        {input}
        
        **수정 프롬프트 (최종 답변용)**
        You are an expert AI assistant. Your task is to refine a draft answer based on the provided documents.
        Review the original question, the retrieved documents, and the draft answer.
        Rewrite the draft answer to be more accurate, detailed, and faithful to the documents.
        The final answer must be in Korean.

        [원본 질문]: {input}
        [검색된 문서]: {context}
        [답변 초안]: {draft_answer}

        [수정된 최종 답변]:
    """,
    "CoT": """
        당신은 사용자의 질문에 대해 주어진 문서를 참고하여 답변하는 AI 챗봇입니다.
        먼저 질문과 관련된 주요 내용을 단계별로 생각하고, 그 근거를 바탕으로 최종 답변을 작성하세요.
        문서의 내용을 벗어난 질문에는 답변하지 마세요.
        답변은 항상 한국어로 작성해주세요.

        [검색된 문서]:
        {context}
        
        [질문]: 
        {input}
        
        [단계별 생각]:
        1. 질문의 핵심 키워드와 의도를 파악합니다.
        2. 검색된 문서에서 질문과 관련된 문장이나 단락을 찾습니다.
        3. 찾아낸 정보를 종합하여 질문에 대한 답변을 구성합니다.
        
        [최종 답변]:
    """
}
DEFAULT_PROMPT_KEY = "Basic"


# --- 모델 및 리트리버 초기화 (캐싱 활용) ---
@st.cache_resource
def load_components(prompt_key=DEFAULT_PROMPT_KEY):
    """
    LangChain의 주요 컴포넌트(LLM, 임베딩, 벡터스토어, 리트리버)를 로드합니다.
    Streamlit의 캐싱을 활용하여 리소스를 효율적으로 관리합니다.
    """
    print("[INFO] LangChain 컴포넌트를 로드합니다...")
    
    # 1. ChatUpstage LLM 로드
    chat = ChatUpstage()
    
        # 2. Upstage 임베딩 모델 로드
    embeddings = UpstageEmbeddings(model="embedding-passage")

    # 3. ChromaDB 벡터스토어 로드
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    # 4. 리트리버 설정
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'score_threshold': 0.4, 'k': 5}
    )
    
    print("[INFO] 컴포넌트 로드 완료.")
    return chat, embeddings, vectorstore, retriever

# --- 전역 컴포넌트 (런타임 초기화) ---
chat = None
embeddings = None
vectorstore = None
retriever = None


# --- RAG 체인 생성 함수 ---
def create_rag_chain(history_aware_retriever, prompt_key=DEFAULT_PROMPT_KEY):
    """
    선택된 프롬프트 템플릿을 기반으로 RAG 체인을 생성합니다.
    'Self-Correction'의 경우 특별한 2단계 체인을 생성합니다.
    """
    if prompt_key == "Self-Correction":
        # 1. 초안 생성을 위한 프롬프트와 체인
        draft_prompt_template = """
            You are an AI assistant. Use the retrieved documents to provide a draft answer to the user's question.
            The answer must be in Korean.
            
            [검색된 문서]:
            {context}
            
            [질문]: 
            {input}
        """
        draft_prompt = ChatPromptTemplate.from_template(draft_prompt_template)
        draft_chain = create_stuff_documents_chain(chat, draft_prompt)

        # 2. 수정을 위한 프롬프트
        refine_prompt_template = """
            You are an expert AI assistant. Your task is to refine a draft answer based on the provided documents.
            Review the original question, the retrieved documents, and the draft answer.
            Rewrite the draft answer to be more accurate, detailed, and faithful to the documents.
            The final answer must be in Korean.

            [원본 질문]: {question}
            [검색된 문서]: {context}
            [답변 초안]: {draft_answer}

            [수정된 최종 답변]:
        """
        refine_prompt = ChatPromptTemplate.from_template(refine_prompt_template)
        
        # 3. 2단계 체인 결합
        # Retrieval 체인 -> 초안 생성 -> 수정 체인
        # 'context'와 'input'을 끝까지 전달하는 것이 중요합니다.
        refine_chain = (
            {
                "draft_answer": create_retrieval_chain(history_aware_retriever, draft_chain) | itemgetter("answer"),
                "input": itemgetter("input"),
                "context": itemgetter("context"),
            }
            | RunnablePassthrough.assign(question=itemgetter("input")) # 'question' 키 추가
            | refine_prompt
            | chat
            | StrOutputParser()
        )
        # 최종 출력 형식을 {"answer": "...", "context": ...} 로 맞추기 위한 래퍼
        # 평가 스크립트 호환성을 위해 context를 전달해야 합니다.
        final_chain = (
             RunnablePassthrough.assign(context=history_aware_retriever)
             | RunnablePassthrough.assign(answer=refine_chain)
             | (lambda x: {"answer": x["answer"], "context": x["context"]})
        )
        return final_chain

    # 기존 프롬프트 처리
    selected_prompt_template = PROMPT_TEMPLATES.get(prompt_key, PROMPT_TEMPLATES[DEFAULT_PROMPT_KEY])
    
    # 1. QA 체인 프롬프트 정의
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", selected_prompt_template),
        ("human", "{input}"),
    ])

    # 2. 문서와 질문을 결합하는 Stuff 체인 생성
    question_answer_chain = create_stuff_documents_chain(chat, qa_prompt)
    
    # 3. 리트리버와 QA 체인을 결합하는 최종 RAG 체인 생성
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


# --- 대화 기록을 고려한 리트리버 생성 함수 ---
def get_history_aware_retriever(chat_history):
    """
    대화 기록을 기반으로 사용자의 현재 질문 의도를 파악하고 관련 문서를 검색하는 리트리버를 생성합니다.
    """
    # 1. 대화 기록 → 검색어 변환 프롬프트
    contextualize_q_system_prompt = (
        "주어진 대화 기록과 최근 사용자 질문을 바탕으로, "
        "대화의 맥락을 이해할 수 있는 독립적인 질문으로 재구성하세요. "
        "재구성된 질문은 검색 엔진에서 사용될 것입니다."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])

    # 2. 검색어 변환 체인 생성
    history_aware_retriever_chain = create_history_aware_retriever(
    chat, retriever, contextualize_q_prompt
)

    # 3. 대화 기록을 체인에 바인딩
    return RunnablePassthrough.assign(
        chat_history=lambda x: chat_history
    ) | history_aware_retriever_chain


# --- 메인 Streamlit 앱 로직 ---
def main():
    """
    Streamlit 기반의 RAG 챗봇 UI를 구성하고 실행합니다.
    """
    st.set_page_config(page_title="RAG Chatbot Demo", page_icon="🤖")

    global chat, embeddings, vectorstore, retriever
    if chat is None or embeddings is None or vectorstore is None or retriever is None:
        chat, embeddings, vectorstore, retriever = load_components()

    st.title("🤖 RAG Chatbot Demo")
    st.caption("IT 기술 블로그 및 뉴스 데이터를 기반으로 답변하는 챗봇입니다.")

    # --- UI 컨트롤러 ---
    col1, col2 = st.columns(2)
    with col1:
        rag_enabled = st.toggle("RAG (검색 증강 생성) 활성화", value=True)
    with col2:
        prompt_key = st.selectbox(
            "프롬프팅 기법 선택:",
            options=list(PROMPT_TEMPLATES.keys()),
            index=0,
            disabled=not rag_enabled
        )

    # --- 세션 상태 및 채팅 기록 관리 ---
    msgs = StreamlitChatMessageHistory(key="chat_history")
    if len(msgs.messages) == 0:
        msgs.add_ai_message("안녕하세요! 저는 IT 기술 블로그 데이터를 기반으로 답변하는 챗봇입니다. 무엇을 도와드릴까요?")

    # 채팅 메시지 표시
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    # 사용자 입력 처리
    if user_input := st.chat_input(placeholder="질문을 입력하세요..."):
        st.chat_message("human").write(user_input)
        msgs.add_user_message(user_input)

        with st.chat_message("ai"):
            # RAG 활성화 시
            if rag_enabled:
                history_aware_retriever = get_history_aware_retriever(msgs.messages)
                rag_chain = create_rag_chain(history_aware_retriever, prompt_key)
                
                response_generator = rag_chain.stream({"input": user_input})
                
                full_response = ""
                response_container = st.empty()
                
                for chunk in response_generator:
                    if "answer" in chunk:
                        full_response += chunk["answer"]
                        response_container.markdown(full_response + "▌")
                response_container.markdown(full_response)
                
                msgs.add_ai_message(full_response)

            # RAG 비활성화 시 (LLM 단독 응답)
            else:
                llm_chain = (
                    {"input": RunnablePassthrough()}
                    | chat
                    | StrOutputParser()
                )
                response = llm_chain.invoke(user_input)
                st.write(response)
                msgs.add_ai_message(response)
                
    # --- 시스템 정보 출력 (최초 실행 시에만) ---
    if 'initial_run' not in st.session_state:
        print(f"\n[INFO] ChromaDB 컬렉션 '{vectorstore._collection.name}'에 저장된 총 문서 조각(Chunk) 수: {vectorstore._collection.count()}개")
        st.session_state['initial_run'] = True


if __name__ == "__main__":
    main()