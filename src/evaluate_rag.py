import pandas as pd
import uuid
from operator import itemgetter

from langsmith import Client
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.evaluation import load_evaluator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# rag_chatbot.py에서 필요한 컴포넌트들을 가져옵니다.
from rag_chatbot import (
    chat,
    retriever, # history_aware_retriever 대신 평가용으로 기본 retriever를 사용합니다.
    PROMPT_TEMPLATES
)
from tabulate import tabulate

# .env 파일 로드
from dotenv import load_dotenv
load_dotenv(override=True)

# --- 설정 ---
DATASET_NAME = "it-news-rag-chatbot"

# --- LangSmith 클라이언트 초기화 ---
client = Client()

def create_rag_chain_for_evaluation(prompt_key: str):
    """평가를 위한 RAG 체인을 생성합니다. 'Self-Correction'은 특별 처리합니다."""
    
    # Self-Correction 체인
    if prompt_key == "Self-Correction":
        # 1. 초안 생성을 위한 프롬프트와 체인
        draft_prompt_template = """
            You are an AI assistant. Use the retrieved documents to provide a draft answer to the user's question.
            [검색된 문서]: {context}
            [질문]: {input}
        """
        draft_prompt = ChatPromptTemplate.from_template(draft_prompt_template)
        
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
        refine_chain = refine_prompt | chat | StrOutputParser()

        # 3. 전체 체인 결합
        # LCEL의 itemgetter와 RunnablePassthrough를 사용하여 데이터 흐름을 명확하게 정의합니다.
        # 최종적으로 evaluator가 요구하는 {"answer": ..., "context": ...} 형태를 반환해야 합니다.
        
        # 단계 1: 문서 검색 및 초안 생성
        # 'input'을 받아 문서를 검색하고, 검색된 문서와 'input'으로 초안 답변 생성
        draft_chain = (
            RunnablePassthrough.assign(
                context=itemgetter("input") | retriever,
            )
            | RunnablePassthrough.assign(
                draft_answer=create_stuff_documents_chain(chat, draft_prompt)
            )
        )

        # 단계 2: 초안을 기반으로 최종 답변 정제
        # 이전 체인에서 'context', 'input', 'draft_answer'를 받아 최종 답변 생성
        final_chain = (
            draft_chain
            | RunnablePassthrough.assign(
                final_answer=lambda x: {
                    "question": x["input"],
                    "context": x["context"],
                    "draft_answer": x["draft_answer"]["answer"],
                }
                | refine_chain
            )
            | (lambda x: {"answer": x["final_answer"], "context": x["context"]})
        )
        return final_chain

    # 기본, CoT, Few-shot 등 다른 프롬프트 처리
    prompt_template = PROMPT_TEMPLATES[prompt_key]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(chat, prompt)
    final_rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return final_rag_chain

def run_evaluation(prompt_name: str):
    """단일 프롬프트에 대해 LangSmith Experiment를 실행하고 결과를 반환합니다."""
    print(f"\n===== '{prompt_name}' 프롬프트로 Experiment 실행 시작 =====")
    
    # 1. 현재 프롬프트에 맞는 체인 생성
    chain_for_eval = create_rag_chain_for_evaluation(prompt_name)

    # 2. 평가자 준비
    # LangSmith Evaluator 이름이 변경되었습니다. (correctness -> qa)
    evaluator_names = [
        "qa", 
        "context_qa", 
    ]
    evaluators = [LangChainStringEvaluator(eval_name) for eval_name in evaluator_names]

    # 3. 평가 실행
    # LangChain v0.2.0+ 에서는 evaluate 함수의 인자가 변경되었습니다.
    # RunEvalConfig 대신, 필요한 평가자, 데이터셋 이름 등을 직접 전달합니다.
    experiment_results = evaluate(
        chain_for_eval,
        data=DATASET_NAME,
        evaluators=evaluators,
        experiment_prefix=f"Prompt '{prompt_name}' Evaluation",
        metadata={
            "prompt_name": prompt_name,
            "llm": "Solar-pro",
            "retriever": "ChromaDB with Upstage Embeddings",
        },
    )
    
    print(f"'{prompt_name}' 프롬프트 평가 완료.")
    # LangSmith v0.1.0+ 에서는 experiment_results에 URL이 직접 포함되지 않을 수 있습니다.
    # 대신 생성된 프로젝트 이름을 출력하여 사용자가 직접 찾아갈 수 있도록 안내합니다.
    if "project_name" in experiment_results:
        print(f"결과 확인: LangSmith 프로젝트 '{experiment_results['project_name']}'")


def main():
    """모든 프롬프트에 대해 각각 Experiment를 실행합니다."""
    print(f"'{DATASET_NAME}' 데이터셋에 대한 전체 평가를 시작합니다...")
    
    successful_runs = []
    failed_runs = []

    # ReAct는 평가에서 제외
    prompts_to_evaluate = {k: v for k, v in PROMPT_TEMPLATES.items() if k != "ReAct"}

    for prompt_name, prompt_template in prompts_to_evaluate.items():
        try:
            run_evaluation(prompt_name)
            successful_runs.append(prompt_name)
    except Exception as e:
            print(f"'{prompt_name}' 프롬프트 평가 중 오류 발생: {e}")
            failed_runs.append(prompt_name)

    print("\n" + "="*80)
    print("                      [ 평가 실행 요약 ]")
    print("="*80)
    print("모든 프롬프트에 대한 평가 실행이 LangSmith에서 시작되었습니다.")
    print("결과는 각 Experiment의 URL 또는 LangSmith 프로젝트 페이지에서 확인해주세요.")
    print(f"\n- 성공적으로 시작된 평가: {len(successful_runs)}개 ({', '.join(successful_runs)})")
    if failed_runs:
        print(f"- 실패한 평가: {len(failed_runs)}개 ({', '.join(failed_runs)})")
    print("="*80)


if __name__ == "__main__":
    main() 