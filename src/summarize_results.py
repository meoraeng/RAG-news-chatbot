import pandas as pd
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from dotenv import load_dotenv
from tabulate import tabulate
import re
from tqdm import tqdm
import numpy as np

# rag_chatbot.py에서 Solar 모델을 가져옵니다.
from rag_chatbot import chat, PROMPT_TEMPLATES, load_components

# .env 파일 로드
load_dotenv(override=True)

# --- LangSmith 클라이언트 초기화 ---
client = Client()
DATASET_NAME = "it-news-rag-chatbot"

# 각 프롬프트 전략에 대한 한글 이름 매핑
prompt_map = {
    "Basic": "기본",
    "Few-shot": "Few-shot",
    "CoT": "CoT (사고의 연쇄)",
    "Self-Correction": "자가 수정",
}

# --- 커스텀 평가 지표 함수 ---

def calculate_jaccard_similarity(str1, str2):
    """두 문자열 간의 Jaccard 유사도를 계산합니다."""
    if not str1 or not str2:
        return 0.0
    # 간단한 공백 기반 토큰화
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

# LLM-based Inclusion 평가를 위한 체인
evaluator_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader evaluating whether a generated answer is grounded in a reference answer. "
            "Respond with 'Yes' if the generated answer includes the core information from the reference answer, and 'No' otherwise."
        ),
        ("human", "Generated Answer: {generated_answer}\nReference Answer: {reference_answer}"),
    ]
)
llm_based_inclusion_chain = evaluator_prompt_template | chat | StrOutputParser()

def get_llm_based_inclusion(reference_answer: str, generated_answer: str) -> float:
    """LLM을 사용하여 생성된 답변이 참조 답변을 포함하는지 평가합니다."""
    result = llm_based_inclusion_chain.invoke(
        {"reference_answer": reference_answer, "generated_answer": generated_answer}
    )
    return 1.0 if "yes" in result.lower() else 0.0

# LangSmith에서 제공하는 기본 평가 지표 이름을 사용합니다.
# "Correctness" -> "qa", "Faithfulness" -> "criteria", "Contextual Accuracy" -> "context_qa" 등으로 이름이 변경될 수 있습니다.
# LangSmith 실행 결과를 보고 실제 키 값으로 수정해야 합니다.
metric_map = {
    "qa": "정확성 (Correctness)",
    "context_qa": "답변 관련성 (Contextual Relevance)",
    # "faithfulness": "사실 기반 여부 (Faithfulness)", # 주석 처리
    "context_recall": "문서 재현율 (Context Recall)",
    "context_precision": "문서 정확도 (Context Precision)",
    "answer_length": "평균 답변 길이 (Token)",
    "hallucination_ratio": "환각 비율 (%)",
    "keyword_inclusion_rate": "키워드 포함률 (%)"
}

# 최종 표에 표시할 평가 지표 순서
METRIC_ORDER = [
    "정확성 (Correctness)",
    "답변 관련성 (Contextual Relevance)",
    "사실 기반 여부 (Faithfulness)", # faithfulness는 현재 평가에서 제외됨
    "문서 재현율 (Context Recall)",
    "문서 정확도 (Context Precision)",
    "평균 답변 길이 (Token)",
    "키워드 포함률 (%)",
    "환각 비율 (%)"
]

def analyze_project_feedback(project, example_map):
    """특정 프로젝트의 실행 기록을 분석하고 모든 지표의 평균을 계산합니다."""
    runs = list(client.list_runs(project_id=project.id, error=False))
    
    if not runs:
        return None

    # 모든 지표를 저장할 리스트
    scores = {
        "qa": [], "context_qa": [], "faithfulness": [], "context_recall": [], "context_precision": [],
        "answer_length": [], "keyword_inclusion": [], "hallucination_ratio": []
    }

    for run in tqdm(runs, desc=f"'{project.name}' 분석 중", leave=False):
        example_id = run.reference_example_id
        if not example_id:
            continue
        
        example = example_map.get(example_id)
        if not example or not example.outputs or not run.outputs:
            continue

        ground_truth = example.outputs.get("output")
        generated_answer = run.outputs.get("answer")
        retrieved_context = run.outputs.get("context")

        if not ground_truth or not generated_answer:
            continue

        # LangSmith 기본 평가 지표 수집
        feedback = client.list_feedback(run_ids=[run.id])
        for f in feedback:
            if f.key in scores and f.score is not None:
                scores[f.key].append(f.score)

        # 커스텀 지표 계산
        scores["answer_length"].append(len(generated_answer))
        scores["keyword_inclusion"].append(calculate_jaccard_similarity(ground_truth, generated_answer))
        if retrieved_context:
            scores["hallucination_ratio"].append(1 - get_llm_based_inclusion(retrieved_context, generated_answer))

    # 평균 계산
    avg_results = {}
    for key, value_list in scores.items():
        if value_list:
            avg_results[key] = np.mean(value_list)
        else:
            avg_results[key] = 0 # 값이 없는 경우 0으로 처리

    # 백분율로 변환해야 하는 지표 처리
    if "keyword_inclusion" in avg_results:
        avg_results["keyword_inclusion_rate"] = avg_results.pop("keyword_inclusion") * 100
    if "hallucination_ratio" in avg_results:
        avg_results["hallucination_ratio"] = avg_results.pop("hallucination_ratio") * 100

    return avg_results

# --- 메인 실행 함수 ---

def main():
    """메인 함수: 모든 프로젝트를 가져와 분석하고 최종 결과를 표로 출력합니다."""
    # 모든 프로젝트와 데이터셋 예시를 미리 가져옵니다.
    projects = list(client.list_projects())
    examples = list(client.list_examples(dataset_name=DATASET_NAME))
    example_map = {example.id: example for example in examples}

    print(f"총 {len(projects)}개의 LangSmith 프로젝트와 {len(examples)}개의 데이터셋 예시를 찾았습니다.")

    results = {}
    # prompt_map의 키(프롬프트 이름)를 기준으로 최신 프로젝트를 찾습니다.
    for prompt_key, display_name in prompt_map.items():
        # 해당 프롬프트로 실행된 모든 프로젝트를 찾습니다.
        prompt_projects = [
            p for p in projects if p.metadata and p.metadata.get("prompt_name") == prompt_key
        ]
        
        if not prompt_projects:
            print(f"'{display_name}'에 대한 평가 결과를 찾을 수 없습니다. 건너뜁니다.")
            continue

        # 가장 최근에 실행된 프로젝트를 선택합니다. API 변경으로 created_at 대신 start_time을 사용합니다.
        latest_project = max(prompt_projects, key=lambda p: p.start_time)
        print(f"'{display_name}'에 대한 최신 평가 프로젝트 '{latest_project.name}'를 분석합니다.")
        
        analysis_result = analyze_project_feedback(latest_project, example_map)
        if analysis_result:
            results[display_name] = analysis_result

    if not results:
        print("분석할 유효한 결과가 없습니다. 스크립트를 종료합니다.")
        return

    # 결과를 표로 변환
    df = pd.DataFrame(results).T
    df.index.name = "프롬프트 전략"

    # NaN 값을 0으로 채우고, metric_map을 사용하여 컬럼 이름을 한글로 변경
    df = df.fillna(0).rename(columns=metric_map)
    
    # 최종 표에 표시할 컬럼만 선택하고 순서를 지정
    final_columns = [metric_map[key] for key in METRIC_ORDER if key in metric_map and metric_map[key] in df.columns]
    df = df[final_columns]

    # 소수점 둘째 자리까지 포매팅
    for col in df.columns:
        if "길이" in col:
            df[col] = df[col].map("{:.0f}".format)
        else:
            df[col] = df[col].map("{:.2f}".format)

    print("\n" + "="*80)
    print(" 프롬프트 전략별 성능 비교 결과".center(80))
    print("="*80)
    print(df.to_markdown())
    print("="*80)

if __name__ == "__main__":
    main() 