import os
from dotenv import load_dotenv
from langsmith import Client

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

def check_connection():
    """
    현재 설정된 API 키를 사용하여 LangSmith에 연결하고,
    접근 가능한 모든 데이터셋의 목록을 출력합니다.
    """
    print("LangSmith 연결 및 데이터셋 목록 조회를 시도합니다...")
    
    # .env 파일에서 API 키를 직접 읽어옵니다.
    api_key = os.getenv("LANGCHAIN_API_KEY")

    if not api_key:
        print("\n❌ 오류: .env 파일에서 LANGCHAIN_API_KEY를 찾을 수 없습니다.")
        return

    # 어떤 키를 사용하고 있는지 확인하기 위해 키의 일부를 출력합니다.
    print(f"사용 중인 API 키 (시작 부분): {api_key[:10]}...")
    
    try:
        # LangSmith 클라이언트를 초기화할 때 API 키를 명시적으로 전달합니다.
        client = Client(api_key=api_key)

        # 현재 API 키로 접근 가능한 데이터셋 목록을 가져옵니다.
        datasets = client.list_datasets()
        
        dataset_names = [d.name for d in datasets]

        if not dataset_names:
            print("성공적으로 연결되었으나, 접근 가능한 데이터셋이 없습니다.")
            print("API 키가 생성된 LangSmith 조직(Organization)과 데이터셋이 생성된 조직이 다른지 확인해주세요.")
            return

        print("\n✅ 연결 성공! 현재 API 키로 접근 가능한 데이터셋 목록:")
        for name in dataset_names:
            print(f"- {name}")
        
        # 우리가 찾고 있는 데이터셋이 있는지 확인합니다.
        if "IT News QA" in dataset_names:
            print("\n>> 'IT News QA' 데이터셋을 찾았습니다! 이제 평가 스크립트를 다시 실행하면 됩니다.")
        else:
            print("\n>> 경고: 'IT News QA' 데이터셋을 찾을 수 없습니다.")
            print("   LangSmith UI에서 현재 로그인된 조직과 API 키를 발급받은 조직이 동일한지 확인해주세요.")

    except Exception as e:
        print("\n❌ 연결 또는 조회 실패.")
        print("오류가 발생했습니다. 아래 상세 정보를 확인해주세요.")
        print(f"에러 상세 정보: {e}")

if __name__ == "__main__":
    check_connection() 