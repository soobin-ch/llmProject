#llm 답변을 위한 tools 정의
from agents.llm_process import get_retriever
import asyncio

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_community import GmailToolkit
from langchain_google_community.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)
from langchain_core.tools import tool
from state import response_state
from langgraph.prebuilt import ToolNode

@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


credentials = get_gmail_credentials(
    token_file="token.json",
    scopes=["https://mail.google.com/"],
    client_sercret_file="credentials.json",
)
api_resource = build_resource_service(credentials=credentials)
gmail_toolkit = GmailToolkit(api_resource=api_resource)
gmail_tool_list = gmail_toolkit.get_tools()
toolkit = GmailToolkit()
search_tool = DuckDuckGoSearchRun()

@tool
async def retrieve_parallel_context(state: response_state):
    """
    [사용 지침]
    1. 정보가 부족하면 최대 2~3개의 핵심 쿼리로 분할하여 병렬 검색을 수행하세요.
    2. 검색 결과에서 숫자가 포함된 세부 조항이나 법적 근거는 누락 없이 모두 나열해야 합니다.
    """
    query = state["messages"][0].content  # 첫 번째 메시지를 질문으로 간주
    tasks=[ get_retriever]


    results = await asyncio.gather(*tasks)

  
    flat_docs = []
    for doc_list in results:
        flat_docs.extend(doc_list)

    return {'messages' : flat_docs}


tool_list = [add, multiply, search_tool, retrieve_parallel_context ] + gmail_tool_list
