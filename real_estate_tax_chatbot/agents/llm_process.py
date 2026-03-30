from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import dotenv 
from langchain_pinecone import PineconeVectorStore
from tools.my_tools import tool_list
from langgraph.prebuilt import ToolNode
from state import response_state

def get_llm():
    llm = ChatOpenAI(model='gpt-4o')
    return llm

def get_retriever():
    index_name = "real-estate-tax"
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    return retriever

def llm_with_tools():
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tool_list)

    return llm_with_tools


def get_tools():
    llm = get_llm()
    llm_with_tools = llm_with_tools()
    tool_node = ToolNode(tool_list)
    
    return tool_node

def retrieve_agent(state: response_state):
    system_prompt = """[역할] 대한민국 종합부동산세(종부세) 계산 및 법령 해석 전문가 도구입니다.
    1. 도구 사용 시, 각 도구의 역할과 기능을 명확히 이해하고 활용하세요.
    3. 도구를 사용하여 얻은 정보를 바탕으로 질문에 대한 최종 답변을 생성하세요.
    4. 도구 사용 후에는 반드시 최종 답변을 생성하여 사용자에게 제공하세요.
    """
    response = state['messages'] + [system_prompt]
    response = llm_with_tools.invoke(response)
    

    return {"messages": [response]}



def should_continue(state: response_state):
    messages = state['messages']
    last_ai_message = messages[-1]
    if last_ai_message.tool_calls:
        return 'tools'
    return 'end'


