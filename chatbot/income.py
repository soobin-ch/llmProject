# %%

from langchain_openai import OpenAIEmbeddings
import dotenv 
from langchain_pinecone import PineconeVectorStore


dotenv.load_dotenv()
index_name = "tax-index"
embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
vectorstor.from_documents(~~~)
retriever = vectorstore.as_retriever(-taxearch_kwargs={"k": 4})

# %%
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-4o')

# %%
from typing import Annotated
from typing_extensions import List, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from langchain_core.documents import Document
import operator

class AgentState(TypedDict) :
    query:str
    context: List
    answer:str

# %%
from langgraph.graph import StateGraph


graph_builder = StateGraph(AgentState)

# %%
def retrieve(state: AgentState) -> AgentState:
 
    query = state['query']
    docs = retriever.invoke(query)
    return {'context': docs}

# %%
# Create a LangSmith API in Settings > API Keys
# Make sure API key env var is set:
# import os; os.environ["LANGSMITH_API_KEY"] = "<your-api-key>"
from langsmith import Client
from typing import Literal

client = Client()
doc_relevance_prompt = client.pull_prompt("langchain-ai/rag-document-relevance")

def check_doc_relevance(state: AgentState) -> Literal['relevant', 'irrelevant']:
   
    query = state['query']
    context = state['context']
    context_text = "\n\n".join([doc.page_content for doc in context])
    doc_relevance_chain = doc_relevance_prompt | llm
    response = doc_relevance_chain.invoke({'question': query, 'documents': context_text})

    print(f'response: {response["Score"]}')
    if response['Score'] == 1:
        # 2.3장과 다르게 `relevant`와 `irrelevant`를 반환합니다
        # node를 직접 지정하는 것보다 실제 판단 결과를 리턴하면서 해당 node의 재사용성을 높일 수 있습니다.
        return 'relevant'
    
    return 'irrelevant'

# %%
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


template = """당신은 질문-답변 과제를 수행하는 인공지능 어시스턴트입니다. 
주어진 문맥(Context)을 사용하여 질문(Question)에 답하세요. 
답을 모른다면 모른다고 답변하세요. 
최대한 세 문장 내로 간결하게 답변하세요.

Question: {question} 
Context: {context} 
Answer:"""

# 직접 프롬프트 객체 생성
generate_prompt = ChatPromptTemplate.from_template(template)
generate_llm = ChatOpenAI(model="gpt-4o")

def generate(state: AgentState) -> AgentState:
    
    context = state['context']
    query = state['query']
    
    
    rag_chain = generate_prompt | generate_llm

    response = rag_chain.invoke({'question': query, 'context': context})
    

    return {'answer': response.content}

# %%
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

dictionary = ['사람과 관련된 표현 -> 거주자']

rewrite_prompt = PromptTemplate.from_template(f"""
사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요 
사전: {dictionary}                                           
질문: {{query}}
""")

def rewrite(state: AgentState) -> AgentState:
    query = state['query']
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    response = rewrite_chain.invoke({'query': query})
    return {'query': response}


# %%
# set the LANGCHAIN_API_KEY environment variable (create key in settings)
from langchain_core.output_parsers import StrOutputParser

hallucination_prompt = PromptTemplate.from_template("""
You are a teacher tasked with evaluating whether a student's answer is based on documents or not,
Given documents, which are excerpts from income tax law, and a student's answer;
If the student's answer is based on documents, respond with "not hallucinated",
If the student's answer is not based on documents, respond with "hallucinated".

documents: {documents}
student_answer: {student_answer}
""")

hallucination_llm = ChatOpenAI(model='gpt-4o', temperature=0)

def check_hallucination(state: AgentState) -> Literal['hallucinated', 'not hallucinated']:
    answer = state['answer']
    context = state['context']
    context = [doc.page_content for doc in context]
    hallucination_chain = hallucination_prompt | hallucination_llm | StrOutputParser()
    response = hallucination_chain.invoke({'student_answer': answer, 'documents': context})

    return response

# %%
from langsmith import Client
from typing import Literal


# LangChain 허브에서 유용성 프롬프트를 가져옵니다
client = Client()
helpfulness_prompt = client.pull_prompt("langchain-ai/rag-answer-helpfulness")

def check_helpfulness_grader(state: AgentState) -> str:
    
    # state에서 질문과 답변을 추출합니다
    query = state['query']
    answer = state['answer']

    # 답변의 유용성을 평가하기 위한 체인을 생성합니다
    helpfulness_chain = helpfulness_prompt | llm
    
    # 질문과 답변으로 체인을 호출합니다
    response = helpfulness_chain.invoke({'question': query, 'student_answer': answer})

    # 점수가 1이면 'helpful'을 반환하고, 그렇지 않으면 'unhelpful'을 반환합니다
    if response['Score'] == 1:
        return 'helpful'
    
    return 'unhelpful'


def check_helpfulness(state: AgentState) -> AgentState:
    """
    유용성을 확인하는 자리 표시자 함수입니다. 
    graph에서 conditional_edge를 연속으로 사용하지 않고 node를 추가해
    가독성을 높이기 위해 사용합니다

    Args:
        state (AgentState): 에이전트의 현재 state.

    Returns:
        AgentState: 변경되지 않은 state를 반환합니다.
    """
    # 이 함수는 현재 아무 작업도 수행하지 않으며 state를 그대로 반환합니다
    return state

# %%
graph_builder.add_node('retrieve', retrieve)
graph_builder.add_node('generate',generate)
graph_builder.add_node('rewrite', rewrite)
graph_builder.add_node('check_helpfulness', check_helpfulness)


# %%
from langgraph.graph import START, END

graph_builder.add_edge(START, 'retrieve')
graph_builder.add_conditional_edges('retrieve', check_doc_relevance, 
{
    'relevant': 'generate',
    'irrelevant': END
}
 )
graph_builder.add_conditional_edges('generate',check_hallucination, {
    'not hallucinated': 'check_helpfulness',
    'hallucinated': 'generate'
})
graph_builder.add_conditional_edges('check_helpfulness', check_helpfulness_grader, {
    'helpful' : END,
    'unhelpful': 'rewrite'
})
graph_builder.add_edge('rewrite', 'retrieve')

# %%
graph = graph_builder.compile()

# %%
initial_state = {'query': '연봉 5천만원인 거주자가 납부할 소득세는 얼만가요?'}
graph.invoke(initial_state)


