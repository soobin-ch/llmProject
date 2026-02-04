from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


store={}

def get_session_history(session_id:str) ->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    return store[session_id]



def get_retrieval():
    model = OpenAIEmbeddings(model="text-embedding-3-large")

    index_name = "bookcity-real-estate"

    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=model)

    llm = get_llm()

    retriever=database.as_retriever()
    return retriever

def get_history_retrieval():
     retrieval = get_retrieval()
     llm = get_llm()
     contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

     contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
     history_aware_retriever = create_history_aware_retriever(
        llm, retrieval, contextualize_q_prompt
    )
     return history_aware_retriever



def get_llm(model="gpt-4o"):
    llm = ChatOpenAI(model=model)
    return llm





def get_rag_chain():
    llm = get_llm()
    retriever = get_retrieval()


  
   
    history_aware_retriever =get_history_retrieval()
    


    system_prompt = (
            "당신은 파주출판단지 부동산 매물 전문가입니다. 이 지역 부동산 매물에 관한 질문에 답변해주세요"
            "아래에 제공된 문서를 활용해서 답변해주시고"
            "답변을 알 수 없다면 모른다고 답변해주세요"
            "답변을 제공할 때는 파주출판단지 매물에 따르면 이라고 시작하면서 답변해주시고"
            "2-3 문장정도의 짧은 내용의 답변을 원합니다"
            "\n\n"
            "{context}"
        )
        
    qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
    
    question_answer_chain =create_stuff_documents_chain(llm, qa_prompt)
    
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    ).pick('answer')

   
    return conversational_rag_chain


def get_ai_response(user_question):
  
  
    rag_chain = get_rag_chain()
    
    

    
    ai_response = rag_chain.stream(
        {"input": user_question},
        config={
        "configurable": {"session_id":"abc123"}
        },
        )
     
    return ai_response


"""
    refined_question = dictionary_chain.invoke({"question": user_question})

    ai_message = rag_chain.invoke(
        {"input": refined_question},
        config={"configurable": {"session_id": "abc123"}}
    )

 

    # 가장 안전하게 답변을 가져오는 방법
    if isinstance(ai_message, dict):
        return ai_message.get("answer", ai_message.get("content", "답변을 생성하지 못했습니다."))
    else:
        # ai_message 자체가 문자열인 경우 (드문 경우)
        return ai_message

"""





