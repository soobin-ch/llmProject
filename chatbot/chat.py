import streamlit as st


from dotenv import load_dotenv

from chatbot.llm import get_ai_response



st.set_page_config(page_title="Chatbot Interface", page_icon="🤖")

st.title("😊 소득세 챗봇")
st.caption("소득세 법률 관련 질문에 답변해 드립니다.")

load_dotenv()  # take environment variables from .env file


if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Initialize chat history



if user_prompt := st.chat_input("Say something"):

    with st.chat_message("user"):
        st.write(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.spinner("답변을 생성하는 중입니다"):
        ai_response = get_ai_response(user_prompt)

        with st.chat_message("assistant"):
            ai_message = st.write_stream(ai_response)
            st.session_state.messages.append({"role": "assistant", "content": ai_message})

