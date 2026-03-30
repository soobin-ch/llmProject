#UI Component with streamlit
import dotenv 
import streamlit as st


dotenv.load_dotenv()

st.set_page_config(page_title="Chatbot Interface", page_icon="🤖")

st.title("😊 부동산세 챗봇")
st.caption("부동산 세금 관련 질문에 답변해 드립니다.")
