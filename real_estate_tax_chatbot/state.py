#전역상태값 정의

import operator
from typing import Annotated, List, Union
from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState

class response_state(MessagesState):
    #MessagesState 객체에서 LLM이 답변한 메시지를 가져와야 합니다.
    context: Annotated[List[str], operator.add] # 리트리버된 문서 리스트를 담을 공간

    
