#design graph node & edge
from state import response_state, StateGraph
from agents.llm_process import retrieve_agent, get_tools ,should_continue
from langgraph.graph import START , END
from IPython.display  import display, Image

from langgraph.checkpoint.memory import MemorySaver




graph_builder = StateGraph(response_state)
graph_builder.add_node('retrieve_agent', retrieve_agent)
graph_builder.add_node('tools', get_tools)

 

graph_builder.add_edge(START, 'retrieve_agent')

graph_builder.add_conditional_edges('retrieve_agent', should_continue,
 {'tools': 'tools',
 'end': END
 })
 
graph_builder.add_edge('tools','retrieve_agent')


checkpointer = MemorySaver()

graph = graph_builder.compile(checkpointer = checkpointer)

display(Image(graph.get_graph().draw_mermaid_png()))