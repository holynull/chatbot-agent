"""Create a ChatVectorDBChain for question/answering."""
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.memory import ConversationBufferMemory 
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.base import VectorStore
from langchain.utilities import GoogleSerperAPIWrapper
import os
from langchain.agents import Tool
from langchain.agents import initialize_agent,AgentType,AgentExecutor
from langchain.chains import RetrievalQA


def get_agent(
    chain_type: str, vectorstore: VectorStore, agent_cb_handler) -> AgentExecutor:
    agent_cb_manager = AsyncCallbackManager([agent_cb_handler])
    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0,
        verbose=True,
        # request_timeout=120,
    )
    search = GoogleSerperAPIWrapper()
    doc_search = RetrievalQA.from_chain_type(llm=llm, chain_type=chain_type, retriever=vectorstore.as_retriever())
    tools = [
		Tool(
        	name = "QA System",
        	func=doc_search.run,
        	description="useful for when you need to answer questions about ruff (a python linter). Input should be a fully formed question."
        ),
        Tool(
            name = "Current Search",
            func=search.run,
            description="useful for when you need to answer questions about current events or the current state of the world. the input to this should be a single search term."
        ),
    ]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    agent_chain = initialize_agent(tools=tools, llm=llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory,callback_manager=agent_cb_manager)
    return agent_chain 
