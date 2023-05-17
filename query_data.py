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
    chain_type: str, vectorstore: VectorStore, question_handler, stream_handler, chainCallbackHandler) -> AgentExecutor:
    question_manager = AsyncCallbackManager([question_handler])
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        verbose=True,
        callback_manager=question_manager,
        # request_timeout=120,
    )
    search = GoogleSerperAPIWrapper()
    doc_search = RetrievalQA.from_chain_type(llm=llm, chain_type=chain_type, retriever=vectorstore.as_retriever())
    tools = [
        Tool(
            name = "Current Search",
            func=search.run,
            description="useful for when you need to answer questions about current events or the current state of the world. the input to this should be a single search term."
        ),
        Tool(
        name = "QA System",
        func=doc_search.run,
        description="useful for when you need to answer questions about ruff (a python linter). Input should be a fully formed question."
        ),
    ]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    manager = AsyncCallbackManager([chainCallbackHandler])
    stream_manager = AsyncCallbackManager([stream_handler])

    
    agent_chain = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)
    return agent_chain 
