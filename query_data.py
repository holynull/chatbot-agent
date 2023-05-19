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
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.llm import LLMChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.question_answering import load_qa_chain
                                                     

def get_qa_chain(
    chain_type: str, vectorstore: VectorStore
) -> ConversationalRetrievalChain:
    """Create a ChatVectorDBChain for question/answering."""
    # Construct a ChatVectorDBChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    question_gen_llm = ChatOpenAI(
        temperature=0,
        verbose=True,
    )
    streaming_llm = ChatOpenAI(
        streaming=True,
        verbose=True,
        temperature=0,
    )

    question_generator = LLMChain(
        llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT,  verbose=True,
    )
    doc_chain = load_qa_chain(
        streaming_llm, chain_type=chain_type,   verbose=True,
    )
    # memory = ConversationBufferMemory(
    #     memory_key='chat_history', return_messages=True, output_key='answer')
    # memory.chat_memory.add_ai_message("I'm the CMO of SWFT Blockchain and Metapath. What can I help you?")
    qa = ConversationalRetrievalChain(         # <==CHANGE  ConversationalRetrievalChain instead of ChatVectorDBChain
        # vectorstore=vectorstore,             # <== REMOVE THIS
        retriever=vectorstore.as_retriever(),  # <== ADD THIS
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        verbose=True,
        # memory=memory,
        # max_tokens_limit=4096,
    )
    return qa

def get_agent(
    chain_type: str, vectorstore: VectorStore, agent_cb_handler) -> AgentExecutor:
    agent_cb_manager = AsyncCallbackManager([agent_cb_handler])
    llm = ChatOpenAI(
        # model_name="gpt-4",
        temperature=0,
        verbose=True,
        # request_timeout=120,
    )
    search = GoogleSerperAPIWrapper()
    # doc_search = RetrievalQA.from_chain_type(llm=llm, chain_type=chain_type, retriever=vectorstore.as_retriever())
    doc_search = get_qa_chain(chain_type=chain_type,vectorstore=vectorstore) 
    tools = [
		Tool(
        	name = "QA System",
        	func=doc_search.run,
        	description="当您需要回答有关swft或metapath的问题时，这很有用。输入应该是一个完整的问题。"
        ),
        Tool(
            name = "Current Search",
            func=search.run,
            description="当你需要回答有关时事或世界现状的问题时，这很有用。对此的输入应该是单个搜索项。"
        ),
    ]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    agent_chain = initialize_agent(tools=tools, llm=llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory,callback_manager=agent_cb_manager)
    return agent_chain 
