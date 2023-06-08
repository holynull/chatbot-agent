"""Create a ChatVectorDBChain for question/answering."""
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.memory import ConversationBufferMemory 
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
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
from langchain.agents.conversational_chat.base import ConversationalChatAgent
from langchain.chains import APIChain
from langchain import HuggingFacePipeline
from chain.cmc_quotes_chain import CMCQuotesChain

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
    chain_type: str, vcs_swft: VectorStore,vcs_path: VectorStore, agent_cb_handler) -> AgentExecutor:
    agent_cb_manager = AsyncCallbackManager([agent_cb_handler])
	# llm=HuggingFacePipeline.from_model_id()
    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0.9,
        verbose=True,
    )
    llm_quotes = OpenAI(
        # model_name="gpt-4",
        temperature=0,
        verbose=True,
    )
    llm_qa = OpenAI(
        temperature=0.9,
        verbose=True,
    ) 
    search = GoogleSerperAPIWrapper()
    doc_search_swft = RetrievalQA.from_chain_type(llm=llm_qa, chain_type=chain_type, retriever=vcs_swft.as_retriever(search_kwargs={"k": 2}),verbose=True)
    doc_search_path = RetrievalQA.from_chain_type(llm=llm_qa, chain_type=chain_type, retriever=vcs_path.as_retriever(search_kwargs={"k": 2}),verbose=True)
    # doc_search = get_qa_chain(chain_type=chain_type,vectorstore=vectorstore) 
    # zapier = ZapierNLAWrapper()
    # toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': os.getenv("CMC_API_KEY"),
    }
    # cmc_quotes_api=APIChain.from_llm_and_api_docs(llm=llm,api_docs=all_templates.cmc_quote_lastest_api_doc,headers=headers,verbose=True)
    cmc_quotes_api=CMCQuotesChain.from_llm(llm=llm_quotes,headers=headers,verbose=True)
    tools = [
        Tool(
            name = "Latest Quotes and Price System",
            func=cmc_quotes_api.run,
            description="When you need to inquire about the latest cryptocurrency market trends or the latest cryptocurrency prices, you can use this tool. The input should be a complete question, and use the original language.",
            coroutine=cmc_quotes_api.arun
        ),
        Tool(
            name = "QA SWFT System",
            func=doc_search_swft.run,
            description="useful for when you need to answer questions about swft. Input should be a fully formed question, and use the original language.",
            coroutine=doc_search_swft.arun
        ),
         Tool(
            name = "QA Metapath System",
            func=doc_search_path.run,
            description="useful for when you need to answer questions about metapath. Input should be a fully formed question, and use the original language.",
            coroutine=doc_search_path.arun
        ),
        Tool(
            name = "Current Search",
            func=search.run,
            description="""
            useful for when you need to answer questions about current events or the current state of the world or you need to ask with search. 
            the input to this should be a single search term.
            """,
            coroutine=search.arun
        ),
    ]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent_excutor = initialize_agent(
        tools=tools,
        llm=llm, 
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True, 
        memory=memory,
        callback_manager=agent_cb_manager,
    )
    return agent_excutor
