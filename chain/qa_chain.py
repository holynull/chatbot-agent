from langchain.callbacks.manager import AsyncCallbackManager
from langchain.chains import ConversationalRetrievalChain, ChatVectorDBChain
from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT,
                                                     QA_PROMPT)
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
import pinecone
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

def get_qa_chain(
    chain_type: str,index_name: str,**kwargs 
) -> LLMChain:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV = os.getenv("PINECONE_ENV")
    # initialize pinecone
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_ENV  # next to api key in console
    )
    embeddings = OpenAIEmbeddings(model="gpt-4")
    vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
    question_gen_llm = ChatOpenAI(
        # model_name="gpt-4",
        temperature=0,
        **kwargs
    )
    streaming_llm = ChatOpenAI(
        # model_name="gpt-4",
        streaming=True,
        temperature=0,
        **kwargs
    )

    question_generator = LLMChain(
        llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT, **kwargs,
    )
    doc_chain = load_qa_chain(
        streaming_llm, chain_type=chain_type, **kwargs, 
    )
    qa = ConversationalRetrievalChain(         # <==CHANGE  ConversationalRetrievalChain instead of ChatVectorDBChain
        # vectorstore=vectorstore,             # <== REMOVE THIS
        retriever=vectorstore.as_retriever(),  # <== ADD THIS
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        **kwargs
    )
    return qa