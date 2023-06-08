"""Main entrypoint for the app."""
import logging
import pathlib
from pathlib import Path
import sys
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from langchain.vectorstores import VectorStore

from callback import AgentCallbackHandler  
from query_data import get_agent 
from schemas import ChatResponse

import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings

from dotenv import load_dotenv
import os

logging.basicConfig(level=logging.INFO)

if getattr(sys, 'frozen', False):
    script_location = pathlib.Path(sys.executable).parent.resolve()
else:
    script_location = pathlib.Path(__file__).parent.resolve()
load_dotenv(dotenv_path=script_location / '.env')

chain_type = os.getenv("CHAIN_TYPE")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_ENV  # next to api key in console
)

app = FastAPI()
templates = Jinja2Templates(directory="templates")
vcs_swft: Optional[VectorStore] = None
vcs_path: Optional[VectorStore] = None
from langchain.vectorstores import Chroma


@app.on_event("startup")
async def startup_event():
    logging.info("loading vectorstore")
    global vcs_swft
    global vcs_path
    index_swft = "data-swft"
    index_path = "data-metapath"
    embeddings = OpenAIEmbeddings(model="gpt-4")
    vcs_swft = Pinecone.from_existing_index(index_swft, embeddings)
    vcs_path = Pinecone.from_existing_index(index_path, embeddings)


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    agent_cb_handler = AgentCallbackHandler(websocket)
    chat_history = []
    qa_chain = get_agent(chain_type=chain_type, vcs_path=vcs_path,vcs_swft=vcs_swft, agent_cb_handler=agent_cb_handler)
    # Use the below line instead of the above line to enable tracing
    # Ensure `langchain-server` is running
    # qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)

    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())

            # result = await qa_chain.arun(input=question)
            result = await qa_chain.arun(input=question)
            print(f"Result: {result}")
            # resp = ChatResponse(sender="bot", message=result, type="stream")
            # await websocket.send_json(resp.dict())
            chat_history.append((question, result))

            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=9000)
