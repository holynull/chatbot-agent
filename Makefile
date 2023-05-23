.PHONY: start
start:
	uvicorn main_pinecone:app --reload --host "" --port 9002 >> ./root.log 2>&1 & 
start_faiss:
	uvicorn main_faiss:app --reload --host "" --port 9000 >> ./root.log 2>&1 & 
start_chroma:
	uvicorn main_chroma:app --reload --host "" --port 9000 >> ./root.log 2>&1 & 
start-map-rerank:
	uvicorn main_pinecone:app --reload --host "" --port 9001 >> ./root.log 2>&1 & 
start-chatbot:
	uvicorn main_pinecone_chatbot:app --reload --host "" --port 9002 >> ./root.log 2>&1 & 

.PHONY: format
format:
	black .
	isort .
dataset_vector_swft:
	python3 ingest_swft.py
get_twitter_SwftCoin:
	python get_twitter_data.py -u SwftCoin
get_twitter_SWFTBridge:
	python get_twitter_data.py -u SWFTBridge