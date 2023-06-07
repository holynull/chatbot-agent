import os
from dotenv import load_dotenv
from chain import qa_chain 
from langchain.chat_models import ChatOpenAI
import chain.all_templates



# if getattr(sys, 'frozen', False):
#     script_location = pathlib.Path(sys.executable).parent.resolve()
# else:
#     script_location = pathlib.Path(__file__).parent.resolve()
load_dotenv(dotenv_path= '.env')

llm = ChatOpenAI(
	# model="gpt-4",
    temperature=0.1,
    verbose=True,
    )
headers = {
  'Accepts': 'application/json',
  'X-CMC_PRO_API_KEY': os.getenv("CMC_API_KEY"),
}
# cmc_currency_map_api=APIChain.from_llm_and_api_docs(llm=llm,api_docs=all_templates.cmc_currency_map_api_doc,headers=headers,verbose=True)
# cmc_quotes_api=APIChain.from_llm_and_api_docs(llm=llm,api_docs=all_templates.cmc_quote_lastest_api_doc,headers=headers,verbose=True)
chain = qa_chain.get_qa_chain(chain_type="map_reduce",index_name="data-swft",verbose=True)
input=input() 
print(f"Output:{chain(inputs=input)}")