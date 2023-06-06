import os
from dotenv import load_dotenv
from chain import cmc_quotes_chain 
from langchain.chat_models import ChatOpenAI
from langchain.chains import APIChain
import chain.all_templates
from langchain.prompts import PromptTemplate
from langchain import HuggingFaceHub



# if getattr(sys, 'frozen', False):
#     script_location = pathlib.Path(sys.executable).parent.resolve()
# else:
#     script_location = pathlib.Path(__file__).parent.resolve()
load_dotenv(dotenv_path= '.env')

repo_id = "tiiuae/falcon-7b"
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0.7, "max_length":64})
headers = {
  'Accepts': 'application/json',
  'X-CMC_PRO_API_KEY': os.getenv("CMC_API_KEY"),
}
# cmc_currency_map_api=APIChain.from_llm_and_api_docs(llm=llm,api_docs=all_templates.cmc_currency_map_api_doc,headers=headers,verbose=True)
# cmc_quotes_api=APIChain.from_llm_and_api_docs(llm=llm,api_docs=all_templates.cmc_quote_lastest_api_doc,headers=headers,verbose=True)
chain = cmc_quotes_chain.CMCQuotesChain.from_llm(llm=llm,headers=headers,verbose=True)
input=input() 
print(f"Output:{chain(inputs=input)}")