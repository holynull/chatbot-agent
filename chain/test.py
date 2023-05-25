import pathlib
import sys
from dotenv import load_dotenv
from cc_balance_chain import from_llm 
from langchain.chat_models import ChatOpenAI


# if getattr(sys, 'frozen', False):
#     script_location = pathlib.Path(sys.executable).parent.resolve()
# else:
#     script_location = pathlib.Path(__file__).parent.resolve()
load_dotenv(dotenv_path= '.env')

llm = ChatOpenAI(temperature=0.9)
chain = from_llm(llm=llm,verbose=True)
input=input() 
print(chain(inputs=input))