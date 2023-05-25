
from langchain import LLMChain, PromptTemplate
from langchain.chains.base import Chain
from langchain.base_language import BaseLanguageModel
from langchain.prompts.chat import ChatPromptTemplate,HumanMessagePromptTemplate


def from_llm(llm:BaseLanguageModel,verbose:bool) ->LLMChain:
	_template="""
	请总结出用户提问中，提到的币种（symbol）、币种所在的区块链名称（chainId）和要查询的地址地址（address）。并组成JSON格式的字符串。
	例如，提问“请帮我查询一下地址0xB0d88027F5dEd975fF6Df7A62952033D67Df277f上的usdt的余额”，
	则总结的JSON如下
	```json
	{{{{
		"symbol":"USDT",
		"chainId":["BSC","ETH"],
		"address":"0xB0d88027F5dEd975fF6Df7A62952033D67Df277f"
	}}}}
	```
	用户提问：{question}
	给出总结：
	"""
	human_message_prompt = HumanMessagePromptTemplate(
      	 prompt=PromptTemplate(
      	     template=_template,
      	     input_variables=["question"],
      	 )
   	)
	prompt = ChatPromptTemplate.from_messages([human_message_prompt])
	return LLMChain(llm=llm, prompt=prompt,verbose=verbose)
