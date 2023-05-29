from __future__ import annotations

from typing import Any, Dict, List, Optional
from langchain import LLMChain

from pydantic import Extra

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.chains import APIChain
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

from chain import all_templates

prompt=PromptTemplate(template=all_templates.quotes_chain_template,input_variables=["user_input"])

class CMCQuotesChain(Chain):
    """
    An example of a custom chain.
    """

    prompt: BasePromptTemplate=prompt
    """Prompt object to use."""
    llm: BaseLanguageModel
    output_key: str = "text"  #: :meta private:

    consider_chain:LLMChain
    
    cmc_quotes_api:APIChain 

    answer_chain:LLMChain

    

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        prompt_value = self.prompt.format_prompt(**inputs)
        
        # Whenever you call a language model, or another chain, you should pass
        # a callback manager to it. This allows the inner run to be tracked by
        # any callbacks that are registered on the outer run.
        # You can always obtain a callback manager for this by calling
        # `run_manager.get_child()` as shown below.
        response = self.llm.generate_prompt(
            [prompt_value],
            callbacks=run_manager.get_child() if run_manager else None
        )

        # If you want to log something about this run, you can do so by calling
        # methods on the `run_manager`, as shown below. This will trigger any
        # callbacks that are registered for that event.
        if run_manager:
            run_manager.on_text(response.generations[0][0].text, color="green", end="\n", verbose=self.verbose)
        original_question=response.generations[0][0].text
        product=self.consider_chain.run(question=original_question)
        if run_manager:
            run_manager.on_text(product, color="yellow", end="\n", verbose=self.verbose) 
        try:
            res=self.cmc_quotes_api.run(product) 
        except Exception as err:
            answer=self.answer_chain.run(question=inputs['user_input'],context=err.args)
            return {self.output_key: answer}
        answer=self.answer_chain.run(question=inputs['user_input'],context=res)
        if run_manager:
            run_manager.on_text(answer, color="yellow", end="\n", verbose=self.verbose) 
        return {self.output_key: answer}
        

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        prompt_value = self.prompt.format_prompt(**inputs)
        
        # Whenever you call a language model, or another chain, you should pass
        # a callback manager to it. This allows the inner run to be tracked by
        # any callbacks that are registered on the outer run.
        # You can always obtain a callback manager for this by calling
        # `run_manager.get_child()` as shown below.
        response = await self.llm.agenerate_prompt(
            [prompt_value],
            callbacks=run_manager.get_child() if run_manager else None
        )

        # If you want to log something about this run, you can do so by calling
        # methods on the `run_manager`, as shown below. This will trigger any
        # callbacks that are registered for that event.
        if run_manager:
            await run_manager.on_text(response.generations[0][0].text, color="green", end="\n", verbose=self.verbose)
        original_question=response.generations[0][0].text
        product=await self.consider_chain.arun(question=original_question)
        if run_manager:
            await run_manager.on_text(product, color="yellow", end="\n", verbose=self.verbose) 
        try:
            res=await self.cmc_quotes_api.arun(product) 
        except Exception as err:
            answer=await self.answer_chain.arun(question=inputs['user_input'],context=err.args)
            return {self.output_key: answer}
        answer=await self.answer_chain.arun(question=inputs['user_input'],context=res)
        if run_manager:
            await run_manager.on_text(answer, color="yellow", end="\n", verbose=self.verbose) 
        return {self.output_key: answer}

    @property
    def _chain_type(self) -> str:
        return "cmc_quotes_chain"
    
    @classmethod
    def from_llm(cls,llm:BaseLanguageModel,headers:dict,**kwargs: Any,)->CMCQuotesChain:
        API_URL_PROMPT_TEMPLATE = """You are given the below API Documentation:
        {api_docs}
        Using this documentation, generate the full API url to call for answering the user question.
        You should build the API url in order to get a response that is as short as possible. Pay attention to deliberately exclude any unnecessary pieces of data in the API call.
        You should not build API url with the word "aux".
        Question:{question}
        API url:"""

        API_URL_PROMPT = PromptTemplate(
            input_variables=[
                "api_docs",
                "question",
            ],
            template=API_URL_PROMPT_TEMPLATE,
        )
        api_llm=ChatOpenAI(
            # model_name="gpt-4",
            temperature=0,
            # request_timeout=120,
            **kwargs
        )
        api=APIChain.from_llm_and_api_docs(llm=api_llm,api_docs=all_templates.cmc_quote_lastest_api_doc,api_url_prompt=API_URL_PROMPT,headers=headers,**kwargs)
        # api=APIChain.from_llm_and_api_docs(llm=llm,api_docs=all_templates.cmc_quote_lastest_api_doc,headers=headers,**kwargs)
        consider_prompt=PromptTemplate(
            input_variables=["question"],
            template=all_templates.consider_what_is_the_product
        )
        consider_llm=ChatOpenAI(
            # model_name="gpt-4",
            temperature=0,
            # request_timeout=120,
            **kwargs
        )
        consider=LLMChain(llm=consider_llm,prompt=consider_prompt,**kwargs)
        answer_llm=ChatOpenAI(
            # model_name="gpt-4",
            temperature=0.9,
            # request_timeout=120,
            **kwargs
        )
        answer_template=PromptTemplate(
            input_variables=["context","question"],
            template=all_templates.quotes_chain_answer,
        )
        answer_chain=LLMChain(llm=answer_llm,prompt=answer_template,**kwargs)
        return cls(llm=llm,cmc_quotes_api=api,consider_chain=consider,answer_chain=answer_chain,**kwargs)