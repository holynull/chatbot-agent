from __future__ import annotations

from typing import Any, Dict, List, Optional

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
import all_templates

prompt=PromptTemplate(template=all_templates.quotes_chain_template,input_variables=["user_input"])

class CMCQuotesChain(Chain):
    """
    An example of a custom chain.
    """

    prompt: BasePromptTemplate=prompt
    """Prompt object to use."""
    llm: BaseLanguageModel
    output_key: str = "text"  #: :meta private:
    
    cmc_quotes_api:APIChain 

    @classmethod
    def from_llm(cls,llm:BaseLanguageModel,headers:dict,**kwargs: Any,)->CMCQuotesChain:
        api=APIChain.from_llm_and_api_docs(llm=llm,api_docs=all_templates.cmc_quote_lastest_api_doc,headers=headers,**kwargs)
        return cls(llm=llm,cmc_quotes_api=api,**kwargs)

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

        if response.generations[0][0].text!="NAN":
            template=PromptTemplate(input_variables=["question"],template=all_templates.cc_map_api_template)
            question=template.format(question=response.generations[0][0].text)
            # print(f"Question: {question}")
            # json=self.cmc_currency_map_api.run(question)
            # print(f"Json: {json}")
            template2=PromptTemplate(input_variables=["question"],template=all_templates.replace_name_to_id_template)
            p=template2.format(question=question)
            res=self.cmc_quotes_api.run(p) 
            return {self.output_key: res}
        else:
            return {self.output_key: response.generations[0][0].text}
        

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
            await run_manager.on_text("Log something about this run")
        
        return {self.output_key: response.generations[0][0].text}

    @property
    def _chain_type(self) -> str:
        return "my_custom_chain"