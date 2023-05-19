"""Callback handlers used in the app."""
from typing import Any, Dict, List, Optional, Union
from uuid import UUID
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.callbacks.base import LLMResult
from schemas import ChatResponse
from langchain.schema import AgentAction,AgentFinish

class AgentCallbackHandler(AsyncCallbackHandler):
    """Callback handler for question generation."""

    def __init__(self, websocket):
        self.websocket = websocket

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        print(f"ON_LLM_START")
        # resp = ChatResponse(
        #     sender="bot", message="Synthesizing question...", type="info"
        # )
        # self.websocket.send_json(resp.dict())

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM ends running.""" 
        print(f"ON_LLM_END: {response.dict}")
    # def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
    #     print(f"llm new token: {token}")
    #     resp = ChatResponse(sender="bot", message=token, type="stream")
    #     self.websocket.send_json(resp.dict())

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        print(f"ON_CHAIN_START: Inputs: {inputs}")

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        print(f"ON_CHAIN_END: Outputs: {outputs}")
        # resp = ChatResponse(sender="bot", message=outputs['output'], type="stream")
        # await self.websocket.send_json(resp.dict())
        # resp = ChatResponse(sender="bot", message=outputs['output'], type="stream")
        # self.websocket.send_json(resp.dict())
        # if outputs['answer'] != None:
        #     resp = ChatResponse(
        #         sender="bot", message=outputs['answer'], type="stream")
        # else:
        #     resp = ChatResponse(
        #         sender="bot", message="Synthesizing question...", type="info"
        #     )
        # self.websocket.send_json(resp.dict())
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool starts running."""
        print(f"ON_TOOL_START: input: {input_str}")

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool ends running."""
        print(f"ON_TOOL_END: output: {output}")
    
    async def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run on agent end."""
        print(f"ON_AGENT_FINISH: {finish.return_values}")
        resp = ChatResponse(sender="bot", message=finish.return_values['output'], type="stream")
        await self.websocket.send_json(resp.dict())
        

    
    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run on agent action."""
        print(f"ON_AGENT_ACTION: tool: {action.tool}")
