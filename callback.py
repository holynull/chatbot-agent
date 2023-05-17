"""Callback handlers used in the app."""
from typing import Any, Dict, List, Optional, Union
from uuid import UUID
from langchain.callbacks.base import AsyncCallbackHandler

from schemas import ChatResponse


class StreamingLLMCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def __init__(self, websocket):
        self.websocket = websocket

    # def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
    #     print(f"llm new token: {token}")
        # resp = ChatResponse(sender="bot", message=token, type="stream")
        # await self.websocket.send_json(resp.dict())


class ChainCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def __init__(self, websocket):
        self.websocket = websocket

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        print(f"Inputs: {inputs}")

    async def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        print(f"Outputs: {outputs}")
        if outputs['answer'] != None:
            resp = ChatResponse(
                sender="bot", message=outputs['answer'], type="stream")
        else:
            resp = ChatResponse(
                sender="bot", message="Synthesizing question...", type="info"
            )
        await self.websocket.send_json(resp.dict())


class QuestionGenCallbackHandler(AsyncCallbackHandler):
    """Callback handler for question generation."""

    def __init__(self, websocket):
        self.websocket = websocket

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        resp = ChatResponse(
            sender="bot", message="Synthesizing question...", type="info"
        )
        await self.websocket.send_json(resp.dict())
