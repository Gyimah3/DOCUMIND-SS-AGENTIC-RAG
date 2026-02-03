import asyncio
from typing import Awaitable, Callable, Dict, List, Set, Union

from loguru import logger

from api.utils.types import ActionType


class ModelRegistry:
    _models: Set[str] = set()
    _action_types: List[str] = [action.value for action in ActionType]
    _on_model_registered_callbacks: List[
        Callable[[str], Union[None, Awaitable[None]]]
    ] = []

    @classmethod
    def register_model(cls, model_name: str) -> None:
        if model_name not in cls._models:
            cls._models.add(model_name)
            for callback in cls._on_model_registered_callbacks:
                try:
                    result = callback(model_name)
                    if asyncio.iscoroutine(result):
                        asyncio.create_task(result)
                except Exception as e:
                    logger.error(
                        f"Error in model registration callback for {model_name}: {e}"
                    )

    @classmethod
    def register_on_model_registered_callback(
        cls, callback: Callable[[str], Union[None, Awaitable[None]]]
    ) -> None:
        cls._on_model_registered_callbacks.append(callback)

    @classmethod
    def get_registered_models(cls) -> List[str]:
        return sorted(cls._models)

    @classmethod
    def clear_registry(cls) -> None:
        cls._models.clear()


model_registry = ModelRegistry()
