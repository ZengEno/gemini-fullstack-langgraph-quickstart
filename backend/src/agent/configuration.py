import os
from pydantic import BaseModel, Field
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

class Configuration(BaseModel):
    """The configuration for the agent."""

    provider: str = Field(
        default="aliyun",
        metadata={"description": "The provider to use for the agent's language model."},
    )

    query_generator_model: str = Field(
        # default="gemini-2.0-flash",
        default="flash",
        metadata={
            "description": "The name of the language model to use for the agent's query generation."
        },
    )

    reflection_model: str = Field(
        # default="gemini-2.5-flash-preview-04-17",
        default="flash",
        metadata={
            "description": "The name of the language model to use for the agent's reflection."
        },
    )

    answer_model: str = Field(
        # default="gemini-2.5-pro-preview-05-06",
        default="pro",
        metadata={
            "description": "The name of the language model to use for the agent's answer."
        },
    )

    number_of_initial_queries: int = Field(
        default=3,
        metadata={"description": "The number of initial search queries to generate."},
    )

    max_research_loops: int = Field(
        default=2,
        metadata={"description": "The maximum number of research loops to perform."},
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)



def get_llm(provider: str="ollama", model_type: str="flash", temperature: float=1.0, max_retries: int=2):
    if provider == "ollama":
        if model_type == "flash":
            return ChatOllama(
                model="qwen2.5:7b",
                temperature=temperature,
                max_retries=max_retries,
            )
        elif model_type == "pro":
            return ChatOllama(
                model="qwen2.5:32b",
                temperature=temperature,
                max_retries=max_retries,
            )
    elif provider == "aliyun":
        if model_type == "flash":
            return ChatOpenAI(
                model="qwen-turbo",
                temperature=temperature,
                max_retries=max_retries,
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url=os.getenv("DASHSCOPE_BASE_URL"),
            )
        elif model_type == "pro":
            return ChatOpenAI(
                model="qwen-plus",
                temperature=temperature,
                max_retries=max_retries,
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url=os.getenv("DASHSCOPE_BASE_URL"),
            )
    else:
        raise ValueError(f"Invalid provider or model type")