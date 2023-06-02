from dataclasses import dataclass
from typing import Any, List

import streamlit as st
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import Embeddings, OpenAIEmbeddings
from langchain.llms import GPT4All

from datachad.constants import GPT4ALL_BINARY, MODEL_PATH
from datachad.utils import logger


class Enum:
    @classmethod
    def all(cls) -> List[Any]:
        return [v for k, v in cls.__dict__.items() if not k.startswith("_")]


@dataclass
class Model:
    name: str
    mode: str
    embedding: str
    path: str = None  # for local models only

    def __str__(self) -> str:
        return self.name


class MODES(Enum):
    # Add more modes as needed
    OPENAI = "OpenAI"
    LOCAL = "Local"


class EMBEDDINGS(Enum):
    # Add more embeddings as needed
    OPENAI = "openai"
    HUGGINGFACE = "all-MiniLM-L6-v2"


class MODELS(Enum):
    # Add more models as needed
    GPT35TURBO = Model(
        name="gpt-3.5-turbo", mode=MODES.OPENAI, embedding=EMBEDDINGS.OPENAI
    )
    GPT4 = Model(name="gpt-4", mode=MODES.OPENAI, embedding=EMBEDDINGS.OPENAI)
    GPT4ALL = Model(
        name="GPT4All",
        mode=MODES.LOCAL,
        embedding=EMBEDDINGS.HUGGINGFACE,
        path=str(MODEL_PATH / GPT4ALL_BINARY),
    )

    @classmethod
    def for_mode(cls, mode) -> List[Model]:
        return [m for m in cls.all() if isinstance(m, Model) and m.mode == mode]


def get_model() -> BaseLanguageModel:
    with st.session_state["info_container"], st.spinner("Loading Model..."):
        model_name = st.session_state["model"].name
        temperature = st.session_state["temperature"]
        openai_api_key = st.session_state["openai_api_key"]
        model_path = st.session_state["model"].path
        model_n_ctx = st.session_state["model_n_ctx"]
        
        if model_name == MODELS.GPT35TURBO.name:
            model = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                openai_api_key=openai_api_key,
            )
        elif model_name == MODELS.GPT4.name:
            model = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                openai_api_key=openai_api_key,
            )
        elif model_name == MODELS.GPT4ALL.name:
            model = GPT4All(
                model=model_path,
                n_ctx=model_n_ctx,
                backend="gptj",
                temp=temperature,
                verbose=True,
                callbacks=[StreamingStdOutCallbackHandler()],
            )
        else:
            msg = f"Model {model_name} not supported!"
            logger.error(msg)
            st.error(msg)
            exit()
    return model


def get_embeddings() -> Embeddings:
    embedding_type = st.session_state["model"].embedding
    if embedding_type == EMBEDDINGS.OPENAI:
        embeddings = OpenAIEmbeddings(
            disallowed_special=(), openai_api_key=st.session_state["openai_api_key"]
        )
    elif embedding_type == EMBEDDINGS.HUGGINGFACE:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDINGS.HUGGINGFACE, cache_folder=str(MODEL_PATH)
        )
    else:
        msg = f"Embeddings {embedding_type} not supported!"
        logger.error(msg)
        st.error(msg)
        exit()
    return embeddings

