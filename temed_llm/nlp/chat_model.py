import logging
import os
import time
from typing import Generator, List, Union

import openai
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFacePipeline, OpenAI
from langchain.schema import HumanMessage, SystemMessage
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# SET LOGGING
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "Examine electronic health records, extract specified key-value pairs, and compile them into a JSON output that adheres to the provided schema."


class InterfaceLLM:
    def __init__(self, **kwargs):
        self.llm = "InstructLLMs"

    def get_model(self):
        return self.llm

    def __call__(self, messages):
        return self.llm

    def forward(self, messages):
        return self.llm


class EndpointsProHF(InterfaceLLM):
    def __init__(self, model_id, temperature=0):
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError(
                "Please install huggingface_hub first via `pip install huggingface_hub` for running HF LLMs"
            )

        token = os.getenv("HF_TOKEN", "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")
        if token is None:
            raise ValueError(
                "Please set the HF_TOKEN environment variable to your Hugging Face API token."
            )

        self.llm = InferenceClient(model="meta-llama/Llama-2-70b-chat-hf", token=token)

    def get_model(self):
        return self.llm

    def __call__(self, messages):
        return self.llm.text_generation(prompt=messages, max_new_tokens=1000, temperature=0)

    def forward(self, messages):
        return self.llm.text_generation(prompt=messages, max_new_tokens=1000, temperature=0)


class InstructOpenAI(InterfaceLLM):
    def __init__(self, temperature=0, openai_api_key=None):
        self.temperature = temperature
        self.openai_api_key = openai_api_key
        self.llm = OpenAI(openai_api_key=self.openai_api_key, temperature=self.temperature)

    def get_model(self):
        return self.llm

    def __call__(self, messages):
        return self.llm(messages)

    def forward(self, messages):
        return self.llm(messages)


class DeepInfraLLM:
    def __init__(
        self,
        llm: str = "meta-llama/Llama-2-70b-chat-hf",
        temperature: float = 0.0,
        stream: bool = False,
        max_retries: int = 2,
        retry_interval: int = 30,
    ):
        # necessary for openai api
        openai.api_key = os.getenv("DEEPINFRA_TOKEN", "QLURf8b11lcQkXDtwRgb4hbCgdYNOzUt")
        openai.api_base = "https://api.deepinfra.com/v1/openai"

        self.llm = llm
        self.temperature = temperature
        self.stream = stream
        self.max_retries = max_retries
        self.retry_interval = retry_interval

    def _response_stream(self, response):
        """Yield response chunks from a stream."""
        for chunk in response:
            if "content" in chunk["choices"][0]["delta"].keys():
                yield chunk["choices"][0]["delta"].get("content", "")

    def _prepare_response(self, response):
        """Prepare response for display."""
        if self.stream:
            return self._response_stream(response)
        else:
            return response["choices"][-1]["message"]["content"]

    def get_model(self):
        return self.llm

    def to_openai_messages(self, human_prompt: str, system_prompt: str = SYSTEM_PROMPT) -> List:
        """Return prompt as OpenAI messages format."""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt},
        ]

    def forward(self, prompt: str) -> Union[str, Generator[str, None, None]]:
        """Generate response from an LLM."""
        messages = self.to_openai_messages(human_prompt=prompt)
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                response = openai.ChatCompletion.create(
                    model=self.llm,
                    temperature=self.temperature,
                    stream=self.stream,
                    messages=messages,
                )
                return self._prepare_response(response)

            except Exception as e:
                print(f"Exception: {e}")
                time.sleep(self.retry_interval)
                retry_count += 1
        return ""

    def __call__(self, prompt: str) -> Union[str, Generator[str, None, None]]:
        return self.forward(prompt)


# Using gpt-3.5-turbo which is pretty cheap, but has worse quality
class ChatLLM(InterfaceLLM):
    def __init__(
        self,
        temperature=0,
        openai_api_key=None,
        system_message="You are a expert in medical electronic health records and write expert medical reports. You follow instructions closely.",
    ):
        self.temperature = temperature
        self.openai_api_key = openai_api_key
        self.system_message = system_message
        self.llm = ChatOpenAI(openai_api_key=self.openai_api_key)

    def get_model(self):
        return self.llm

    def __call__(self, messages):
        messages = [SystemMessage(content=self.system_message), HumanMessage(content=messages)]
        return self.llm(messages)

    def forward(self, messages):
        messages = [SystemMessage(content=self.system_message), HumanMessage(content=messages)]
        return self.llm(messages).content


# class InstructHuggingFace(InterfaceLLM):
#     def __init__(self, model_id, temperature=0):
#         try:
#             import torch
#         except ImportError:
#             raise ImportError(
#                 "Please install torch first via `pip install torch` for running HF LLMs"
#             )

#         self.temperature = temperature
#         self.model_id = model_id

#         self.tokenizer = AutoTokenizer.from_pretrained(model_id)
#         self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # for open-ended generation

#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_use_double_quant=True,
#         )

#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_id,
#             quantization_config=bnb_config,
#             device_map="auto",
#             trust_remote_code=True,
#         )

#         self.pipe = pipeline(
#             "text-generation",
#             model=self.model,
#             tokenizer=self.tokenizer,
#             trust_remote_code=True,
#             device_map="auto",  # finds GPU
#         )
#         self.llm = HuggingFacePipeline(pipeline=self.pipe)

# def get_model(self):
#     return self.llm

# def __call__(self, messages):
#     return self.llm(messages)

# def forward(self, messages):
#     return self.llm(messages)


def create_llm(model_name="gpt-3.5-turbo", openai_api_key="", temperature=0):
    """Initialize the OpenAI LLM model."""
    if "gpt" in model_name:
        huggingface_model = False
    else:
        huggingface_model = True

    if model_name == "gpt-3.5-turbo":
        logger.info("Using gpt-3.5-turbo model.")
        llm = ChatLLM(temperature=temperature, openai_api_key=openai_api_key)
    elif model_name == "meta-llama/Llama-2-70b-chat-hf":
        logger.info("Using HuggingFaceHub model. Repo ID: " + model_name)
        llm = DeepInfraLLM()
    elif huggingface_model is True:
        logger.info("Using HuggingFaceHub model. Repo ID: " + model_name)
        llm = None
    else:
        logger.info("Using Instruct GPT model.")
        llm = InstructOpenAI(temperature=temperature, openai_api_key=openai_api_key)

    return llm
