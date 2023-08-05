import logging
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)

# SET LOGGING
logger = logging.getLogger(__name__)

class InterfaceLLM:
    def __init__(self, **kwargs):
        self.llm = "InterfaceLLM"
    
    def get_model(self):
        return self.llm

    def __call__(self, messages):
        return self.llm(messages)

    def forward(self, messages):
        return self.llm(messages)

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


# Using gpt-3.5-turbo which is pretty cheap, but has worse quality
class ChatLLM(InterfaceLLM):
    def __init__(self, temperature=0,
                 openai_api_key=None,
                 system_message="You are a expert in medical electronic health records and write expert medical reports. You follow instructions closely."):
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


def create_llm(openai_api_key="", temperature=0, model_name="gpt-3.5-turbo"):
    """Initialize the OpenAI LLM model."""
    if model_name == "gpt-3.5-turbo":
        logger.info("Using gpt-3.5-turbo model.")
        llm = ChatLLM(temperature=temperature, openai_api_key=openai_api_key)
    else:
        logger.info("Using Instruct GPT model.")
        llm = InstructOpenAI(temperature=temperature, openai_api_key=openai_api_key)
    return llm
