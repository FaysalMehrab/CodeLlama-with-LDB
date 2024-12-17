from typing import List, Union, Optional, Literal
import dataclasses
import time
import logging
from vllm import LLM, SamplingParams
from tenacity import (
    retry,
    stop_after_attempt,  # type: ignore
    wait_random_exponential,  # type: ignore
)
from openai import OpenAI
from transformers import AutoTokenizer

MessageRole = Literal["system", "user", "assistant"]
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

@dataclasses.dataclass
class Message:
    role: MessageRole
    content: str

def message_to_str(message: Message) -> str:
    return f"{message.role}: {message.content}"

def messages_to_str(messages: List[Message]) -> str:
    return "\n".join([message_to_str(message) for message in messages])

def change_messages(tokenizer, messages, max_len):
    if isinstance(messages, str):
        message_lines = messages.split("\n")
        acc_msg_len = 0
        new_messages = ""
        for l in reversed(message_lines):
            acc_msg_len += len(tokenizer.tokenize(l))
            if acc_msg_len < max_len:
                new_messages = l + "\n" + new_messages
            else:
                break
        new_messages = new_messages.strip()
        return new_messages
    else:
        original_messages = messages
        new_messages = messages[:1]
        total_msg_len = len(tokenizer.tokenize(messages[0].content))
        rest_messages = []
        for msg in reversed(messages[1:]):
            msg_len = len(tokenizer.tokenize(msg.content))
            if msg_len + total_msg_len < max_len:
                rest_messages = [msg] + rest_messages
                total_msg_len += msg_len
            else:
                break
        messages = new_messages + rest_messages
    return messages

class ModelBase:
    def __init__(self, name: str):
        self.name = name
        self.is_chat = False

    def __repr__(self) -> str:
        return f'{self.name}'

    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1) -> Union[List[str], str]:
        raise NotImplementedError

    def generate(self, prompt: str, max_tokens: int = 1024, stop_strs: Optional[List[str]] = None, temperature: float = 0.0, num_comps=1) -> Union[List[str], str]:
        raise NotImplementedError

class VLLMModelBase(ModelBase):
    """
    Base for huggingface chat models
    """

    def __init__(self, model, port=""):
        super().__init__(model)
        port = port or "8000"
        self.model = model
        self.vllm_client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.max_length = 7000

        # Initialize logger
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
    
    def vllm_chat(
        self,
        prompt: str,
        stop: List[str] = [""],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        num_comps=1,
    ) -> Union[List[str], str]:
        max_length = self.max_length
        retries = 3  # Number of retries for connection issues
        for attempt in range(retries):
            try:
                self.logger.debug(f"Attempt {attempt+1}: Sending request to VLLM API")
                prompt = change_messages(self.tokenizer, prompt, max_length)
                responses = self.vllm_client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    echo=False,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=1,
                    stop=stop,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    n=num_comps,
                )
                self.logger.debug(f"Attempt {attempt+1}: Successfully received response from VLLM API")
                break
            except Exception as e:
                self.logger.error(f"Attempt {attempt+1}: Error - {e}")
                if "maximum context length" in str(e):
                    max_length -= 2000
                elif attempt < retries - 1:
                    time.sleep(2)  # Wait before retrying
                else:
                    raise AssertionError("VLLM API error: " + str(e))
        if num_comps == 1:
            return responses.choices[0].text  # type: ignore
        return [response.choices[0].text for response in responses]  # type: ignore

    def generate_completion(self, messages: str, stop: List[str] = [""], max_tokens: int = 1024, temperature: float = 0.0, num_comps: int = 1) -> Union[List[str], str]:
        ret = self.vllm_chat(messages, stop, max_tokens, temperature, num_comps)
        return ret

    def prepare_prompt(self, messages: List[Message]):
        prompt = ""
        for i, message in enumerate(messages):
            prompt += message.content + "\n"
            if i == len(messages) - 1:
                prompt += "\n"
        return prompt

    def extract_output(self, output: str) -> str:
        return output

class CodeLlama(VLLMModelBase):
    def __init__(self, port="8000"):
        super().__init__("codellama/CodeLlama-7b-Python-hf", port)
