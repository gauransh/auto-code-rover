"""
For models other than those from OpenAI, use LiteLLM if possible.
Create all models managed by Ollama here, since they need to talk to ollama server.
"""

import sys
from collections.abc import Mapping
from copy import deepcopy
from typing import Literal, cast, Optional, List
import httpx
import ollama
import timeout_decorator
from ollama._types import Message, Options
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from app.model import common
from app.model.common import Model
from tenacity import retry, stop_after_attempt, wait_random_exponential
import time
import json
from app.data_structures import FunctionCallIntent

class OllamaModel(Model):
    """
    Base class for creating Singleton instances of Ollama models.
    """

    _instances = {}

    def __new__(cls):
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)
            cls._instances[cls]._initialized = False
        return cls._instances[cls]

    def __init__(self, name: str, cost_per_second: float):
        if self._initialized:
            return
        # local models are free
        super().__init__(name, 0.0, 0.0)
        self.client: ollama.Client | None = None
        self.cost_per_second = cost_per_second
        self._initialized = True

    def setup(self) -> None:
        """
        Check API key.
        """
        self.check_api_key()
        try:
            self.send_empty_request()
            print(f"Model {self.name} is up and running.")
        except timeout_decorator.TimeoutError as e:
            print(
                "Ollama server is taking too long (more than 2 mins) to respond. Please check whether it's running.",
                e,
            )
            sys.exit(1)
        except Exception as e:
            print("Could not communicate with ollama server due to exception.", e)
            sys.exit(1)

    @timeout_decorator.timeout(120)  # 2 min
    def send_empty_request(self):
        """
        Send an empty request to the model, for two purposes
        (1) check whether the model is up and running
        (2) preload the model for faster response time (models will be kept in memory for 5 mins after loaded)
        (see https://github.com/ollama/ollama/blob/main/docs/faq.md#how-can-i-pre-load-a-model-to-get-faster-response-times)
        """
        # localhost is used when (1) running both ACR and ollama on host machine; and
        #   (2) running ollama in host, and ACR in container with --net=host
        local_client = ollama.Client(host="http://localhost:11434")
        # docker_host_client is used when running ollama in host and ACR in container, and
        # Docker Desktop is installed
        docker_host_client = ollama.Client(host="http://host.docker.internal:11434")
        try:
            local_client.chat(model=self.name, messages=[])
            self.client = local_client
            return
        except httpx.ConnectError:
            # failed to connect to client at localhost
            pass

        try:
            docker_host_client.chat(model=self.name, messages=[])
            self.client = docker_host_client
        except httpx.ConnectError:
            # also failed to connect via host.docker.internal
            print("Could not connect to ollama server.")
            sys.exit(1)

    def check_api_key(self) -> str:
        return "No key required for local models."

    def extract_resp_content(
        self, chat_completion_message: ChatCompletionMessage
    ) -> str:
        """
        Given a chat completion message, extract the content from it.
        """
        content = chat_completion_message.content
        return content if content is not None else ""

    def extract_resp_func_calls(self, response: dict) -> list[FunctionCallIntent]:
        result = []
        tool_calls = response.get('tool_calls', [])
        for call in tool_calls:
            func_name = call.get('function', {}).get('name', '')
            func_args_str = call.get('function', {}).get('arguments', '{}')
            try:
                args_dict = json.loads(func_args_str)
            except json.JSONDecodeError:
                args_dict = {}
            func_call_intent = FunctionCallIntent(func_name, args_dict, call.get('function', {}))
            result.append(func_call_intent)
        return result

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def call(
        self,
        messages: list[dict],
        top_p: float = 1,
        tools: Optional[List[dict]] = None,
        response_format: Literal["text", "json_object"] = "text",
        temperature: Optional[float] = None,
        **kwargs,
    ) -> tuple[
        str,
        Optional[List[ChatCompletionMessageToolCall]],
        List[FunctionCallIntent],
        float,
        int,
        int,
    ]:
        stop_words = ["assistant", "\n\n \n\n"]
        json_stop_words = deepcopy(stop_words) + ["```", " " * 10]

        assert self.client is not None

        try:
            options = {"temperature": temperature or common.MODEL_TEMP, "top_p": top_p}
            if response_format == "json_object":
                json_instruction = {
                    "role": "user",
                    "content": "Stop your response after a valid json is generated.",
                }
                messages.append(json_instruction)
                options.update({"stop": json_stop_words, "num_predict": 128})
            else:
                options.update({"stop": stop_words, "num_predict": 1024})

            start_time = time.time()
            response = self.client.chat(
                model=self.name,
                messages=cast(list[Message], messages),
                options=cast(Options, options),
                stream=False,
            )
            end_time = time.time()

            assert isinstance(response, Mapping)
            resp_msg = response.get("message", {})
            content: str = resp_msg.get("content", "")

            # Calculate cost based on time
            elapsed_time = end_time - start_time
            cost = self.cost_per_second * elapsed_time

            # Extract function calls and tool calls
            func_call_intents = self.extract_resp_func_calls(resp_msg)
            tool_calls = resp_msg.get('tool_calls', None)

            # Update thread cost
            common.thread_cost.process_cost += cost
            common.thread_cost.process_input_tokens += len(str(messages))  # Approximation
            common.thread_cost.process_output_tokens += len(content)  # Approximation

            return content, tool_calls, func_call_intents, cost, len(str(messages)), len(content)

        except Exception as e:
            raise e


class Llama3_8B(OllamaModel):
    def __init__(self):
        super().__init__("llama3", 0.00001)  # Example cost per second
        self.note = "Llama3 8B model."

class Llama3_70B(OllamaModel):
    def __init__(self):
        super().__init__("llama3:70b", 0.00001)
        self.note = "Llama3 70B model."

class Llama3_1_8B(OllamaModel):
    def __init__(self):
        super().__init__("llama3.1", 0.00001)
        self.note = "Llama3.1 8B model."

class Llama3_1_70B(OllamaModel):
    def __init__(self):
        super().__init__("llama3.1:70b", 0.00001)
        self.note = "Llama3.1 70B model."

class Llama3_1_405B(OllamaModel):
    def __init__(self):
        super().__init__("llama3.1:405b", 0.00001)
        self.note = "Llama3.1:405B model."

class FTllama3_1_8B(OllamaModel):
    def __init__(self):
        super().__init__("gtandon/ft_llama3_1_swe_bench", 0.00001)
        self.note = "FT_llama_3.1 8b model."

class Nemotron(OllamaModel):
    def __init__(self):
        super().__init__("nemotron", 0.00001)
        self.note = "Nemotron Llama3.1 70B model.

class MistralLarge(OllamaModel):
    def __init__(self):
        super().__init__("mistral-large", 0.00001)
        self.note = "Mistral Large model."
class DeepSeekCoder_V2_16B(OllamaModel):
    def __init__(self):
        super().__init__("deepseek-coder-v2", 0.00001)
        self.note = "Deep-seek-coder-v2:16b model."

class DeepSeekCoder_V2_236B(OllamaModel):
    def __init__(self):
        super().__init__("deepseek-coder-v2:236b", 0.00001)
        self.note = "Deep-seek-coder-v2:236b model."
