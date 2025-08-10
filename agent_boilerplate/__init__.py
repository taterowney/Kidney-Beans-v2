# Unified interface for local and API models from various providers
# Written by Tate Rowney (CMU L3 Lab)

import os, re, multiprocessing
import openai
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, Field
from pathlib import Path


# OpenAI doesn't want you to use 'em, Azure can't live without 'em :(
class ChatCompletionMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender, e.g., 'user', 'assistant', 'system'.")
    content: str = Field(..., description="Content of the message.")
    name: str | None = Field(None, description="Optional name of the message sender.")
    function_call: dict | None = Field(None, description="Optional function call information if applicable.")

class ChatCompletionChoice(BaseModel):
    finish_reason: str
    index: int
    message: ChatCompletionMessage

class ChatCompletionResponse(BaseModel):
    choices: list[ChatCompletionChoice]

class ConversationStep(BaseModel):
    role: str = Field(..., description="Role of the message sender, e.g., 'system', 'user', 'assistant'.")
    content: str = Field(..., description="Content of the message.")


class Conversation(BaseModel):
    messages: list[ConversationStep] = Field(default_factory=list, description="List of messages in the conversation.")


def get_model_vendor(model_name, is_api=False):
    if model_name.lower().startswith("gpt") or re.match("^o[0-9].*$", model_name.lower()) is not None:
    # if model_name in models.list():
        return "openai"
    elif model_name.lower().startswith("gemini"):
        return "google"
    elif model_name.startswith("claude"):
        return "anthropic"
    else:
        if is_api:
            raise ValueError(f"Unsupported API model name: {model_name}")
        return "vllm"

class Client:
    def __init__(self, model_name, model_source=None, endpoint="http://localhost:8000/v1", **kwargs):
        if model_source is None:
            model_source = get_model_vendor(model_name)
        elif model_source == "api":
            model_source = get_model_vendor(model_name, is_api=True)

        self.model_source = model_source
        self.model_name = model_name
        if self.model_source == "test":
            self.client = None
            return
        if self.model_source == "vllm":
            from vllm import LLM
            self.client = LLM(
                model=model_name,
                tensor_parallel_size=kwargs.get("tp_degree", 1),
                dtype='bfloat16',
            )
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            return

        elif self.model_source == "vllm-online":
            self.client = openai.OpenAI(
                api_key="EMPTY",
                base_url=endpoint
            )
            return

        elif self.model_source == "ray":
            import ray, torch
            import pandas as pd
            from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
            from ray.data import DataContext
            cpus = multiprocessing.cpu_count()
            try:
                gpus = torch.cuda.device_count()
            except Exception as e:
                raise RuntimeError("No GPUs detected on CUDA. Make sure this task has access to properly configured GPUs.") from e
            if gpus == 0:
                raise RuntimeError("No GPUs detected on CUDA. Make sure this task has access to properly configured GPUs.")
            os.environ["NCCL_P2P_DISABLE"] = "1"
            ray.init(num_cpus=cpus, num_gpus=gpus, _temp_dir=(Path.home() / "ray_tmp").as_posix(), log_to_driver=False)
            DataContext.get_current().wait_for_min_actors_s = 1800

            config = vLLMEngineProcessorConfig(
                model_source=model_name,
                engine_resources={"CPU": max(1, cpus // max(1, gpus)), "GPU": 1},
                concurrency=max(1, gpus),
                engine_kwargs={
                    "tensor_parallel_size": 1,
                    "enable_chunked_prefill": True,
                    "max_model_len": 16384,
                    "max_num_batched_tokens": 65536,
                },
                max_concurrent_batches=32,
                batch_size=32,
            )
            self.client = build_llm_processor(
                config,
                # preprocess=lambda row: dict(
                #     messages=[{"role": "user", "content": row["prompt"]}] if type(row) != list else row,
                #     sampling_params=dict(truncate_prompt_tokens=16384 - 2048, max_tokens=2048), #TODO: context windows
                # ),
                postprocess= lambda row: {
                    "generated_text": row["generated_text"],
                    "id": row["id"],
                },
            )
            return


        from dotenv import load_dotenv
        try:
            load_dotenv()
        except:
            pass

        if self.model_source == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
            self.client = openai.OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
            )

        elif self.model_source == "azure":
            if get_model_vendor(model_name) != "openai":
                raise ValueError(f"Azure only supports OpenAI models, and {model_name} is not one.")
            from openai import AzureOpenAI
            if not os.getenv("AZURE_OPENAI_API_KEY"):
                raise EnvironmentError("AZURE_OPENAI_API_KEY environment variable not set.")
            if not os.getenv("AZURE_OPENAI_ENDPOINT"):
                raise EnvironmentError("AZURE_OPENAI_ENDPOINT environment variable not set.")
            self.client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-12-01-preview",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            )

        elif self.model_source == "google":
            if not os.getenv("GEMINI_API_KEY"):
                raise EnvironmentError("GEMINI_API_KEY environment variable not set.")
            self.client = openai.OpenAI(
                api_key=os.getenv("GEMINI_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )

        elif self.model_source == "anthropic":
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise EnvironmentError("ANTHROPIC_API_KEY environment variable not set.")
            self.client = openai.OpenAI(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                base_url="https://api.anthropic.com/v1/"
            )
        else:
            raise ValueError(f"Unsupported model vendor: {self.model_source}. Supported vendors are: openai, azure, google, anthropic, vllm, vllm-online, ray, test.")

    def get_chat_completion(self, messages, **kwargs):
        if self.model_source == "vllm-online" or self.model_source == "openai" or self.model_source == "azure" or self.model_source == "google":
            try:
                kwargs = self.to_OpenAI_format(kwargs)
                kwargs["model"] = self.model_name
                kwargs["messages"] = messages
                # API can't handle a ton of requests at once for some reason :(
                if "n" in kwargs and type(kwargs["n"]) == int and kwargs["n"] > 8:
                    with ThreadPoolExecutor() as executor:
                        futures = []
                        requests = kwargs["n"]
                        while requests > 0:
                            batch_size = min(requests, 8)
                            kwargs["n"] = batch_size
                            futures.append(executor.submit(self.client.chat.completions.create, **kwargs))
                            requests -= batch_size
                        all_results = [future.result() for future in futures]
                        res = all_results[0]
                        res.choices = sum([f.choices for f in all_results], [])
                    return res
                res = self.client.chat.completions.create(**kwargs)
                return res
            except openai.BadRequestError as e:
                if "'code': 'unknown_model'" in e.message:
                    provider_dict = {
                        "openai": ("OpenAI", "https://platform.openai.com/docs/models"),
                        "azure": ("Azure OpenAI", "the Azure AI foundry"),
                        "google": ("Google Gemini", "https://ai.dev"),
                        "anthropic": ("Anthropic", "https://docs.anthropic.com/en/docs/about-claude/models/overview"),
                    }
                    raise ValueError(f"\"{self.model_name}\" does not appear to be a valid {provider_dict[self.model_source][0]} model. Visit {provider_dict[self.model_source][1]} for a list of available models, or specify the model source manually. {'Make sure you have set up a deployment for the model in your Azure OpenAI resource, and that you are using the deployment name as the model name.' if self.model_source == 'azure' else ''}")
                else:
                    raise e
        elif self.model_source == "vllm":
            params = self.to_vLLM_format(kwargs)
            outputs = self.client.chat(
                messages,
                sampling_params=params,
                use_tqdm=False,
            )
            return self.from_vLLM_format(outputs[0]) # Only one concurrent response
        elif self.model_source == "test":
            # For testing purposes, return a dummy response
            n = kwargs.get("n", 1)
            import random
            return ChatCompletionResponse(
                choices=[
                    ChatCompletionChoice(
                        finish_reason="stop",
                        index=0,
                        message=ChatCompletionMessage(
                            role="assistant",
                            content="Hello I am jeff" if random.random() < 0.75 else ("bonjour je suis jeff" if random.random() < 0.5 else "大家好，我的名字是jeff")
                        )
                    )
                ] * n
            )
        else:
            raise NotImplementedError

    def get_response(self, messages, **kwargs):
        if type(messages) is str:
            messages = [{"role": "user", "content": messages}]
        res = self.get_chat_completion(messages, **kwargs)
        return res.choices[0].message.content

    def batch_completion(self, messages_list : list[Conversation], batch_name="default", **kwargs):
        if self.model_source == "azure":
            from .azure_batch_inference import batch_inference
            tags = kwargs.pop("job_tags", {})
            return batch_inference(self.model_name, messages_list, job_tags={"name": batch_name, "n": kwargs.get("n", 1), **tags}, **self.to_OpenAI_format(kwargs))
        elif self.model_source == "vllm":
            from vllm import LLM
            from vllm import SamplingParams
            params = SamplingParams(**self.to_vLLM_format(kwargs))
            outputs = self.client.chat(messages_list, sampling_params=params, use_tqdm=False) # Will be a list of responses now
            return [self.from_vLLM_format(out) for out in outputs]
        elif self.model_source == "vllm-online":
            with ThreadPoolExecutor() as executor:
                futures = []
                for messages in messages_list:
                    kwargs_copy = self.to_OpenAI_format(kwargs)
                    kwargs_copy["model"] = self.model_name
                    kwargs_copy["messages"] = messages
                    kwargs_copy["timeout"] = None
                    futures.append(executor.submit(self.client.chat.completions.create, **kwargs_copy))
                all_results = [future.result() for future in futures]
                return all_results
        elif self.model_source == "ray":
            print("running")
            import ray, torch
            messages_list = [messages for messages in messages_list if len(messages) > 0]
            # TODO: format params
            messages_list = [{"messages": messages, "sampling_params": {"max_tokens": 2500, "truncate_prompt_tokens": 7168}, "id": i} for i, messages in enumerate(messages_list)]

            ds = ray.data.from_items(messages_list).repartition(max(1, torch.cuda.device_count()) * 8)
            ds = self.client(ds).materialize()
            # TODO: best of n
            for row in ds.sort("id").iter_rows():
                yield ChatCompletionResponse(
                    choices=[
                        ChatCompletionChoice(
                            finish_reason="stop",
                            index=row["id"],
                            message=ChatCompletionMessage(
                                role="assistant",
                                content=row["generated_text"]
                            )
                        )
                    ]
                )
        elif self.model_source == "test":
            n = kwargs.get("n", 1)
            import random
            assert len(messages_list) > 0
            for message in messages_list:
                Conversation.model_validate({"messages": message})

                yield ChatCompletionResponse(
                    choices=[
                        ChatCompletionChoice(
                            finish_reason="stop",
                            index=0,
                            message=ChatCompletionMessage(
                                role="assistant",
                                content="Hello I am jeff" if random.random() < 0.75 else ("bonjour je suis jeff" if random.random() < 0.5 else "大家好，我的名字是jeff")
                            )
                        )
                    ] * n
                )
        else:
            raise NotImplementedError

    def to_OpenAI_format(self, kwargs):
        """
        Convert the kwargs to OpenAI format.
        """
        flags = ['messages', 'model', 'audio', 'frequency_penalty', 'function_call', 'functions', 'logit_bias', 'logprobs', 'max_completion_tokens', 'max_tokens', 'metadata', 'modalities', 'n', 'parallel_tool_calls', 'prediction', 'presence_penalty', 'reasoning_effort', 'response_format', 'seed', 'service_tier', 'stop', 'store', 'stream', 'stream_options', 'temperature', 'tool_choice', 'tools', 'top_logprobs', 'top_p', 'user', 'web_search_options', 'extra_headers', 'extra_query', 'extra_body', 'timeout']
        for kw in kwargs:
            if kw not in flags:
                if "extra_body" not in kwargs:
                    kwargs["extra_body"] = {}
                kwargs["extra_body"][kw] = kwargs[kw]
        return kwargs

    def to_vLLM_format(self, kwargs):
        from vllm import SamplingParams
        return SamplingParams(**kwargs)

    def from_vLLM_format(self, out):
        """
        Convert vLLM SamplingParams to OpenAI format.
        """

        choices = []
        for i, response in enumerate(out.outputs):
            message = ChatCompletionMessage(role="assistant", content=response.text)
            choice = ChatCompletionChoice(
                finish_reason="stop",
                index=i,
                message=message
            )
            choices.append(choice)

        return ChatCompletionResponse(choices=choices)