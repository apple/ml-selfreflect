#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import collections
import contextlib
import copy
import gc
import logging
import logging.config
import math
import os
import random
import re
import string
import sys
import numpy as np
import pynvml
import torch
import torch.distributed
import vllm
import vllm.distributed
import vllm.inputs
from typing import Any
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoConfig, AutoTokenizer  # type: ignore

logger = logging.getLogger(__name__)


def setup_logging(
    log_file: str = "app.log", debug_log_file: str = "debug.log", str_log_level: str = "DEBUG"
):
    # Set the root logger level based on log_level
    str_log_level = os.environ.get("LOGLEVEL", str_log_level).upper()
    int_log_level = getattr(logging, str_log_level, logging.INFO)

    logging.getLogger().setLevel(int_log_level)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console handler for INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    # File handler for INFO and above
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # File handler for DEBUG and above
    debug_fh = logging.FileHandler(debug_log_file)
    debug_fh.setLevel(logging.DEBUG)
    debug_fh.setFormatter(formatter)

    logging.basicConfig(
        handlers=(ch, fh, debug_fh),
        level=int_log_level,
    )


@contextlib.contextmanager
def set_logging_level(level, logger):
    # Access the specified logger (or root logger if not specified)
    old_level = logger.level  # Save the current level
    logger.setLevel(level)  # Set the new level

    try:
        yield  # Control returns to the block using the context manager
    finally:
        logger.setLevel(old_level)  # Restore the old level when exiting the context


def set_seeds(seed=1):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def cpu_concurrency() -> int:
    try:
        num_cpus = len(os.sched_getaffinity(0))  # type: ignore
    except AttributeError:
        logger.error(
            "Error in os.sched_getaffinity(0) in cpu_concurrency. Defaulting to cpu_concurrency=1"
        )
        return 1
    return max(1, num_cpus - torch.cuda.device_count() - 1)


def list_dict_to_dict_list(input_list: list[dict[str, Any]]) -> dict[str, list[Any]]:
    result = collections.defaultdict(list)

    for item in input_list:
        for key, value in item.items():
            result[key].append(value)

    return dict(result)


def average_inner_dicts(data):
    # Calculates the averages of a list of dictionaries
    sums = {}
    counts = {}
    for inner_dict in data:
        for key, value in inner_dict.items():
            if not isinstance(value, str):
                if key not in sums:
                    sums[key] = 0.0
                    counts[key] = 0
                if not np.isnan(value):
                    sums[key] += value
                    counts[key] += 1

    averages = {
        key: value / counts[key] if counts[key] > 0 else None for key, value in sums.items()
    }

    return averages


def gpu_batch_size() -> int:
    return 32


class ChatTokenizer:
    """
    A wrapper for huggingface tokenizers. It takes care of
    stitching together user and assistant prompts and automatically
    applies chat templates if necessary.
    """

    def __init__(self, model_name: str, use_chat_template: bool, enable_thinking: bool):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.use_chat_template = use_chat_template
        self.enable_thinking = enable_thinking

        if enable_thinking and not use_chat_template:
            raise ValueError("enable_thinking requires use_chat_template to be True.")

    def __call__(
        self,
        user_prompts: list[str],
        assistant_prompts: list[str] | None = None,
        add_generation_prompt: bool = False,
        system: str = "You are a friendly chatbot who always responds with short and concise answers.\n",
        continue_final_message: bool = False,
        return_tensors: None | str = "pt",
    ):
        """
        Stitches together a user - assistant chat, applies chat template if necessary, and returns
        the tokenized sequence
        :param user_prompts: List of strings, messages by the user
        :param assistant_prompts: List of strings, messages by the assistant. Should be same length
          as the user or one less
        :return: torch tensor of integers, the token IDs
        """
        if system is None:
            system = self.system_prompt
        if not isinstance(user_prompts, list):
            user_prompts = [user_prompts]
        if assistant_prompts is None:
            assistant_prompts = []
        if not isinstance(assistant_prompts, list):
            assistant_prompts = [assistant_prompts]

        # Stitch the prompts of user and assistant together in a back and forth
        messages = []
        if system is not None:
            if "gemma" in self.model_name.lower():
                # Gemma has no system prompt, needs to be added to user prompt
                # https://github.com/abetlen/llama-cpp-python/issues/1580
                user_prompts = copy.deepcopy(user_prompts)  # Modifying inplace requires copy
                user_prompts[0] = "\n".join((system, user_prompts[0]))
            else:
                messages.append({"role": "system", "content": system})
        for i in range(len(user_prompts)):
            if user_prompts[i] is not None:
                messages.append({"role": "user", "content": user_prompts[i]})
            if len(assistant_prompts) > i and assistant_prompts[i] is not None:
                messages.append({"role": "assistant", "content": assistant_prompts[i]})

        # Apply chat template if necessary
        if self.use_chat_template:
            text_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
                enable_thinking=self.enable_thinking,
            )
            # We need to stop the reasoning for the reasoning-models for which you enable_thinking=False is not implemented
            if not self.enable_thinking and self.model_name in (
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                "Qwen/QwQ-32B",
            ):
                text_input += "</think>\n\n"
        else:
            text_input = "\n".join([msg["content"] for msg in messages])

        # Tokenize
        if return_tensors is not None:
            token_input = self.tokenizer(text_input, return_tensors=return_tensors)
        else:
            token_input = self.tokenizer.encode(text_input)

        return token_input


class TFIDF:
    def __init__(self):
        # Load a default corpus (20 Newsgroups dataset)
        newsgroups = fetch_20newsgroups(
            subset="all",
            remove=("headers", "footers", "quotes"),
            data_home="/mnt/tmp",
            download_if_missing=False,
        )
        corpus = newsgroups.data  # type: ignore

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        feature_names = np.array(vectorizer.get_feature_names_out())

        # Compute average TF-IDF score per word
        tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()  # type: ignore
        self.word_tfidf = dict(zip(feature_names, tfidf_scores, strict=False))

    def __call__(self, word_list):
        """
        Sorts words by the rarest word first, example:
        (TFIDF()(["christ", "you", "acidic", "telekfngieffh"]))
        > [2 0 1 3]
        Note that "telekfngieffh" is listed at the end,
        because it is not part of the vocabulary, so it's considered rare.
        """
        return np.argsort(self.get_tfidf_score(word_list))

    def _preprocess(self, word_list):
        return [word.strip(string.whitespace + string.punctuation).lower() for word in word_list]

    def get_tfidf_score(self, word_list):
        # For words in the vocabulary: Great, give their TF-IDF score
        # For words not in the vocabulary: Wow, probably super rare. Mark as rarerest possible.
        return [self.word_tfidf.get(word, -1) for word in self._preprocess(word_list)]

    def is_stopword(self, word_list, threshold=0.012):
        """
        Expects a list of words as input, outputs a list of booleans
        threshold is tuned manually to roughly exclude stopwords, but no important words
        """
        tfidf_scores = self.get_tfidf_score(word_list)
        return [s > threshold for s in tfidf_scores]


def get_vllm_kwargs(model_name: str, args) -> dict[str, Any]:
    output_kwargs: dict[str, Any] = {}

    output_kwargs["enable_prefix_caching"] = True

    output_kwargs["max_model_len"] = 32 * 1024  # 32K
    if any(model_name.startswith(k) for k in ("microsoft/phi-4",)):
        output_kwargs["max_model_len"] = 16 * 1024  # 16K since that's its context length

    # for multiomodal models, we're not inputting any images
    if any(
        model_name.startswith(k)
        for k in (
            "google/gemma-3",
            "meta-llama/Llama-4",
        )
    ) and not model_name.startswith("google/gemma-3-1b"):
        output_kwargs["limit_mm_per_prompt"] = dict(image=0)

    if args.seed is not None:
        output_kwargs["seed"] = args.seed

    if any(
        model_name.startswith(k)
        for k in (
            "mistralai/Mistral-7B-Instruct-v0.1",
            "mistralai/Mistral-7B-v0.1",
            "mistralai/Ministral-8B-Instruct-2410",
            "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
            "mistralai/Mistral-Large-Instruct-2411",
        )
    ):
        output_kwargs["disable_sliding_window"] = True
        output_kwargs["enable_prefix_caching"] = False

    if args.debug_mode:
        output_kwargs["enforce_eager"] = True

    return output_kwargs


def get_gpus_distribution(model_name: str) -> tuple[int, int]:
    """
    Returns how many GPUs are required to fit a model in tensor-parralel (TP),
    and how many model instances can be run in data-parallel (DP).
    """
    gpus_to_fit_model = get_num_of_gpus_to_fit_model(model_name)

    gpu_concurrency = math.floor(torch.cuda.device_count() / gpus_to_fit_model)

    return gpus_to_fit_model, gpu_concurrency


def get_num_of_gpus_to_fit_model(
    model_name: str,
    # cant extract data info from model name alone: assuming everything is FP16 -> 2 bytes
    data_type_size: int = 2,
    # assume the model actually allocates (safety_factor * model_size) memory, for overhead
    safety_factor: float = 1.5,
) -> int:
    # Modify this manually.
    gpus_per_model_registry = {
        # Falcon
        "tiiuae/falcon-7b-instruct": 1,
        "tiiuae/falcon-7b": 1,
        "tiiuae/falcon-40b-instruct": 2,
        "tiiuae/falcon-40b": 2,
        # gemma-2
        "google/gemma-2-9b-it": 1,
        "google/gemma-2-9b": 1,
        "google/gemma-2-27b-it": 2,
        "google/gemma-2-27b": 2,
        # gemma-3
        "google/gemma-3-1b-pt": 1,
        "google/gemma-3-1b-it": 1,
        "google/gemma-3-4b-pt": 1,
        "google/gemma-3-4b-it": 1,
        "google/gemma-3-12b-pt": 2,
        "google/gemma-3-12b-it": 2,
        "google/gemma-3-27b-pt": 4,
        "google/gemma-3-27b-it": 4,
        # Qwen2.5
        "Qwen/Qwen2.5-0.5B-Instruct": 1,
        "Qwen/Qwen2.5-0.5B": 1,
        "Qwen/Qwen2.5-7B-Instruct": 1,
        "Qwen/Qwen2.5-7B": 1,
        "Qwen/Qwen2.5-14B": 2,
        "Qwen/Qwen2.5-14B-Instruct": 2,
        "Qwen/Qwen2.5-32B-Instruct": 2,
        "Qwen/Qwen2.5-32B": 2,
        "Qwen/Qwen2.5-72B-Instruct": 4,
        "Qwen/Qwen2.5-72B": 4,
        # Qwen3
        "Qwen/Qwen3-0.6B": 1,
        "Qwen/Qwen3-0.6B-Base": 1,
        "Qwen/Qwen3-1.7B": 1,
        "Qwen/Qwen3-1.7B-Base": 1,
        "Qwen/Qwen3-4B": 1,
        "Qwen/Qwen3-4B-Base": 1,
        "Qwen/Qwen3-8B": 1,
        "Qwen/Qwen3-8B-Base": 1,
        "Qwen/Qwen3-14B": 2,
        "Qwen/Qwen3-14B-Base": 2,
        "Qwen/Qwen3-32B": 2,
        "Qwen/Qwen3-32B-Base": 2,
        # Mistral
        "mistralai/Mistral-7B-Instruct-v0.1": 1,
        "mistralai/Mistral-7B-v0.1": 1,
        "mistralai/Mistral-7B-Instruct-v0.3": 1,
        "mistralai/Mistral-7B-v0.3": 1,
        "mistralai/Mistral-Small-24B-Base-2501": 2,
        "mistralai/Mistral-Small-24B-Instruct-2501": 2,
        "mistralai/Mistral-Small-3.1-24B-Instruct-2503": 2,
        "mistralai/Ministral-8B-Instruct-2410": 1,
        "mistralai/Mistral-Large-Instruct-2411": 8,
        # llama
        "meta-llama/Llama-3.1-8B-Instruct": 1,
        "meta-llama/Llama-3.1-70B-Instruct": 4,
        "meta-llama/Llama-3.3-70B-Instruct": 4,
        "meta-llama/Llama-4-Scout-17B-16E-Instruct": 4,
        # Phi4
        "microsoft/phi-4": 2,
        # Reasoning models
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": 4,
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": 8,
        "Qwen/QwQ-32B": 4,
    }

    if model_name in gpus_per_model_registry:
        gpus_per_model = gpus_per_model_registry[model_name]
    else:
        logger.info("Guesstimating number of GPUs necessary to hold model")
        model_size = extract_model_size(model_name) * data_type_size / 1024**2  # in mb
        total_mem_necessary = model_size * safety_factor
        mem_per_device = get_available_devices_memory()
        gpus_per_model = math.ceil(
            total_mem_necessary / mem_per_device[0]
        )  # assume all GPUs are same

    # make sure that gpus_per_model divides num_attention_heads exactly,
    # otherwise vLLM will complain...
    cfg = AutoConfig.from_pretrained(model_name)
    if hasattr(cfg, "num_attention_heads"):
        num_heads = cfg.num_attention_heads
    elif hasattr(cfg, "text_config") and hasattr(cfg.text_config, "num_attention_heads"):
        num_heads = cfg.text_config.num_attention_heads
    else:
        num_heads = gpus_per_model
    while gpus_per_model <= min(torch.cuda.device_count(), num_heads):
        if num_heads % gpus_per_model == 0:
            break
        else:
            gpus_per_model = gpus_per_model + 1

    assert gpus_per_model <= torch.cuda.device_count(), (
        "Not enough GPUs(' memory) to fit this model"
    )

    return gpus_per_model


@contextlib.contextmanager
def nvml_context():
    pynvml.nvmlInit()
    try:
        yield
    finally:
        pynvml.nvmlShutdown()


def get_available_devices_memory() -> list[float]:
    assert torch.cuda.is_available(), "No GPUs available"

    gpu_info = []

    with nvml_context():
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices:
            indices = [int(idx.strip()) for idx in visible_devices.split(",")]
        else:
            indices = list(range(pynvml.nvmlDeviceGetCount()))

        for i in indices:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mb = mem_info.total / (1024**2)
            gpu_info.append(total_mb)
    return gpu_info


def extract_model_size(model_name: str) -> float | int:
    # This regex looks for a number (optionally with a decimal)
    # followed by optional spaces or dashes and then the letter B or M.
    pattern = re.compile(r"(\d+(?:\.\d+)?)[\s\- _]*([bm])", re.IGNORECASE)
    match = pattern.search(model_name)
    if match:
        number_str, unit = match.groups()
        number = float(number_str)
        unit = unit.lower()
        if unit == "b":
            number *= 1e9
        elif unit == "m":
            number *= 1e6
    else:
        raise ValueError(f"Model name format not recognised from model_name={model_name}")

    return int(number) if number.is_integer() else number


def dict_list_to_list_dict(input_dict: dict[str, list[str]]) -> list[dict[str, str]]:
    keys = list(input_dict.keys())
    values = zip(*input_dict.values(), strict=False)

    result = [dict(zip(keys, value_set, strict=False)) for value_set in values]
    return result


class VLLMBaseClass:
    def __del__(self):
        # https://github.com/vllm-project/vllm/issues/1908#issuecomment-2461174904
        vllm.distributed.parallel_state.destroy_model_parallel()
        vllm.distributed.parallel_state.destroy_distributed_environment()
        del self.llm  # type: ignore
        clean_up_torch()


def clean_up_torch():
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()


def format_mmlu_question(subject, question, answers):
    MMLU_CHOICES = ["A", "B", "C", "D"]
    prompt = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
    prompt += f"{question}"
    for j, answer in enumerate(answers):
        prompt += f"\n{MMLU_CHOICES[j]}. {answer}"
    prompt += "\nAnswer:"

    return prompt


class MockVLLMOutput:
    def __init__(self, string: str):
        assert isinstance(string, str)
        self.text: str = string


class MockVLLMRequestOutput:
    def __init__(self, strings: list[str]):
        assert isinstance(strings, list)
        self.outputs: list[MockVLLMOutput] = [MockVLLMOutput(string) for string in strings]


class VLLMGenerator(VLLMBaseClass):
    def __init__(
        self,
        model_name,
        tensor_parallel_size: int = 1,
        args=None,
        ray_step_name=None,
    ):
        # for chat template, https://github.com/vllm-project/vllm/issues/6416
        logger.info(f"HF_TOKEN: {os.getenv('HF_TOKEN', None)}")
        self.vllm_kwargs = get_vllm_kwargs(model_name, args)
        self.llm = vllm.LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            **self.vllm_kwargs,
        )
        self.ray_step_name = ray_step_name if ray_step_name else "VLLMGenerator"

    def __repr__(self):
        return self.ray_step_name

    def __call__(
        self,
        batch: dict[str, list[Any]],
        output_key_sampling_params_dict: dict[str, vllm.SamplingParams],
    ) -> dict[str, list[Any]]:
        for (
            output_key,
            sampling_params_instance,
        ) in output_key_sampling_params_dict.items():
            batch = self.helper_call(batch, sampling_params_instance, output_key)

        return batch

    def helper_call(
        self,
        batch: dict[str, Any],
        sampling_params: vllm.SamplingParams,
        output_key: str,
    ) -> dict[str, Any]:
        string_already_computed = output_key == "concatenated_answers_summary"
        if string_already_computed:
            llm_outputs = [MockVLLMRequestOutput([text]) for text in batch["token_input"]]
        else:
            batch["token_input"] = [x.squeeze().tolist() for x in batch["token_input"]]

            # If the prompt is longer than the models context length, truncate it, so that it doesn't cause an error.
            max_input_length = self.vllm_kwargs["max_model_len"] - sampling_params.max_tokens - 1
            batch["token_input"] = [
                (
                    tokens_list
                    if len(tokens_list) <= max_input_length
                    else tokens_list[-max_input_length:]
                )
                for tokens_list in batch["token_input"]
            ]

            llm_outputs = self.llm.generate(
                [vllm.inputs.TokensPrompt(prompt_token_ids=x) for x in batch["token_input"]],
                sampling_params,
            )

        batch_t = dict_list_to_list_dict(batch)

        def map_func(
            item: dict[str, Any], llm_output: vllm.RequestOutput | MockVLLMRequestOutput
        ) -> dict[str, Any]:
            # Convert vllm.RequestOutput objects into python-native dicts
            vllm_outputs_dicts = [vllm_outputs.__dict__ for vllm_outputs in llm_output.outputs]

            output = [elem["text"] for elem in vllm_outputs_dicts]
            if len(output) == 1:
                # Only a single output, i.e. a summary, was generated,
                # so we don't pass it back as a list
                output = output[0]
            return {
                output_key: output,
                **item,
            }

        outputs_t = list(map(map_func, batch_t, llm_outputs))
        outputs = list_dict_to_dict_list(outputs_t)

        return outputs

    def remove_chat_template(self, output, empty_answer):
        # Empty answer starts the same way as output (with user message etc)
        # and ends the same way (with assistant tokens). Remove both of them.

        output = output.strip()
        empty_answer = empty_answer.strip()

        # Remove common prefix
        prefix_len = 0
        for a, b in zip(output, empty_answer, strict=False):
            if a == b:
                prefix_len += 1
            else:
                break
        output = output[prefix_len:]

        # Remove common suffix
        suffix_len = 0
        for a, b in zip(output[::-1], empty_answer[::-1], strict=False):
            if a == b:
                suffix_len += 1
            else:
                break
        if suffix_len:
            output = output[:-suffix_len]

        return output


def generate(model_name, ray_dataset, sampling_params, args, ray_step_name):
    """
    Assumes data to have "token_input" entry, writes out into data in sampling_params keys key
    -> so make sure to only pass one method as key to sampling_params at a time to generate
    """
    if args.debug_mode:
        gpus_per_model, gpu_concurrency = 1, 1
    else:
        gpus_per_model, gpu_concurrency = get_gpus_distribution(args.model_name)
    logger.info(
        f"args.model_name={args.model_name}: gpus_per_model={gpus_per_model}, gpu_concurrency={gpu_concurrency}"
    )

    fn_kwargs = dict(output_key_sampling_params_dict=sampling_params)
    fn_constructor_kwargs = dict(
        model_name=model_name,
        tensor_parallel_size=gpus_per_model,
        args=args,
        ray_step_name=ray_step_name,
    )

    if args.debug_mode:
        extractor = VLLMGenerator(**fn_constructor_kwargs)
        batch = ray_dataset.take_batch(batch_size=2)
        batch = extractor(batch, **fn_kwargs)
        raise Exception("Converting this batch in debug_mode=True doesn't work.")
    else:
        vllm_port = 29501

        def ray_remote_args_fn():
            nonlocal vllm_port
            ray_remote_args = dict(
                runtime_env=dict(env_vars={"VLLM_PORT": str(vllm_port := vllm_port + 5)})
            )
            return ray_remote_args

        ray_dataset = ray_dataset.map_batches(
            VLLMGenerator,
            concurrency=gpu_concurrency,
            # the number of GPUs per LLM, where number of LLMs is concurrency value
            num_gpus=gpus_per_model,
            batch_size=gpu_batch_size(),
            fn_kwargs=fn_kwargs,
            fn_constructor_kwargs=fn_constructor_kwargs,
            ray_remote_args_fn=ray_remote_args_fn,
        )
        # ray_dataset = ray_change_token_input_schema_to_list_int(ray_dataset)
        ray_dataset = ray_dataset.drop_columns(["token_input"])
        ray_dataset = ray_dataset.materialize()

    return ray_dataset
