#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import functools
import json
import logging
import re
import time
from typing import Any

import numpy as np
import ray
import vllm
import wandb

from src.selfreflect.args import parse_args
from src.selfreflect.utils import (
    ChatTokenizer,
    cpu_concurrency,
    generate,
    set_logging_level,
    set_seeds,
    setup_logging,
)

setup_logging()
logger = logging.getLogger(__name__)

DEFAULT_SAMPLING_PARAMS: dict[str, bool | float | int | str | None] = dict(
    spaces_between_special_tokens=False,
    skip_special_tokens=True,
    include_stop_str_in_output=False
)


class SummaryTokenizedPromptGenerator:
    """
    Generates prompts for various summaries for questions and answers.
    """

    def __init__(
        self,
        model_name: str,
        use_chat_template: bool,
        enable_thinking: bool,
        add_generation_prompt: bool = True,
    ):
        """
        :param model_name: string of the model name.
        """
        self.tokenizer = ChatTokenizer(model_name, use_chat_template, enable_thinking)
        self.add_generation_prompt = add_generation_prompt

    def __call__(self, method, question, answers=None):
        cleartext_prompt = None
        # Switch on strategy
        if method == "greedy":
            tokens = self.greedy_summary(question)
        elif method == "basic":
            tokens = self.basic_summary(question)
        elif method == "sample_and_summarize":
            if answers is not None:
                tokens = self.sample_and_summarize_summary(question, answers)
            else:
                raise RuntimeError(
                    f"Error SummaryTokenizedPromptGenerator method={method}: answers={answers}"
                )
        elif method == "concatenated_answers":
            if answers is not None:
                tokens = self.concatenated_answers_summary(answers)
            else:
                raise RuntimeError(
                    f"Error SummaryTokenizedPromptGenerator method={method}: answers={answers}"
                )
        elif method == "cot":
            tokens = self.cot_summary(question)
        elif method == "sample_and_summarize_generation":
            tokens = self.greedy_summary(question, add_empty_answer=False)
        elif method == "cot_analysis":
            tokens, cleartext_prompt = self.generate_cot_analysis(question)
        elif method == "twostep_cot":
            tokens = self.generate_cot_summary_prompt(answers)
        else:
            logger.error(f"Unknown summary strategy: {method}. Skipping.")

        # Stitch together tokenized prompt and optionally cleartext_prompt
        output = {"token_input": tokens}
        if cleartext_prompt is not None:
            output[f"{method}_prompt"] = cleartext_prompt

        return output

    def generate_cot_analysis(self, question: str):
        """
        CoT Step 1: Generate prompt for analysis.
        """
        user_prompt = (
            f"{question}\nBrainstorm all plausible answers and list each potential answer.\n"
        )
        # Use the ChatTokenizer to apply the chat template
        tokenized_prompt = self.tokenizer(
            user_prompts=user_prompt, add_generation_prompt=self.add_generation_prompt
        )
        return tokenized_prompt["input_ids"], user_prompt

    def generate_cot_summary_prompt(self, context):
        """
        CoT Step 2: Generate prompt for summary using analysis.
        """
        user_prompt = (
            "Now distill these insights into a summary. Mention all details and possibilities, but make it one coherent answer.\n"
            "Return only that summary. Write it like a response to the question, do not start with 'The analysis includes...' but start it like you would start a response to the question."
        )
        # Use the ChatTokenizer to apply the chat template
        tokenized_prompt = self.tokenizer(
            user_prompts=[context[0], user_prompt],
            assistant_prompts=context[1],
            add_generation_prompt=self.add_generation_prompt,
        )
        return tokenized_prompt["input_ids"]

    # -------------------------------------------------------------------------
    # INDIVIDUAL SUMMARY METHODS
    # -------------------------------------------------------------------------

    # Bottom Baseline: Don't ask the model to summarize, just take the greedy answer to the
    # question as summary
    def greedy_summary(self, question: str, add_empty_answer=False):
        """
        1) Greedy approach: simply produce a direct answer without special instructions.
        """
        user_prompt = f"{question}"
        tokenized_prompt = self.tokenizer(
            user_prompts=user_prompt, add_generation_prompt=self.add_generation_prompt
        )
        if add_empty_answer:
            empty_answer = self.tokenizer(
                user_prompts=user_prompt, assistant_prompts=[""], add_generation_prompt=False
            )
            empty_answer = self.tokenizer.tokenizer.batch_decode(empty_answer["input_ids"])[0]
            return [tokenized_prompt["input_ids"], empty_answer]
        else:
            return tokenized_prompt["input_ids"]

    # Ask the model to "give a summary of answers" without providing the answers or CoT instructions
    def basic_summary(self, question: str) -> str:
        """
        2) Basic approach: ask for a concise summary of possible answers, no CoT.
        """
        user_prompt = f"""Please respond to the following question '{question}'.

Your goal is to summarize all possible answers to this question:
* If there are multiple possible answers, the summarized answer should mention the main possible answers. However, you do not have to list possibilities that are too unlikely.
* If some possibilities are more likely than others, delineate which possibilities are more more likely by using words like "most likely" and "could also be".
* The format of the summarized answer should be the same as a normal answer.
* If there is only clear answer to the question, just provide that answer, without hedging across possibilities.

Please provide the summarized answer."""
        tokenized_prompt = self.tokenizer(
            user_prompts=user_prompt, add_generation_prompt=self.add_generation_prompt
        )
        return tokenized_prompt["input_ids"]

    # Top Baseline: Provide the answer distribution and ask the model to summarize it
    def sample_and_summarize_summary(self, question: str, answers: list[str]) -> str:
        """
        Given answers sampled from the model, instructs to summarize them.
        """
        n_answers = len(answers)
        answers = "\n".join([f"x_{str(i + 1)} = '{a}'" for i, a in enumerate(answers)])
        user_prompt = f"""Below, you are given {n_answers} individual answers to the question '{question}' that were given by different people.

Your goal is to summarize the {n_answers} answers into one concise summarized answer.
* The summarized answer should mention the main possibility mentioned in the {n_answers} answers, i.e. the one that is mentioned most often.
* The summarized answer should also mention further possibilites mentioned in the {n_answers}.
* If some possibilities are mentioned more often than others, delineate which possibilities are more often found in the others by using words like "most likely" and "could also be".
* If individual answers differ in numeric values, aggregate them into ranges. For example, if two answers say '20', one answer says '22' and another one '24', a good summary would be to give a range from 20 to 24.
* The format of the summarized answer should be the same as each individual answer. Provide only the answer, as if it were part of the {n_answers} answers, without statements like "The answers include...".
* The summarized answer should not use synonyms for the original wording used in individual answers.
* The summarized answer should reflect what the {n_answers} answers deem possible. They can contain factually wrong options. Do not correct those, just report the possibilities as they are given in the answers.

Here are the {n_answers} answers:
{answers}

Please provide the summarized answer."""
        tokenized_prompt = self.tokenizer(
            user_prompts=user_prompt, add_generation_prompt=self.add_generation_prompt
        )
        return tokenized_prompt["input_ids"]

    def concatenated_answers_summary(self, answers: list[str]) -> str:
        """
        Simply concatenates some sampled answers as a baseline.
        """
        summary_string = "\n".join([f"x_{str(i + 1)} = '{a}'" for i, a in enumerate(answers)])

        return summary_string

    def cot_summary(self, question: str) -> str:
        """
        Generates a CoT and then a summary
        """
        user_prompt = f"""
Please respond to the following question: '{question}'.

Your goal is to **reason about multiple plausible answers** to this question, not just provide a single answer (unless there is only one possible answer):
* First, reflect explicitly on whether there are multiple reasonable answers to this question. Consider ambiguity, interpretation, and context.
* Then, explore at least 2–3 **distinct answer possibilities**, especially if the question allows it.
* Clearly indicate if some possibilities are more likely than others using words like "most likely", "could also be", or "another possible interpretation is".
* You do not need to include very unlikely answers.
* After reasoning, **summarize the best answer(s)** in a way that reflects the uncertainty or multiplicity, while still sounding natural.
* If there is only one clearly correct answer, explain why it is the only plausible one, and just give that.

Use the following format:
Reasoning: [Detailed reasoning about the different possible answers and their likelihoods.]
Summary: [Stand-alone answer reflecting the reasoning.]

Make sure to show multiple possibilities if relevant before committing to the final summary.
"""
        tokenized_prompt = self.tokenizer(
            user_prompts=user_prompt, add_generation_prompt=self.add_generation_prompt
        )
        return tokenized_prompt["input_ids"]


class PromptApplier:
    def __init__(
        self,
        model_name: str,
        method: str,
        use_chat_template: bool,
        enable_thinking: bool,
    ):
        self.summary_prompt_creator = SummaryTokenizedPromptGenerator(
            model_name, use_chat_template, enable_thinking
        )
        self.method = method

    def __call__(self, item: dict[str, Any]) -> dict[str, Any]:
        question = item["question"]
        if self.method in ("sample_and_summarize", "concatenated_answers"):
            answers = item["conditioning_answers_sampled"]
        elif self.method == "twostep_cot":
            answers = [item["cot_analysis_prompt"], item["cot_analysis_result"]]
        else:
            answers = None
        item.update(self.summary_prompt_creator(self.method, question, answers))

        return item


def iterate_over_ray_dataset(
    model_name: str,
    use_chat_template: bool,
    ray_dataset,
    args,
    method=None,
    cls: type = PromptApplier,
    enable_thinking: bool = False,
):
    fn_constructor_kwargs = dict(
        model_name=model_name,
        use_chat_template=use_chat_template,
        enable_thinking=enable_thinking,
    )
    if method is not None:
        fn_constructor_kwargs["method"] = method

    if args.debug_mode:
        extractor = cls(**fn_constructor_kwargs)
        dataset = ray_dataset.take(limit=2)
        dataset = [extractor(item) for item in dataset]
        with set_logging_level(logging.INFO, logger):
            ray_dataset = ray.data.from_items(dataset)
    else:
        ray_dataset = ray_dataset.map(
            cls,
            concurrency=max(1, cpu_concurrency() // 2),
            fn_constructor_kwargs=fn_constructor_kwargs,
        )

    return ray_dataset


def cot_chain_remover(item: dict[str, Any], method) -> dict[str, Any]:
    match = re.search(r"(?i)summary:\s*(.*)", item[f"{method}_summary"])
    if match:
        item[f"{method}_summary"] = match.group(1)
    return item


def remove_cot_chain(ray_dataset, args, method):
    if args.debug_mode:
        dataset = ray_dataset.take(limit=2)
        dataset = [cot_chain_remover(item=item, method=method) for item in dataset]
        with set_logging_level(logging.INFO, logger):
            ray_dataset = ray.data.from_items(dataset)
    else:
        ray_dataset = ray_dataset.map(
            functools.partial(cot_chain_remover, method=method),
            # for state-less functions concurrency is set automatically
        )

    return ray_dataset


def reasoning_remover(item: dict[str, Any], method) -> dict[str, Any]:
    splited_text = item[f"{method}_summary"].split("</think>\n\n")
    if len(splited_text) > 2:
        print("Error: Multiple </think>")
    elif len(splited_text) == 2:
        item[f"{method}_summary"] = splited_text[1]
    else:
        pass  # do nothing
    return item


def remove_reasoning_trace(ray_dataset, args, method):
    if args.debug_mode:
        dataset = ray_dataset.take(limit=2)
        dataset = [reasoning_remover(item=item, method=method) for item in dataset]
        with set_logging_level(logging.INFO, logger):
            ray_dataset = ray.data.from_items(dataset)
    else:
        ray_dataset = ray_dataset.map(
            functools.partial(reasoning_remover, method=method),
            # for state-less functions concurrency is set automatically
        )

    return ray_dataset


def dataset_cleaner(item: dict[str, Any]) -> dict[str, Any]:
    new_dict = {}
    new_dict["question"] = item["question"]
    new_dict["summaries"] = {key: value for key, value in item.items() if key.endswith("_summary")}

    return new_dict


def remove_everything_except_summaries(ray_dataset, args):
    if args.debug_mode:
        dataset = ray_dataset.take(limit=2)
        dataset = [dataset_cleaner(item) for item in dataset]
        with set_logging_level(logging.INFO, logger):
            ray_dataset = ray.data.from_items(dataset)
    else:
        ray_dataset = ray_dataset.map(
            dataset_cleaner,
            # for state-less functions concurrency is set automatically
        )

    return ray_dataset


def get_summary_sampling_params(method, args):
    default_sampling_params = DEFAULT_SAMPLING_PARAMS.copy()
    if not args.use_chat_template:
        default_sampling_params["stop"] = [". "]

    if method in {
        "greedy",
        "basic",
        "sample_and_summarize",
        "cot",
        "concatenated_answers",
        "cot_analysis_generation",
        "twostep_cot",
    }:
        tag = f"{method}_summary" if method != "cot_analysis_generation" else "cot_analysis_result"
        sampling_params = vllm.SamplingParams(
            max_tokens=args.model_max_new_tokens,
            n=1,
            temperature=0.0,
            **default_sampling_params,
        )

        if args.enable_thinking_for_summaries:
            sampling_params = vllm.SamplingParams(
                max_tokens=args.model_max_new_tokens,
                n=1,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0.0,
                **default_sampling_params,
            )
    elif method == "sample_and_summarize_generation":
        tag = "conditioning_answers_sampled"
        sampling_params = vllm.SamplingParams(
            max_tokens=args.model_max_new_tokens,
            n=args.num_conditioning_answers,
            temperature=args.conditioning_generation_temperature,
            **default_sampling_params,
        )
    return {tag: sampling_params}


def generate_summaries(data, args):
    """
    Iterates over all questions in data, applies all chosen summary strategies for a chosen model.

    :param data: a dataset read from json that can be converted to ray dataset
    :param args: arguments of the task
    :return: A list of dicts containing question, answers, and the generated summaries.
    """
    # Dictionary to track timing of each summary generation method
    summary_timings = {}

    ray_dataset = ray.data.from_items(data)

    # Preprocessing: Generate conditioning answers for methods sample_and_summarize
    if ("sample_and_summarize" in args.summary_strats) or (
        "concatenated_answers" in args.summary_strats
    ):
        logger.info(
            f"Generating conditioning answers with model {args.model_name} for sample_and_summarize and concatenated_answers summary strategy."
        )
        start_time = time.time()

        ray_dataset = iterate_over_ray_dataset(
            args.model_name,
            args.use_chat_template_summary_model,
            ray_dataset,
            args,
            "sample_and_summarize_generation",
            PromptApplier,
            enable_thinking=False,
        )
        sampling_param_dict = get_summary_sampling_params("sample_and_summarize_generation", args)
        ray_dataset = generate(
            args.model_name,
            ray_dataset,
            sampling_param_dict,
            args,
            ray_step_name="generate_summaries:sampling_conditioning_answers",
        )

        summary_timings["sample_and_summarize"] = time.time() - start_time
        summary_timings["concatenated_answers"] = time.time() - start_time
        wandb.log({"summary_time/sample_and_summarize": summary_timings["sample_and_summarize"]})
        wandb.log({"summary_time/concatenated_answers": summary_timings["concatenated_answers"]})

    if "twostep_cot" in args.summary_strats:
        logger.info("Twostep CoT Step 1: Generating analysis.")
        start_time = time.time()

        ray_dataset = iterate_over_ray_dataset(
            args.summary_model_name,
            args.use_chat_template_summary_model,
            ray_dataset,
            args,
            "cot_analysis",
            PromptApplier,
        )

        sampling_param_dict = get_summary_sampling_params("cot_analysis_generation", args)
        ray_dataset = generate(
            args.summary_model_name,
            ray_dataset,
            sampling_param_dict,
            args,
            ray_step_name="generate_summaries:twostep_cot",
        )

        summary_timings["twostep_cot"] = time.time() - start_time
        wandb.log({"summary_time/twostep_cot": summary_timings["twostep_cot"]})

    # Now we can actually generate the summaries
    for method in args.summary_strats:
        enable_thinking_for_this_summary = args.enable_thinking_for_summaries and method in (
            "basic",
            "sample_and_summarize",
        )

        logger.info(
            f"Generating summaries with model {args.summary_model_name} using strategy='{method}'."
        )
        start_time = time.time()

        # Generate "token_input" key with summary prompt according to method before generating (CPU)
        ray_dataset = iterate_over_ray_dataset(
            args.summary_model_name,
            args.use_chat_template_summary_model,
            ray_dataset,
            args,
            method,
            PromptApplier,
            enable_thinking=enable_thinking_for_this_summary,
        )

        # Generate summary (GPU)
        sampling_param_dict = get_summary_sampling_params(method, args)
        ray_dataset = generate(
            args.summary_model_name,
            ray_dataset,
            sampling_param_dict,
            args,
            ray_step_name=f"generate_summaries:{method}",
        )

        # Postprocessing, like cleaning up reasoning chains (CPU)
        # If you want to play around with cot and retain the Reasoning chain,
        # comment out the following lines. Then the chain will be part of the output.
        if method in ("cot"):
            ray_dataset = remove_cot_chain(ray_dataset, args, method)

        if enable_thinking_for_this_summary:
            ray_dataset = remove_reasoning_trace(ray_dataset, args, method)

        # Record and log the time taken for this summary method
        if method not in summary_timings:
            summary_timings[method] = 0.0
        summary_timings[method] += time.time() - start_time
        wandb.log({f"summary_time/{method}": summary_timings[method]})

    # Drop method-specific fields
    ray_dataset = remove_everything_except_summaries(ray_dataset, args)

    updated_data = ray_dataset.take_all()
    data = updated_data

    # Log summary of all timings
    wandb.log({"summary_time/total": sum(summary_timings.values())})
    logger.info(f"Summary generation timings: {summary_timings}")

    return data


def main(args):
    set_seeds(args.seed)

    # Load the dataset
    with open(args.questions_file) as file:
        data = json.load(file)

    # Generate summaries
    logger.info("Generating summaries")
    data_with_summaries = generate_summaries(data, args)

    for item in data_with_summaries:
        if "conditioning_answers_sampled" in item and isinstance(
            item.get("conditioning_answers_sampled"), np.ndarray
        ):
            item["conditioning_answers_sampled"] = item["conditioning_answers_sampled"].tolist()

    # Save results
    logger.info("Saving results")
    with open(args.summaries_file, "w") as f:
        json.dump(data_with_summaries, f, indent=4)


if __name__ == "__main__":
    wandb.init()
    # Setup
    args = parse_args()

    logger.info("Current config: %s", args)

    main(args)
