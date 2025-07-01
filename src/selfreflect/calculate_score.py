#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import json
import logging

import ray
import sklearn.datasets
import torch
import tqdm
import transformers
import vllm
import vllm.inputs
from numpy import nan

from src.selfreflect.args import parse_args
from src.selfreflect.utils import (
    TFIDF,
    ChatTokenizer,
    VLLMBaseClass,
    average_inner_dicts,
    get_gpus_distribution,
    get_vllm_kwargs,
    set_seeds,
    setup_logging,
)

setup_logging()
logger = logging.getLogger(__name__)


def generate_masked_out_tasks(args, data: list[dict]) -> list | ray.data.Dataset:
    """
    Takes the dataset (questions, answers, and summaries),
    and then for every question adds either the sampled answers or one of the summaries as context,
    and adds a masked-out sentence, asking the assistant to predict the missing word.

    Outputs the list of all tokenized prompts we are interested in measuring the likelihood of.

    :param args: Command line arguments containing score_model_name, use_chat_template_score_model and debug_mode
    :param data: List of dictionaries. Each dict contains
        "question": String, the question / prompt
        "conditioning_answers": List of strings, these answers will be fed as context,
        "summaries": A dict of strings, one or more candidate summaries to be fed as context,
        "masked_out_task_answers" list of strings, these are used to generate the masked-out tasks.
    :return: nested list:
        Outmost list is across all questions
            then across all masked-out-tasks (all answers and all masked words)
                then a dict with one entry for each summary / the conditioning_answers set
        The innermost dict contains:
            "without_target_token_input_ids": List[int]
    """
    # Necessary to download this in the main thread to avoid race conditions when multithreading with Ray
    sklearn.datasets.fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
        data_home="/mnt/tmp",
        download_if_missing=True,
    )

    tokenizer = ChatTokenizer(
        model_name=args.score_model_name,
        use_chat_template=args.use_chat_template_score_model,
        enable_thinking=False,
    )
    tf_idf = TFIDF()

    def masked_out_task_generator_fn(question: dict):
        """
        Generate the masked out tasks for one question

        :param question: See generate_masked_out_tasks()
        :return: A list (task) of dicts (summaries) of dicts (prompts, masked word positions, ...)
        """
        # Add the summary prompts
        question["summaries"]["conditioning_answers"] = "\n".join(
            [
                "x_" + str(i + 1) + " = '" + a + "'"
                for i, a in enumerate(question["conditioning_answers"])
            ]
        )

        per_question_summary_tasks = []
        per_question_reference_tasks = []  # reference here refers to summary_key=="conditioning_answers"

        for summary_key, summary in question["summaries"].items():
            is_summary = summary_key != "conditioning_answers"

            per_question_task_list = (
                per_question_summary_tasks if is_summary else per_question_reference_tasks
            )

            for _answer_idx, answer in enumerate(question["masked_out_task_answers"]):
                answer_words = answer.split()  # This will not truncate away points etc, but this is
                # on purpose to prevent breaking, e.g., float numbers

                if not answer_words:
                    print(f"Skipping empty answer at question: {question}")
                    continue

                has_any_non_stopword = not all(tf_idf.is_stopword(answer_words))

                for word_idx, masked_word in enumerate(answer_words):
                    # Skip stopwords
                    if has_any_non_stopword and tf_idf.is_stopword([masked_word])[0]:
                        continue

                    # Create a masked version of the answer
                    temp_words = answer_words[:]
                    temp_words[word_idx] = "_"
                    masked_out_answer = " ".join(temp_words)

                    # Create prompt with masked answer
                    user_question = question["question"]
                    if not user_question.startswith("Question: "):
                        user_question = "Question: " + user_question
                    if user_question.endswith("\n"):
                        user_question = user_question[:-1]
                    if not user_question.endswith("?"):
                        user_question += "?"
                    if not is_summary:
                        user_question += (
                            f"\nSample {len(question['conditioning_answers'])} "
                            + "answers to this question."
                        )
                    assistant_answer = summary
                    user_masked_out_prompt = (
                        'We now show a text with a missing word "_". '
                        'Fill in the missing word "_" only based on the '
                        f"{'answer' if is_summary else 'distribution of answers'}"
                        f" you gave above: {masked_out_answer}\n"
                        'Please provide only the missing word "_", not the whole sentence.'
                    )

                    # Stitch together the prompt
                    task_with_answer = tokenizer(
                        user_prompts=[user_question, user_masked_out_prompt],
                        assistant_prompts=[assistant_answer, masked_word],
                        add_generation_prompt=False,
                        continue_final_message=True,
                        return_tensors=None,
                    )

                    # Stitch together the prompt without the masked word, to detect which tokens
                    # the masked word is in
                    task_without_answer = tokenizer(
                        user_prompts=[user_question, user_masked_out_prompt],
                        assistant_prompts=[assistant_answer],
                        add_generation_prompt=True,
                        continue_final_message=False,
                        return_tensors=None,
                    )
                    target_seq_len = len(task_with_answer) - len(task_without_answer)
                    if target_seq_len == 0:
                        print("No valid target tokens found at the end of the chat prompt.")
                        return float("-inf"), [], [], []
                    target_ids = list(range(len(task_without_answer), len(task_with_answer)))

                    # Create separate tasks, each predicting only 1 token for each of the `target_ids`
                    for target_id in target_ids:
                        per_question_task_list.append(
                            {
                                "without_target_token_input_ids": task_with_answer[:target_id],
                                "quantity": 1.0,
                            }
                        )

        summary_idxs_list = list(question["summaries"].keys())

        if (len(per_question_summary_tasks) == 0) or (len(per_question_reference_tasks) == 0):
            return list()

        return [
            dict(
                per_question_summary_task_list=per_question_summary_tasks,
                per_question_reference_task_list=per_question_reference_tasks,
                summary_idxs_list=summary_idxs_list,
            )
        ]

    if args.debug_mode:
        data = [masked_out_task_generator_fn(x) for x in tqdm.tqdm(data)]
    else:
        data = ray.data.from_items(data).flat_map(masked_out_task_generator_fn)

    return data


def calculate_metric(
    args,
    tasks: list | ray.data.Dataset,
) -> list[dict[str, float]]:
    """
    Given a list of tasks (generated by generate_masked_out_tasks),
    runs those prompts through the model and records the logprobs at the missing words,
    compares them between every possible pair of (summary, conditioning_answers).
    Loops this over all questions and masked-out-tasks to calculate the SelfReflect metric.

    :param args: Command line arguments containing score_model_name, seed, post_hoc_temperature, and debug_mode
    :param tasks: List of dicts, see output of generate_masked_out_tasks

    :return: List[Dict[summary_id : str, selfreflect_score : float]]. List of Dicts, one Dict per question. Each Dict contains the SelfReflect scores for all the summaries for the respective question. The Dict-key is summary_id and the value is the corresponding selfreflect_score.
    """
    if args.debug_mode:
        gpus_per_model, gpu_concurrency = 1, 1
    else:
        gpus_per_model, gpu_concurrency = get_gpus_distribution(args.score_model_name)
    logger.info(
        f"args.score_model_name={args.score_model_name}: gpus_per_model={gpus_per_model}, gpu_concurrency={gpu_concurrency}"
    )

    fn_constructor_kwargs = dict(
        model_name=args.score_model_name,
        args=args,
        post_hoc_temperature=args.post_hoc_temperature,
        tensor_parallel_size=gpus_per_model,
        cpu_gpu_variant_for_wasserstein=args.cpu_gpu_variant_for_wasserstein,
        clear_logprobs_cache=args.clear_logprobs_cache,
        clear_final_cache=args.clear_final_cache,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    if args.debug_mode:
        if isinstance(tasks, ray.data.Dataset):
            tasks = tasks.take_all()
        extractor = SelfReflectCalculator(**fn_constructor_kwargs)  # type: ignore
        tasks = [extractor(task) for task in tqdm.tqdm(tasks)]
    else:
        vllm_port = 29501

        def ray_remote_args_fn():
            nonlocal vllm_port
            ray_remote_args = dict(
                runtime_env=dict(
                    env_vars={
                        "VLLM_PORT": str(vllm_port := vllm_port + 5),
                        "VLLM_USE_V1": str(0),
                    }
                )
            )
            return ray_remote_args

        if isinstance(tasks, list):
            tasks = ray.data.from_items(tasks)
        tasks = tasks.map(
            SelfReflectCalculator,
            concurrency=gpu_concurrency,
            num_gpus=gpus_per_model,
            fn_constructor_kwargs=fn_constructor_kwargs,
            ray_remote_args_fn=ray_remote_args_fn,
        ).take_all()
    return tasks


class CaptureLogprobsProcessor:
    def __init__(self, post_hoc_temperature, cpu_gpu_variant_for_wasserstein):
        self.captured_logprobs = []
        self.post_hoc_temperature = post_hoc_temperature
        self.device = {"gpu": "cuda", "cpu_logitsprocessor": "cpu", "cpu_stack": "cuda"}[
            cpu_gpu_variant_for_wasserstein
        ]

    def __call__(self, prompt_tokens_ids, past_tokens_ids, logits_row):
        self.captured_logprobs.append(
            {
                "prompt_tokens_ids": prompt_tokens_ids,
                "past_tokens_ids": past_tokens_ids,
                "temperature_adjusted_logprobs": torch.nn.functional.softmax(
                    logits_row / self.post_hoc_temperature, dim=-1, dtype=torch.float32
                )
                .to(torch.float16)
                .to(self.device),
            }
        )
        return logits_row


class SelfReflectCalculator(VLLMBaseClass):
    def __init__(
        self,
        model_name: str,
        args,
        post_hoc_temperature: float = 5.0,
        tensor_parallel_size: int = 1,
        cpu_gpu_variant_for_wasserstein: bool = False,
        clear_logprobs_cache: bool = False,
        clear_final_cache: bool = False,
        gpu_memory_utilization: float = 0.5,
    ):
        # NB: not the same as AutoTokenizer.from_pretrained(model_name).vocab_size for an unknown reason
        self.vocab_size = transformers.AutoConfig.from_pretrained(model_name).vocab_size

        self.sampling_params = dict(temperature=1.0, top_p=1.0, max_tokens=1, n=1)
        self.post_hoc_temperature = post_hoc_temperature
        self.cpu_gpu_variant_for_wasserstein = cpu_gpu_variant_for_wasserstein
        self.clear_logprobs_cache = clear_logprobs_cache
        self.clear_final_cache = clear_final_cache

        self.llm = vllm.LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            **get_vllm_kwargs(model_name, args),
        )

    def __call__(self, per_question_task_list_tuple):
        try:
            with torch.no_grad():
                per_question_summary_task_list = per_question_task_list_tuple[
                    "per_question_summary_task_list"
                ]
                per_question_reference_task_list = per_question_task_list_tuple[
                    "per_question_reference_task_list"
                ]
                summary_idxs_list = per_question_task_list_tuple["summary_idxs_list"]

                per_question_llm_generate_input_ids_list = []

                per_question_llm_generate_input_ids_list += [
                    task["without_target_token_input_ids"]
                    for task in per_question_summary_task_list
                ]
                len_summary_task_list = len(per_question_llm_generate_input_ids_list)

                per_question_llm_generate_input_ids_list += [
                    task["without_target_token_input_ids"]
                    for task in per_question_reference_task_list
                ]

                capture_processor = CaptureLogprobsProcessor(
                    self.post_hoc_temperature, self.cpu_gpu_variant_for_wasserstein
                )
                sampling_params = vllm.SamplingParams(
                    **self.sampling_params,
                    logits_processors=[capture_processor],  # type: ignore
                )
                self.llm.generate(
                    [
                        vllm.inputs.TokensPrompt(prompt_token_ids=x)
                        for x in per_question_llm_generate_input_ids_list
                    ],
                    sampling_params=sampling_params,
                )

                # Remove logprobs in case ever more than one token is generated
                logprobs = [
                    item
                    for item in capture_processor.captured_logprobs
                    if len(item["past_tokens_ids"]) == 0
                ]
                # Sort logprobs back into the ordering of the prompts
                logprobs = {
                    item["prompt_tokens_ids"]: item["temperature_adjusted_logprobs"]
                    for item in logprobs
                }

                # Stack logprobs - keep on GPU or move to CPU based on parameter
                device = {"gpu": "cuda", "cpu_logitsprocessor": "cpu", "cpu_stack": "cpu"}[
                    self.cpu_gpu_variant_for_wasserstein
                ]
                summaries_logprobs = torch.stack(
                    [
                        logprobs[tuple(per_question_llm_generate_input_ids_list[i])].to(device)
                        for i in range(len_summary_task_list)
                    ],
                    dim=0,
                )
                reference_logprobs = torch.stack(
                    [
                        logprobs[tuple(per_question_llm_generate_input_ids_list[i])].to(device)
                        for i in range(
                            len_summary_task_list, len(per_question_llm_generate_input_ids_list)
                        )
                    ],
                    dim=0,
                )

                del capture_processor.captured_logprobs
                del capture_processor
                if self.clear_logprobs_cache:
                    torch.cuda.empty_cache()

                num_masked_out_token_tasks_per_summary = reference_logprobs.shape[0]
                num_summaries = int(
                    summaries_logprobs.shape[0] / num_masked_out_token_tasks_per_summary
                )

                summaries_logprobs = summaries_logprobs.reshape(
                    num_summaries, num_masked_out_token_tasks_per_summary, self.vocab_size
                )
                reference_logprobs = reference_logprobs.reshape(
                    1,  # broadcast dim
                    num_masked_out_token_tasks_per_summary,
                    self.vocab_size,
                )
                task_quantities = torch.tensor(
                    [task["quantity"] for task in per_question_summary_task_list],
                    device=summaries_logprobs.device,
                )
                task_quantities = task_quantities.reshape(
                    num_summaries, num_masked_out_token_tasks_per_summary
                )

                # taking mean() across individual masked-out-tokens
                # This is all in place to do this on as low VRAM as possible
                # (we want to give as much VRAM as possible to the vLLM prefix cache)
                summaries_logprobs.sub_(reference_logprobs).abs_()
                del reference_logprobs

                wasserstein_score = (summaries_logprobs.sum(dim=-1) * task_quantities).mean(
                    dim=-1
                ) * 0.5

                del summaries_logprobs

                output_dict = {
                    k: v
                    for k, v in zip(
                        summary_idxs_list, wasserstein_score.cpu().tolist(), strict=False
                    )
                }

                if self.clear_final_cache:
                    torch.cuda.empty_cache()

                return output_dict
        except Exception:
            logger.exception(
                "Can't calculate the SelfReflect score for this question, skipping it. Error:"
            )
            return {k: nan for k in per_question_task_list_tuple["summary_idxs_list"]}


def main(args):
    set_seeds(args.seed)

    # Load the summaries and the evaluation data and recombine them
    with open(args.test_answers_file) as file:
        test_answers = json.load(file)
    with open(args.summaries_file) as file:
        summaries = json.load(file)
    summaries = {item["question"]: item["summaries"] for item in summaries}
    combined = []
    for d in test_answers:
        if d["question"] in summaries:
            d["summaries"] = summaries[d["question"]]
            combined.append(d)
        else:
            logger.warning(
                f"Could not find summaries for a question, skipping it (Question = {d['question']})"
            )

    # Prepare the masked-out tasks (CPU only)
    logger.info("Generating masked-out tasks")
    tasks = generate_masked_out_tasks(args, combined)

    # Run all prompts (GPU)
    logger.info("Recording masked-out logprobs and calculating metric")
    results = calculate_metric(args, tasks)

    # Save results
    logger.info(f"Saving results: {args.scores_file}")
    with open(args.scores_file, "w") as f:
        json.dump(results, f, indent=4)

    avg_results = average_inner_dicts(results)
    logger.info(avg_results)

    return avg_results


if __name__ == "__main__":
    import wandb

    wandb.init()
    # Setup
    args = parse_args()

    logger.info("Current config: %s", args)

    main(args)
