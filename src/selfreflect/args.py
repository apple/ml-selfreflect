#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import argparse
import os
import re

ALL_IMPLEMENTED_STRATEGIES = "greedy basic cot sample_and_summarize"


def simplify_string(name: str, replacement: str = "_") -> str:
    # Remove characters typically disallowed in filenames
    unsafe_chars = r'[<>:"/\\|?*\x00-\x1F]'
    safe_name = re.sub(unsafe_chars, replacement, name)
    safe_name = re.sub(f"{re.escape(replacement)}+", replacement, safe_name)
    safe_name = safe_name.strip(replacement)
    return safe_name


def auto_detect_chat_template(model_name):
    use_chat_template = not any(
        (
            model_name.startswith("google/gemma-3") and model_name.endswith("-pt"),
            model_name.startswith("Qwen/Qwen2.5-") and model_name.endswith("B"),
            model_name.startswith("Qwen/Qwen3-") and model_name.endswith("-Base"),
        )
    )
    return use_chat_template


def parse_args():
    parser = argparse.ArgumentParser()
    ############### GENERAL ARGUMENTS ##################
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Huggingface identifier of model to use.",
    )
    parser.add_argument(
        "--use_chat_template",
        type=string_to_bool,
        default=None,
        help="Whether to apply chat template when generating prompts for summary generation. If None, tries to autodetect.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="WARNING",
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed (default that manual summaries have been generated for = 1)",
    )
    parser.add_argument(
        "--debug_mode",
        type=string_to_bool,
        default=False,
        help="When debug_mode=True, ray.data will not use multiprocessing and instead run in the main thread.",
    )
    parser.add_argument(
        "--regenerate_all",
        type=string_to_bool,
        default=True,
        help="In run_all.py, regenerate answer distributions and summaries even if the files already exist. Set to False to save some compute.",
    )

    ############### WANDB ARGUMENTS ##################
    parser.add_argument(
        "--experiment_sweep_name",
        type=str,
        default="debug",
        help="Wandb project name to use for logging",
    )

    ############### ANSWER GENERATION ARGUMENTS ##################
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="google-research-datasets/natural_questions",
        help="Name of the dataset to use.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Which subset to use for the huggingface dataset. If None, tries to guess it. Not necessary for all huggingface datasets.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Which split to use for the huggingface dataset. If None, tries to guess it.",
    )
    parser.add_argument(
        "--question_column",
        type=str,
        default=None,
        help="Which column in the huggingface dataset is the question. If None, tries to guess it.",
    )
    parser.add_argument(
        "--num_questions",
        type=int,
        default=1000,
        help="Number of questions to generate answers for (default: 1000)",
    )
    parser.add_argument(
        "--num_answers",
        type=int,
        default=50,
        help="Number of generated answers per question (default: 50)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256, help="How many tokens generated answer can have."
    )
    parser.add_argument(
        "--questions_file",
        type=str,
        default=None,
        help="File to store the questions. Default: Dataset + model name + questions.json",
    )
    parser.add_argument(
        "--test_answers_file",
        type=str,
        default=None,
        help="File to store the generated comparison answers. Default: Dataset + model name + test_answers.json",
    )

    ############### SUMMARY GENERATION ARGUMENTS ##################
    parser.add_argument(
        "--summary_model_name",
        type=str,
        default=None,
        help="Huggingface identifier of model to use to create summaries. In almost all cases, this should be the same that generated the answers (because only the model itself can try to selfreflect on its uncertainties), in which case you can leave this at None.",
    )
    parser.add_argument(
        "--use_chat_template_summary_model",
        type=string_to_bool,
        default=None,
        help="Whether to apply chat template for the summary model when generating prompts for summary generation. If None, tries to autodetect.",
    )
    parser.add_argument(
        "--summary_strats",
        type=str,
        default=ALL_IMPLEMENTED_STRATEGIES,  # Use a string that we split
        # ourselves instead of nargs="+", because of wandb and general config logging
        help="Which strategies to use to generate summaries. Use space between multiple ones",
    )
    parser.add_argument(
        "--model_max_new_tokens",
        type=int,
        default=4096,
        help="How many tokens generated summary can have.",
    )
    parser.add_argument(
        "--num_conditioning_answers",
        type=int,
        default=50,
        help="Number of generated answers per question for conditioning of summary via answer distribution (default: 50)",
    )
    parser.add_argument(
        "--conditioning_generation_temperature",
        type=float,
        default=1.0,
        help="Temperature to generate conditioning answers for sample_and_summarize.",
    )
    parser.add_argument(
        "--enable_thinking_for_summaries",
        type=string_to_bool,
        default=False,
        help="Whether to enable thinking mode for generate_summaries models that support it.",
    )
    parser.add_argument(
        "--summaries_file",
        type=str,
        default=None,
        help="File to save the summaries. Default: Dataset + model name + summaries.json",
    )

    ############### SELFREFLECT CALCULATION ARGUMENTS ##################
    parser.add_argument(
        "--score_model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Huggingface identifier of model to use to calculate the SelfReflect metric. Recommended: Qwen 2.5 7B, but you can also set this to None to make it the same as --model_name.",
    )
    parser.add_argument(
        "--use_chat_template_score_model",
        type=string_to_bool,
        default=None,
        help="Whether to apply chat template for the score model when generating prompts for summary generation. If None, tries to autodetect.",
    )
    parser.add_argument(
        "--scores_file",
        type=str,
        default=None,
        help="File to save the selfreflect scores. Default: Dataset + model name + scores.json",
    )
    parser.add_argument(
        "--post_hoc_temperature",
        type=float,
        default=5.0,
        help="Temperature to scale the logits in order to catch synonyms better. Recommended "
        "for instruct models: 5. While this appear quite heavy, it works quite well.",
    )
    parser.add_argument(
        "--cpu_gpu_variant_for_wasserstein",
        type=str,
        default="cpu_logitsprocessor",
        choices=("gpu", "cpu_logitsprocessor", "cpu_stack"),
        help="Whether to keep tensors on GPU for Wasserstein score computation. May be faster but uses more VRAM.",
    )
    parser.add_argument(
        "--clear_logprobs_cache",
        type=string_to_bool,
        default=False,
        help="Whether to call torch.cuda.empty_cache() after processing logprobs. May reduce VRAM usage.",
    )
    parser.add_argument(
        "--clear_final_cache",
        type=string_to_bool,
        default=False,
        help="Whether to call torch.cuda.empty_cache() at the end of processing each batch. May reduce VRAM usage.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="How much of the GPU memory should vLLM use (0.0 to 1.0). Higher values may improve performance but risk OOM errors.",
    )

    args = parser.parse_args()

    # POSTPROCESSING and setting defaults that depend on other args
    if args.summary_strats == "all":
        args.summary_strats = ALL_IMPLEMENTED_STRATEGIES
    args.summary_strats = args.summary_strats.split(" ")
    # Models
    if args.summary_model_name is None or args.summary_model_name.lower() == "none":
        args.summary_model_name = args.model_name
    if args.score_model_name is None or args.score_model_name.lower() == "none":
        args.score_model_name = args.model_name
    if args.use_chat_template is None:
        args.use_chat_template = auto_detect_chat_template(args.model_name)
    if args.use_chat_template_summary_model is None:
        args.use_chat_template_summary_model = auto_detect_chat_template(args.summary_model_name)
    if args.use_chat_template_score_model is None:
        args.use_chat_template_score_model = auto_detect_chat_template(args.score_model_name)
    if args.model_name in ("microsoft/phi-4",):
        # Phi4 has too small context length for sampling more
        args.num_conditioning_answers = max(10, args.num_conditioning_answers)
    # Filenames
    if args.questions_file is None:
        args.questions_file = os.path.join(
            "data",
            simplify_string(args.dataset_name)
            + "_"
            + simplify_string(args.model_name)
            + "_questions.json",
        )
    if args.test_answers_file is None:
        args.test_answers_file = os.path.join(
            "data",
            simplify_string(args.dataset_name)
            + "_"
            + simplify_string(args.model_name)
            + "_test_answers.json",
        )
    if args.summaries_file is None:
        args.summaries_file = os.path.join(
            "data",
            simplify_string(args.dataset_name)
            + "_"
            + simplify_string(args.model_name)
            + "_summarizedby_"
            + simplify_string(args.summary_model_name)
            + "_summaries.json",
        )
    if args.scores_file is None:
        args.scores_file = os.path.join(
            "data",
            simplify_string(args.dataset_name)
            + "_"
            + simplify_string(args.model_name)
            + "_summarizedby_"
            + simplify_string(args.summary_model_name)
            + "_scoredby_"
            + simplify_string(args.score_model_name)
            + "_scores.json",
        )
    # Default dataset splits
    args.subset = (
        {"mandarjoshi/trivia_qa": "rc", "cais/mmlu": "all"}.get(args.dataset_name, None)
        if args.subset is None
        else args.subset
    )
    args.split = (
        {
            "google-research-datasets/natural_questions": "validation"  # has no test split
        }.get(args.dataset_name, "test")
        if args.split is None
        else args.split
    )
    args.question_column = (
        {"basicv8vc/SimpleQA": "problem"}.get(args.dataset_name, "question")
        if args.question_column is None
        else args.question_column
    )

    if args.enable_thinking_for_summaries and not (
        (
            args.summary_model_name.startswith("Qwen/Qwen3-")
            and not args.summary_model_name.endswith("Base")
        )
        or (
            args.summary_model_name
            in (
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                "Qwen/QwQ-32B",
            )
        )
    ):
        raise ValueError("Thinking mode is not implemented for this model.")

    # Validate GPU utilization is within valid range
    if args.gpu_memory_utilization < 0.0 or args.gpu_memory_utilization > 1.0:
        ValueError(
            f"GPU memory utilization must be between 0.0 and 1.0. Got {args.gpu_memory_utilization}."
        )

    return args


def string_to_bool(x):
    if isinstance(x, bool):
        return x
    elif isinstance(x, str):
        if x.lower() in ("true", "y", "1"):
            return True
        elif x.lower() in ("false", "n", "0"):
            return False
    else:
        raise argparse.ArgumentTypeError("Please provide a Boolean value")
