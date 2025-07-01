#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import json
import logging

import ray
from datasets import load_dataset
from vllm import SamplingParams

from src.selfreflect.args import parse_args
from src.selfreflect.generate_summaries import (
    DEFAULT_SAMPLING_PARAMS,
    iterate_over_ray_dataset,
)
from src.selfreflect.utils import (
    ChatTokenizer,
    format_mmlu_question,
    generate,
    set_seeds,
    setup_logging,
)

setup_logging()
logger = logging.getLogger(__name__)


def load_questions_from_huggingface(
    dataset_name, subset=None, split=None, question_column=None, num_questions: int = 1000
):
    """
    Samples questions from a huggingface dataset.
    We do not return the correct answer, because this work is only about subjective uncertainty,
    not about whether the correct answer is returned.

    :param dataset_name: String, Huggingface dataset name
    :param subset: String, subset of the dataset. If None, tries to guess it.
    :param split: String, split of the dataset. If None, tries to guess it.
    :param question_column: String, what the "question" column is called in the dataset. If None,
                            tries to guess it
    :param num_questions: int, how many questions to sample.
    :return: list of dicts each with one key, question, containing formatted questions.
    """
    # Load dataset
    load_dataset_kwargs = {}
    if subset is not None:
        load_dataset_kwargs["name"] = subset
    if split is not None:
        load_dataset_kwargs["split"] = split
    if dataset_name == "google-research-datasets/natural_questions":
        load_dataset_kwargs["name"] = "dev"
    ds = load_dataset(dataset_name, **load_dataset_kwargs)
    ds = ds.shuffle().select(range(min(num_questions, len(ds))))
    if dataset_name != "cais/mmlu":
        questions = ds[question_column]
        if dataset_name == "google-research-datasets/natural_questions":
            questions = [q["text"] for q in questions]
        for idx, q in enumerate(questions):
            q = q.strip()
            if not q.startswith("Question: "):
                q = "Question: " + q
            if not q.endswith("?"):
                q += "?"
            questions[idx] = q
    else:
        # NOTE: We do not add few-shot examples, because this did not improve performance
        #       when we tested it
        questions = [format_mmlu_question(q["subject"], q["question"], q["choices"]) for q in ds]

    # Format to be Ray compatible
    questions = [{"question": q} for q in questions]

    return questions


class QuestionTokenizer:
    def __init__(
        self,
        model_name: str,
        use_chat_template: bool,
        enable_thinking: bool,
    ):
        self.tokenizer = ChatTokenizer(
            model_name=model_name,
            use_chat_template=use_chat_template,
            enable_thinking=enable_thinking,
        )

    def __call__(self, item):
        question = item["question"]
        item["token_input"] = self.tokenizer(user_prompts=[question], add_generation_prompt=True)[
            "input_ids"
        ]

        return item


class AnswerSetSplitter:
    def __init__(
        self,
        **kwargs,
    ):
        pass

    def __call__(self, item):
        n_half = int(len(item["answers"]) // 2)
        item["masked_out_task_answers"] = item["answers"][:n_half]
        item["conditioning_answers"] = item["answers"][n_half:]
        item.pop("answers")
        item.pop("token_input", None)

        return item


def main(args):
    set_seeds(args.seed)

    # Load dataset
    questions = load_questions_from_huggingface(
        dataset_name=args.dataset_name,
        subset=args.subset,
        split=args.split,
        question_column=args.question_column,
        num_questions=args.num_questions,
    )
    with open(args.questions_file, "w") as f:
        json.dump(questions, f, indent=4)

    # Tokenize inputs (CPU)
    ray_dataset = ray.data.from_items(questions)
    ray_dataset = iterate_over_ray_dataset(
        args.model_name,
        args.use_chat_template,
        ray_dataset,
        args,
        cls=QuestionTokenizer,
        enable_thinking=False,
    )

    # Generate answers (GPU)
    sampling_param_dict = DEFAULT_SAMPLING_PARAMS.copy()
    sampling_param_dict["max_tokens"] = args.max_new_tokens
    sampling_param_dict["n"] = 2 * args.num_answers  # 2x because conditioning + test answers
    sampling_param_dict["temperature"] = 1.0
    sampling_param_dict = {"answers": SamplingParams(**sampling_param_dict)}  # type: ignore
    ray_dataset = generate(
        args.model_name,
        ray_dataset,
        sampling_param_dict,
        args,
        ray_step_name="generate_answers",
    )

    # Post-process (CPU)
    ray_dataset = iterate_over_ray_dataset(
        args.model_name, args.use_chat_template, ray_dataset, args, cls=AnswerSetSplitter
    )
    answers = ray_dataset.take_all()

    # Save
    with open(args.test_answers_file, "w") as f:
        json.dump(answers, f, indent=4)


if __name__ == "__main__":
    import wandb

    wandb.init()
    # Setup
    args = parse_args()

    logger.info("Current config: %s", args)

    main(args)
