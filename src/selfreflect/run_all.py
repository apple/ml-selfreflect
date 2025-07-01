#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import logging
import os

from src.selfreflect.args import parse_args
from src.selfreflect.calculate_score import main as calculate_score_main
from src.selfreflect.generate_answers import main as generate_answers_main
from src.selfreflect.generate_summaries import main as generate_summaries_main

logger = logging.getLogger(__name__)


def main(args):
    # Run through all three subscripts.
    # They output files as side effect, containing questions, answers, summaries, and scores.
    if not (os.path.isfile(args.questions_file) and os.path.isfile(args.test_answers_file)) or args.regenerate_all:
        generate_answers_main(args)
    if (not os.path.isfile(args.summaries_file)) or args.regenerate_all:
        generate_summaries_main(args)
    if (not os.path.isfile(args.scores_file)) or args.regenerate_all:
        avg_results = calculate_score_main(args)
        wandb.log(avg_results)


if __name__ == "__main__":
    import wandb

    wandb.init()

    # Setup
    args = parse_args()
    logger.info("Current config: %s", args)

    main(args)
