import argparse

import bsparse.commands
import bsparse.datasets
import bsparse.models


COMMANDS = {
    "encode": bsparse.commands.Encode,
    "check": bsparse.commands.Check,
    "search": bsparse.commands.Search,
    "memsearch": bsparse.commands.MemSearch,
}

DEFAULT_MODELS = {
    "splade": bsparse.models.SpladeModel,
    "spladepsg": bsparse.models.SpladePsgModel,
    "multilsr": bsparse.models.MultiLSRModel,
}
DEFAULT_MODEL = "splade"

DEFAULT_DATASETS = {
    "irds": bsparse.datasets.IRDSDataset,
    "jsonl": bsparse.datasets.JSONLDataset,
    "tsv": bsparse.datasets.TSVDataset,
    "hgf": bsparse.datasets.HgfDataset,
}
DEFAULT_DATASET = "irds"


def get_command():
    parser = argparse.ArgumentParser(description="bsparse CLI")
    parser.add_argument("command", choices=list(COMMANDS.keys()), help="Command")
    known_args, remaining_args = parser.parse_known_args()
    return known_args.command


def main():
    # we create three argparsers:
    # (1) one to parse the command name using get_command()
    # (2) one to parse the --dataset and --model specified, so we can add the appropriate arguments
    # (3) one to parse the full command after adding arguments specific to the command, dataset, and model

    # parse the command name, so that we can see whether it needs --dataset and --model
    command_cls = COMMANDS[get_command()]

    # parse the --dataset and --model, so we can add dataset-specific and model-specific args
    minimal_parser = argparse.ArgumentParser(description="bsparse CLI")
    minimal_parser.add_argument("command", choices=list(COMMANDS.keys()), help="Command")
    if command_cls.needs_dataset:
        minimal_parser.add_argument("--dataset", choices=list(DEFAULT_DATASETS.keys()), required=True, help="Dataset")
    if command_cls.needs_model:
        minimal_parser.add_argument("--model", choices=list(DEFAULT_MODELS.keys()), required=True, help="Model")

    known_args, remaining_args = minimal_parser.parse_known_args()

    # parse the full command, so we can run() it
    full_parser = argparse.ArgumentParser(parents=[minimal_parser], add_help=False)

    if command_cls.needs_dataset:
        DEFAULT_DATASETS[known_args.dataset].add_arguments(full_parser)
    if command_cls.needs_model:
        DEFAULT_MODELS[known_args.model].add_arguments(full_parser)

    command_cls.add_arguments(full_parser)
    args = full_parser.parse_args()

    kwargs = {}
    if command_cls.needs_dataset:
        kwargs["dataset"] = DEFAULT_DATASETS[args.dataset](args)
    if command_cls.needs_model:
        kwargs["model"] = DEFAULT_MODELS[args.model](args)

    command = command_cls(args, **kwargs)
    command.run()


if __name__ == "__main__":
    main()
