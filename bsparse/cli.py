import argparse

import bsparse.commands


COMMANDS = {
    "encode": bsparse.commands.Encode,
    "check": bsparse.commands.Check,
    "index": bsparse.commands.Index,
    "search": bsparse.commands.Search,
    "memsearch": bsparse.commands.MemSearch,
}

DEFAULT_MODEL = "splade"
DEFAULT_DATASET = "irds"


# bsparse.models and bsparse.datasets are imported lazily so that commands that
# don't need them (e.g. index/search on already-encoded jsonl) don't pull in
# big packages like torch, transformers, and datasets
def get_models():
    import bsparse.models

    return {
        "splade": bsparse.models.SpladeModel,
        "spladepsg": bsparse.models.SpladePsgModel,
        "multilsr": bsparse.models.MultiLSRModel,
    }


def get_datasets():
    import bsparse.datasets

    return {
        "irds": bsparse.datasets.IRDSDataset,
        "jsonl": bsparse.datasets.JSONLDataset,
        "tsv": bsparse.datasets.TSVDataset,
        "hgf": bsparse.datasets.HgfDataset,
    }


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

    datasets = get_datasets() if command_cls.needs_dataset else None
    models = get_models() if command_cls.needs_model else None

    # parse the --dataset and --model, so we can add dataset-specific and model-specific args
    minimal_parser = argparse.ArgumentParser(description="bsparse CLI")
    minimal_parser.add_argument("command", choices=list(COMMANDS.keys()), help="Command")
    if command_cls.needs_dataset:
        minimal_parser.add_argument("--dataset", choices=list(datasets.keys()), required=True, help="Dataset")
    if command_cls.needs_model:
        minimal_parser.add_argument("--model", choices=list(models.keys()), required=True, help="Model")

    known_args, remaining_args = minimal_parser.parse_known_args()

    # parse the full command, so we can run() it
    full_parser = argparse.ArgumentParser(parents=[minimal_parser], add_help=False)

    if command_cls.needs_dataset:
        datasets[known_args.dataset].add_arguments(full_parser)
    if command_cls.needs_model:
        models[known_args.model].add_arguments(full_parser)

    command_cls.add_arguments(full_parser)
    args = full_parser.parse_args()

    kwargs = {}
    if command_cls.needs_dataset:
        kwargs["dataset"] = datasets[args.dataset](args)
    if command_cls.needs_model:
        kwargs["model"] = models[args.model](args)

    command = command_cls(args, **kwargs)
    command.run()


if __name__ == "__main__":
    main()
