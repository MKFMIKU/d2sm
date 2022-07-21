import argparse
import models
import datasets

parser = argparse.ArgumentParser(description="Major options for PyAnole")
parser.add_argument(
    "--template", default=".", help="You can set various templates in option.py"
)

parser.add_argument(
    "--debug", action="store_true", help="allow printing function on other process",
)

parser.add_argument(
    "-m", "--model", metavar="MODEL", default="raw2rgb", help="name for model choosing"
)
parser.add_argument(
    "-d",
    "--dataset",
    metavar="DATASET",
    default="ffdnet",
    help="name for dataset choosing",
)

# Training Related
parser.add_argument("--name", default="PyAnole", help="name of the experiments logging")
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training."
)
parser.add_argument(
    "--epochs", default=200, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=1,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument(
    "-j",
    "--workers",
    default=32,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)",
)
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--rank", default=-1, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--dist-url",
    default="tcp://localhost:10001",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.03,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--print_freq",
    type=int,
    default=100,
    help="frequency of showing training results on console",
)
parser.add_argument(
    "--display_freq",
    type=int,
    default=1000,
    help="frequency of showing training results on screen",
)
parser.add_argument(
    "--test_freq",
    type=int,
    default=1,
    help="frequency of running evaluation",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--checkpoint", default="", type=str, metavar="DIR", help="path to save checkpoints"
)


# Testing Related
parser.add_argument("--output", metavar="DIR", help="path to save output")


args, _ = parser.parse_known_args()

# modify dataset-related parser options
dataset_name = args.dataset
dataset_option_setter = datasets.get_option_setter(dataset_name)
parser = dataset_option_setter(parser)
opt, _ = parser.parse_known_args()  # parse again with new defaults

# modify model-related parser options
model_name = opt.model
model_option_setter = models.get_option_setter(model_name)
parser = model_option_setter(parser)

# args = parser.parse_args()
