"""Main entrypoint

"""

import argparse
import os, sys


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(args)


def main(args):
    # Process command line args
    SEED = args.seed
    raise NotImplementedError("Main entrypoint is not implemented!")


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
