#!/usr/bin/env python
"""driver for Newton-Krylov solver"""

import argparse
import logging
import os
import sys

from .utils import isclose_all_vars


def parse_args(args_list_in=None):
    """parse command line arguments"""

    args_list = [] if args_list_in is None else args_list_in
    parser = argparse.ArgumentParser(
        description="compare netCDF4 file to baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--fname", help="name of file to be compared")
    parser.add_argument("--expr_dir", help="directory with file")
    parser.add_argument("--baseline_dir", help="directory with baseline file")
    parser.add_argument("--rtol", help="relative tolerance", type=float, default=1.0e-7)
    parser.add_argument("--atol", help="absolute tolerance", type=float, default=2.0e-9)

    return parser.parse_args(args_list)


def main(args):
    """compare netCDF4 file to baseline"""

    logging_format_list = ["%(filename)s", "%(funcName)s", "%(message)s"]
    logging_format = ":".join(logging_format_list)
    logging.basicConfig(format=logging_format, level="INFO", stream=sys.stdout)
    logger = logging.getLogger(__name__)

    baseline_fname = os.path.join(args.baseline_dir, args.fname)
    expr_fname = os.path.join(args.expr_dir, args.fname)

    logger.info("expr_fname = %s", expr_fname)
    logger.info("baseline_fname = %s", baseline_fname)

    res = isclose_all_vars(expr_fname, baseline_fname, rtol=args.rtol, atol=args.atol)
    sys.exit(0 if res else 1)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
