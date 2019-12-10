# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Utils for CLI
"""
import argparse
import logging

from nucleus7.utils import io_utils


def parse_config(parser: argparse.ArgumentParser) -> dict:
    """
    Parse config using parser

    Parameters
    ----------
    parser
        parser

    Returns
    -------
    config
        config
    """
    args = parser.parse_args()
    print(io_utils.print_gflags(args))
    if args.logLevel:
        logging.basicConfig(level=getattr(logging, args.logLevel))
    config = vars(args)
    config.pop("logLevel")
    return config
