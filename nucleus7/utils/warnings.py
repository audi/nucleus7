# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
nucleus7 specific warnings
"""
from functools import wraps
from typing import Callable
import warnings


class RegisterWarning(UserWarning):
    """
    Warning used to warn about the register issues
    """


def suppress_warnings(warning_category: Warning = RegisterWarning) -> Callable:
    """
    Suppresses warnings inside of the decorated method
    """

    def wrapped(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", warning_category)
                return function(*args, **kwargs)

        return wrapper

    return wrapped
