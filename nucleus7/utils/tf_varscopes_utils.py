# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Utils to work with tensorflow variable scopes
"""

from functools import wraps
from typing import Callable
from typing import List
from typing import Union

import tensorflow as tf


def with_named_variable_scope(function: Callable) -> Callable:
    """
    Decorate the function to use the variable scope and if attribute
    `initializer` exists in object, uses it as initializer of variable scope

    Parameters
    ----------
    function : callable
        function to decorate

    Returns
    -------
    decorated_f
        decorated method
    """

    @wraps(function)
    def wrapped(self, *args, **kwargs):
        initializer = getattr(self, 'initializer', None)
        name = self.name if hasattr(self, "name") else self.__class__.__name__
        with tf.variable_scope(name, initializer=initializer):
            res = function(self, *args, **kwargs)
        return res

    return wrapped


def remove_scope_from_name(var_name: str, var_scope: str) -> str:
    """
    Remove variable scope from the variable name

    Parameters
    ----------
    var_name
        name of the variable
    var_scope
        name of the scope to remove

    Returns
    -------
    variable_name_without_scope
        variable name without scope in it

    """
    return var_name.replace(var_scope, '', 1)[1:]


def add_var_scope_and_variables(function: Callable) -> Callable:
    """
    Decorator to add variable scope and variables to self._variable_scope and
    self._variables
    """

    @wraps(function)
    def wrapped(self, *args, **kwargs):
        res = function(self, *args, **kwargs)
        # pylint: disable=protected-access
        # _variable_scope is a private argument of the self, so it must be
        # not altered in the API, but this decorated is only for it
        self._variable_scope = tf.get_variable_scope().name
        self._variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, self._variable_scope)
        # pylint: enable=protected-access
        return res

    return wrapped


def get_var_scope_and_variables() -> (str, List[Union[tf.Variable, tf.Tensor]]):
    """
    Get variables inside of current variable scope

    Returns
    -------
    variable_scope
        name of current variable scope
    variables
        list of tensorflow variables
    """
    variable_scope = tf.get_variable_scope().name
    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  variable_scope)
    return variable_scope, variables


def with_name_scope(name: str) -> Callable:
    """
    Decorator to execute the decorated method under specific name scope

    Parameters
    ----------
    name
        name of name_scope

    Returns
    -------
    wrapper
        decorated method
    """

    def wrapper(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with tf.name_scope(name):
                result = function(*args, **kwargs)
            return result

        return wrapped

    return wrapper
