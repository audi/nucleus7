# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Factories for tensorflow objects
"""

from functools import partial
import logging
from typing import Callable
from typing import Union
from typing import Sequence
from inspect import getfullargspec

from tensorflow import keras
import tensorflow as tf

from nucleus7.utils.utils import get_member_from_package_and_member_names
from nucleus7.optimization import optimizers as xtra_optimizers


def activation_factory(activation_or_name_with_params: Union[dict, str, type]
                       ) -> Union[Callable[[tf.Tensor], tf.Tensor], partial]:
    """
    Factory to get the activation function

    Parameters
    ----------
    activation_or_name_with_params
        either activation fn itself, then will be returned as is, or only the
        name of activation, which will be get from tf.nn or from
        keras.activations modules or a dict with name and kwargs to pass
        to the activation_fn

    Returns
    -------
    activation_fn
        activation function

    """
    if callable(activation_or_name_with_params):
        return activation_or_name_with_params

    if activation_or_name_with_params is None:
        return tf.identity

    name_with_params = activation_or_name_with_params
    name, params = _get_name_and_params(name_with_params)

    if _check_should_import_function(name):
        activation_function = _import_function(name, params)
        _check_signature(activation_function, [], 1)
        return activation_function

    assert hasattr(tf.nn, name) or hasattr(keras.activations, name), (
        "Use activation name from tf.nn or keras.activations "
        "(got {})".format(name))

    if hasattr(tf.nn, name):
        activation = getattr(tf.nn, name)
    else:
        activation = getattr(keras.activations, name)

    if not params:
        return activation
    return partial(activation, **params)


def dropout_factory(
        dropout_or_name_with_params: Union[dict, str, Callable, None]
) -> Union[Callable[[tf.Tensor, bool], tf.Tensor], None, partial]:
    """
    Factory to get the dropout function

    Parameters
    ----------
    dropout_or_name_with_params
        either dropout fn itself, then will be returned as is, or only the
        name of dropout, e.g. normal or alpha, or a dict with name and kwargs
        to pass to the dropout_fn, e.g. rate

    Returns
    -------
    dropout_fn
        dropout function or None, if rate was not provided
    """

    def _alpha_dropout(inputs, training, rate):
        keep_prob = 1 - rate
        if training:
            return tf.contrib.nn.alpha_dropout(inputs, keep_prob)
        return tf.identity(inputs)

    def _normal_dropout(inputs, training, rate, **kwargs):
        return tf.layers.dropout(inputs, rate=rate, training=training, **kwargs)

    if callable(dropout_or_name_with_params):
        return dropout_or_name_with_params

    if dropout_or_name_with_params is None:
        return None

    name_with_params = dropout_or_name_with_params
    name, params = _get_name_and_params(name_with_params, 'normal')

    if _check_should_import_function(name):
        dropout_function = _import_function(name, params)
        _check_signature(dropout_function, ['inputs', 'training'], 2)
        return dropout_function

    if 'rate' not in params or not isinstance(params['rate'], float):
        logger = logging.getLogger()
        logger.warning("To use dropout, please provide the rate parameter")
        return None

    assert name in ['normal', 'alpha'], (
        "Possible name values for dropout are ['normal', 'alpha']"
    )

    dropout = {'normal': _normal_dropout,
               'alpha': _alpha_dropout}[name]
    return partial(dropout, **params)


def initializer_factory(initializer_or_name_with_params
                        ) -> Union[keras.initializers.Initializer, None]:
    """
    Factory to construct keras initializer from its name or config

    Parameters
    ----------
    initializer_or_name_with_params
        initilaizer itself, then returned as is, or the name of initializer
        from keras.initializers namespace or config with name and params
        to pass to constructor

    Returns
    -------
    initializer
        initializer

    """
    if isinstance(initializer_or_name_with_params,
                  keras.initializers.Initializer):
        return initializer_or_name_with_params

    if initializer_or_name_with_params is None:
        return None

    name_with_params = initializer_or_name_with_params
    name, params = _get_name_and_params(name_with_params, 'RandomNormal')
    if not hasattr(keras.initializers, name):
        assert ("Use names from keras.initializers class names as initializer "
                "name (got {})".format(name))
    initializer = getattr(keras.initializers, name)(**params)

    return initializer


def optimizer_factory_from_name_and_parameters(
        optimizer_name: str,
        learning_rate: Union[tf.Tensor, float],
        **optimizer_parameters) -> tf.keras.optimizers:
    """
    Factory to construct the optimizer from its name and config

    Parameters
    ----------
    optimizer_name
        name of the optimizer
    learning_rate
        learning rate to use
    optimizer_parameters
        optimizer parameters, which will be passed to its constructor

    Returns
    -------

    """
    _remap_names = {'rmsprop': "RMSprop",
                    'adam': "Adam",
                    'adagrad': "Adagrad",
                    'sgd': "SGD",
                    'moment': "MomentumOptimizer",
                    'adadelta': "Adadelta",
                    'adamax': "Adamax",
                    'ftrl': "Ftrl",
                    'nadam': "Nadam",
                    'radam': "RectifiedAdam"}
    name = _remap_names.get(optimizer_name, optimizer_name)
    check = hasattr(tf.keras.optimizers, name) or hasattr(xtra_optimizers, name)
    assert check, ("Use optimizer name from tf.keras.optimizers or"
                   "tfa.optimizers module, e.g. RMSprop (got {})".format(name))
    if hasattr(tf.keras.optimizers, name):
        optimizer_cls = getattr(tf.keras.optimizers, name)
    else:
        optimizer_cls = getattr(xtra_optimizers, name)
    try:
        assert issubclass(optimizer_cls, tf.keras.optimizers.Optimizer), (
            "Used optimizer {} is not optimizer".format(optimizer_cls))
    except TypeError:
        raise AssertionError(
            "Used optimizer {} is not optimizer".format(optimizer_cls)
        )
    return optimizer_cls(learning_rate=learning_rate, **optimizer_parameters)


def learning_rate_decay_factory(starter_learning_rate: float,
                                global_step: tf.Tensor,
                                name: str = None,
                                **decay_params) -> tf.Tensor:
    """
    Factory to construct the learning rate decay and apply it to learning rate

    Parameters
    ----------
    starter_learning_rate
        initial learning rate
    global_step
        global step
    name
        name of the decay from tf.nn namespace excluding '_decay' suffix, e.g.
        for exponential_decay it should be exponential
    decay_params
        parameters of the decay, which will be passed to decay function

    Returns
    -------
    learning_rate
        learning_rate with decay or starter_learning_rate

    """
    if name is None:
        return tf.convert_to_tensor(starter_learning_rate, tf.float32)
    if name == 'piecewise_constant':
        assert 'boundaries' in decay_params and 'values' in decay_params, (
            "Provide boundaries and values if you want to use "
            "piecewise_constant learning rate decay! "
            "({})".format(decay_params))
        boundaries = decay_params['boundaries']
        values = decay_params['values']
        assert isinstance(boundaries, list) and isinstance(values, list), (
            "boundaries and values should be lists! ({}, {})".format(
                boundaries, values))
        assert len(values) == len(boundaries), (
            "Length of boundaries should be equal to length of values! "
            "Initial learning rate should be provided separately as "
            "learning_rate! ({}, {})".format(boundaries, values))
        values = [starter_learning_rate] + values
        learning_rate = tf.train.piecewise_constant(
            global_step, boundaries, values)
        return learning_rate

    decay_fn = getattr(tf.train, '{}_decay'.format(name))
    # pylint: disable=not-callable
    # decay_fn is really callable
    # pylint: disable=no-member
    # decay_fn has __code__ attribute since it is a callable
    if "learning_rate" in decay_fn.__code__.co_varnames:
        return decay_fn(learning_rate=starter_learning_rate,
                        global_step=global_step, **decay_params)
    return decay_fn(global_step=global_step, **decay_params)


def _get_name_and_params(params_or_name, default_name=None):
    if isinstance(params_or_name, dict):
        params = params_or_name
        name = params.pop('name', default_name)
    else:
        name = params_or_name
        params = {}
    return name, params


def _import_function(name: str, params: dict) -> Callable:
    function_to_output = get_member_from_package_and_member_names(name)
    if not callable(function_to_output):
        raise ValueError('{} is not callable'.format(name))
    if not params:
        return function_to_output
    return partial(function_to_output, **params)


def _check_should_import_function(name: str) -> bool:
    return '.' in name


def _check_signature(
        method: Callable,
        keys_must: Sequence[str],
        number_required_args: int):
    argspec = getfullargspec(method)
    number_defaults = len(argspec.defaults or [])
    actual_number_required_args = len(argspec.args) - number_defaults
    assert actual_number_required_args == number_required_args, (
        'Wrong number of parameters in function')
    parameters = argspec.args
    for counter, target_key in enumerate(keys_must):
        assert target_key == parameters[counter], (
            'function misses key {}'.format(target_key))
