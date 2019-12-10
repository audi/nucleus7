# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Builders for callbacks
"""
from nucleus7.builders.builder_lib import build_chain_of_objects
from nucleus7.builders.builder_lib import build_registered_object
from nucleus7.coordinator import CoordinatorCallback


def build(callback_config: dict) -> CoordinatorCallback:
    """
    Build callback based on its config

    Parameters
    ----------
    callback_config
        config of the callback or of KPIEvaluator to treat as a callback

    Returns
    -------
    callback
        callback
    """

    callback = build_registered_object(base_cls=CoordinatorCallback,
                                       **callback_config)
    return callback


def build_callbacks_chain(callbacks_chain_config: list) -> list:
    """
    Build a list of callbacks based on its configs

    Parameters
    ----------
    callbacks_chain_config
        list of configurations of callbacks

    Returns
    -------
    callbacks
        list of callbacks
    """
    callbacks = build_chain_of_objects(callbacks_chain_config, build)
    return callbacks
