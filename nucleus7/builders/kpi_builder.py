# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Builders for KPI evaluator
"""
import copy
from typing import List
from typing import Optional
from typing import Union

from nucleus7.builders import builder_lib
from nucleus7.core import project_global_config
from nucleus7.core import register
from nucleus7.kpi.accumulator import KPIAccumulator
from nucleus7.kpi.cacher import KPICacher
from nucleus7.kpi.kpi_callback import KPIEvaluatorCallback
from nucleus7.kpi.kpi_callback import convert_evaluator_to_callback
from nucleus7.kpi.kpi_evaluator import KPIEvaluator
from nucleus7.kpi.kpi_plugin import KPIPlugin
from nucleus7.kpi.saver import KPISaver
from nucleus7.utils import nucleotide_utils
from nucleus7.utils.utils import get_member_from_package_and_member_names


def build_kpi_plugins(configs: Union[list, dict]):
    """
    Build kpi plugins from single config or list of configs

    Parameters
    ----------
    configs
        configs for plugins to build from

    Returns
    -------
    list_of_kpi_plugins
        built kpi plugins
    """
    if not configs:
        return None
    return builder_lib.build_chain_of_objects(
        configs, build_fn=_build_kpi_plugin)


def build_kpi_evaluator(
        plugins_and_accumulators: Optional[List[Union[KPIPlugin,
                                                      KPIAccumulator]]],
) -> Optional[KPIEvaluator]:
    """
    Build kpi evaluator from plugins and accumulators

    Parameters
    ----------
    plugins_and_accumulators
        list of kpi plugins and accumulators

    Returns
    -------
    kpi_evaluator
        built kpi evaluator
    """
    if not plugins_and_accumulators:
        return None

    plugins = []
    accumulators = []
    for each_item in plugins_and_accumulators:
        if isinstance(each_item, KPIAccumulator):
            accumulators.append(each_item)
        else:
            plugins.append(each_item)
    return KPIEvaluator(plugins=plugins, accumulators=accumulators).build()


def build_kpi_evaluator_as_callback(
        plugins_and_accumulators: Optional[List[Union[KPIPlugin,
                                                      KPIAccumulator]]],
) -> Optional[KPIEvaluatorCallback]:
    """
    Build kpi evaluator from plugins and accumulators and convert it to
    callback

    Parameters
    ----------
    plugins_and_accumulators
        list of kpi plugins and accumulators

    Returns
    -------
    kpi_evaluator_as_callback
        built kpi evaluator as callback
    """
    if not plugins_and_accumulators:
        return None

    kpi_evaluator = build_kpi_evaluator(plugins_and_accumulators)
    kpi_evaluator_callback = convert_evaluator_to_callback(kpi_evaluator)
    return kpi_evaluator_callback


def _build_helper(class_name: str, base_cls: type, **kwargs):
    helper_cls = get_member_from_package_and_member_names(
        class_name, default_package_names=["nucleus7.kpi"])
    helper = helper_cls(**kwargs).build()
    if not isinstance(helper, base_cls):
        raise ValueError("Instance of {} is not instance of base class {}"
                         "".format(class_name, base_cls.__name__))
    return helper


def _build_list_of_helpers(config: Union[list, dict], base_cls: type) -> list:
    if isinstance(config, dict):
        config = [config]

    helpers = []
    for each_config in config:
        helper = _build_helper(base_cls=base_cls, **each_config)
        helpers.append(helper)
    return helpers


def _build_kpi_plugin(config, base_cls=KPIPlugin):
    config = copy.deepcopy(config)
    class_name = config.pop("class_name")
    plugin_kwargs = config

    class_name = nucleotide_utils.get_class_name_and_register_package(
        class_name=class_name)
    cls = register.retrieve_from_register(class_name, base_cls)
    plugin_kwargs = project_global_config.add_defaults_to_nucleotide(
        cls, plugin_kwargs)
    cachers_config = plugin_kwargs.pop("cachers", None)
    savers_config = plugin_kwargs.pop("savers", None)
    if cachers_config:
        plugin_kwargs["cachers"] = _build_list_of_helpers(
            cachers_config, KPICacher)
    if savers_config:
        plugin_kwargs["savers"] = _build_list_of_helpers(
            savers_config, KPISaver)
    plugin = cls(**plugin_kwargs).build()
    return plugin

# def build(evaluator_config: dict) -> KPIEvaluator:
#     """
#     Build kpi evaluator based on its config
#
#     Parameters
#     ----------
#     evaluator_config
#         config of KPIEvaluator
#
#     Returns
#     -------
#     evaluator
#         evaluator
#     """
#
#     plugged_config_keys_and_builds = [
#         builder_lib.PluggedConfigKeyAndBuildFn(
#             config_key="data_filter",
#             build_fn=data_filter_builder.build,
#             add_to_config=False),
#         builder_lib.PluggedConfigKeyAndBuildFn(
#             config_key="file_list",
#             build_fn=file_list_builder.build,
#             add_to_config=True)
#     ]
#     evaluator = builder_lib.build_with_plugged_objects(
#         config=evaluator_config,
#         build_fn=partial(builder_lib.build_registered_object_from_config,
#                          base_cls=KPIEvaluator),
#         plugged_config_keys_and_builds=plugged_config_keys_and_builds,
#         build_callbacks=[data_filter_builder.data_filter_build_callback])
#     return evaluator
