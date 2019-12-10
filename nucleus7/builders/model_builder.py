# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Builders for model and its nucleotides
"""
import copy
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from nucleus7.builders import optimization_builders
from nucleus7.builders.builder_lib import build_registered_object
from nucleus7.model import Model
from nucleus7.model import ModelLoss
from nucleus7.model import ModelMetric
from nucleus7.model import ModelPlugin
from nucleus7.model import ModelPostProcessor
from nucleus7.model import ModelSummary
from nucleus7.model.configs import MixedPrecisionConfig
from nucleus7.optimization.configs import OptimizationConfig
from nucleus7.utils import deprecated


def build(
        model_config: Union[dict, None],
        plugins: List[ModelPlugin],
        losses: List[ModelLoss],
        postprocessors: Union[List[ModelPostProcessor], None] = None,
        metrics: Union[List[ModelMetric], None] = None,
        summaries: Union[List[ModelSummary], None] = None,
        mixed_precision_config: Optional[MixedPrecisionConfig] = None
) -> Model:
    """
    Build model and if needed build its components based of config

    Parameters
    ----------
    model_config
        model configuration
    plugins
        list of plugin configs or plugins itself
    losses
        list of loss configs or loss itself
    postprocessors
        list of postprocessor configs or postprocessor itself
    metrics
        list of metric configs or metric itself
    summaries
        list of summary configs or summary itself
    mixed_precision_config
        configuration for mixed precision

    Returns
    -------
    model
        model
    """
    # pylint: disable=too-many-arguments
    # model takes so many arguments, more split will be more confusing

    model_config.update({
        'plugins': plugins,
        'losses': losses,
        'postprocessors': postprocessors,
        'metrics': metrics,
        'summaries': summaries
    })

    model = build_registered_object(
        default_cls=Model,
        base_cls=Model,
        mixed_precision_config=mixed_precision_config,
        **model_config)
    return model


def build_model_nucleotides(
        nucleotide_configs: List[dict],
        base_cls: Union[type, None] = None
) -> Union[list, None]:
    """
    Build model nucleotides

    Parameters
    ----------
    nucleotide_configs
        list of single nucleotide configuration
    base_cls
        base class of nucleotide

    Returns
    -------
    nucleotides
        list of nucleotides
    """
    if not nucleotide_configs:
        return None
    if not isinstance(nucleotide_configs, list):
        nucleotide_configs = [nucleotide_configs]
    nucleotides = []
    for each_config in nucleotide_configs:
        if base_cls == ModelPlugin:
            nucleotide = build_plugin(**each_config)
        else:
            nucleotide = build_registered_object(
                base_cls=base_cls, **each_config)
        nucleotides.append(nucleotide)
    return nucleotides


@deprecated.replace_deprecated_parameter(
    'optimization_parameters', 'optimization_configs', required=False)
@deprecated.replace_deprecated_parameter(
    'weights_std', 'initializer::stddev', required=False)
@deprecated.replace_deprecated_parameter(
    'dropout_rate', 'dropout::rate', required=False)
@deprecated.replace_deprecated_parameter(
    'dropout_fun', 'dropout::name', required=False)
def build_plugin(**config):
    """
    Build single plugin based on its configuration

    Parameters
    ----------
    **config
        plugin configuration

    Returns
    -------
    plugin
        plugin
    """
    config = copy.deepcopy(config)
    optimization_configs_config = config.pop("optimization_configs", None)
    if optimization_configs_config is not None:
        optimization_configs = _get_optimization_configs(
            optimization_configs_config)
    else:
        optimization_configs = None
    plugin = build_registered_object(base_cls=ModelPlugin,
                                     optimization_configs=optimization_configs,
                                     **config)
    return plugin


def _get_optimization_configs(
        optimization_configs_config: dict
) -> Union[OptimizationConfig, Dict[str, OptimizationConfig]]:
    config_fields = OptimizationConfig._fields
    if set(config_fields).intersection(
            set(optimization_configs_config.keys())):
        optimization_configs = optimization_builders.build_optimization_config(
            optimization_configs_config, is_global=False)
    else:
        optimization_configs = {
            vars_pattern: optimization_builders.build_optimization_config(
                each_config, is_global=False)
            for vars_pattern, each_config in optimization_configs_config.items()
        }
    return optimization_configs
