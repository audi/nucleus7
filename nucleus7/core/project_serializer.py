# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Interfaces for config serialization
"""

import abc
from functools import wraps
import json
import logging
import os
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import tensorflow as tf

from nucleus7.core import config_logger
from nucleus7.core import project
from nucleus7.utils import file_utils
from nucleus7.utils import io_utils
from nucleus7.utils import mlflow_utils
from nucleus7.utils import nest_utils
from nucleus7.utils import object_utils
from nucleus7.utils import project_utils
from nucleus7.utils import run_utils

_EXCLUDE_CONFIG_FIELDS = [config_logger.GET_MODE_CONFIG_NAME]


class Serializer(metaclass=abc.ABCMeta):
    """
    Abstract serializer

    Parameters
    ----------
    save_dir
        directory to store the serialized config
    name
        name of the serializer
    """

    # pylint: disable=too-few-public-methods
    # this is a small interface
    def __init__(self, save_dir: str, name: Optional[str] = None):
        self.save_dir = save_dir
        self.name = name

    @abc.abstractmethod
    def serialize(self):
        """
        Serialize the configs
        """


class ConfigSerializer(Serializer, metaclass=abc.ABCMeta):
    """
    Serializer for main config logger
    """

    @staticmethod
    def get_configs_to_log() -> dict:
        """
        Get actual configs used to construct the objects
        """
        return config_logger.get_logged_configs()

    @abc.abstractmethod
    def serialize_to_file(self, configs_to_log: dict):
        """
        Abstract method to serialize the configs_to_log to the file

        Parameters
        ----------
        configs_to_log
            configs to log obtained from config_logger
        """

    def serialize(self):
        """
        Serialize the configs
        """
        configs_to_log = self.get_configs_to_log()
        self.serialize_to_file(configs_to_log)


class RunConfigSerializer(ConfigSerializer):
    """
    Serialize run configuration

    Parameters
    ----------
    save_dir
        directory to store the serialized config
    name
        name of the serializer
    single_config_names
        list of config names, where only single config is allowed; in that case,
        only first item of the config list will be used

    Attributes
    ----------
    NOT_SERIALIZABLE
        name that will be used if a config value is not json serializable
    """
    NOT_SERIALIZABLE = "NOT_SERIALIZABLE"

    def __init__(self, save_dir: str, name: Optional[str] = None,
                 single_config_names=None):
        super(RunConfigSerializer, self).__init__(save_dir, name)
        self.single_config_names = single_config_names or []

    def serialize_to_file(self, configs_to_log: dict):
        logger = logging.getLogger(__name__)
        if not self.save_dir:
            logger.warning("No save_dir provided to config serializer!"
                           "Set the project if you want to save the configs")
            return
        save_file_name = self._get_save_file_name()
        logger.info("Save run config to %s", save_file_name)
        configs_to_log = self.format_configs_to_log(configs_to_log)
        if os.path.exists(save_file_name):
            configs_to_log_loaded = io_utils.load_json(save_file_name)
            configs_to_log = project_utils.update_config_with_other_config(
                configs_to_log_loaded, configs_to_log)
        with open(save_file_name, "w", encoding='utf8') as file:
            json.dump(configs_to_log, file, indent=4, sort_keys=True,
                      default=lambda o: RunConfigSerializer.NOT_SERIALIZABLE)

    def format_configs_to_log(self, configs_to_log: dict,
                              replace_list_with_names: bool = False) -> dict:
        """
        Format the configs from raw logged format to following

        `{{name_scope}_config(s): {mode: [list of configs]}}` in case
        if _get_mode function was defined inside of the config and
        `{{name_scope}_config(s): [list of configs]}` otherwise.

        Parameters
        ----------
        configs_to_log
            configs to log in format
            `{name_scope: {object_name: [list_of_configs]}}`
        replace_list_with_names
            if the list configs must be replaced with dict of type
            {name: config}

        Returns
        -------
        configs_to_log_formatted
            formatted configs
        """
        configs_to_log_formatted = {}
        configs_to_log = _replace_namedtuples_with_dict(configs_to_log)
        for name_scope, name_scope_configs in sorted(configs_to_log.items()):
            name_scope_configs_all = []
            for _, each_obj_configs in sorted(
                    name_scope_configs.items()):
                name_scope_configs_all.extend(each_obj_configs)

            name_scope_configs_with_mode = _add_mode_to_configs(
                name_scope_configs_all)
            for each_mode, each_mode_configs in (
                    name_scope_configs_with_mode.items()):
                each_mode_configs = _filter_configs(each_mode_configs)
                each_mode_configs = _collapse_single_configs(
                    each_mode_configs, name_scope, self.single_config_names)
                if replace_list_with_names:
                    each_mode_configs = _format_list_configs_to_dict_with_name(
                        each_mode_configs)
                name_scope_configs_with_mode[each_mode] = each_mode_configs
            name_scope_configs_formatted = _collapse_modes(
                name_scope_configs_with_mode)

            name_scope_with_suffix = _add_suffix_to_config_name_space(
                name_scope, self.single_config_names)
            configs_to_log_formatted[name_scope_with_suffix] = (
                name_scope_configs_formatted)
        return configs_to_log_formatted

    def _get_save_file_name(self) -> str:
        file_name = "_config_{}.json".format(self.name)
        file_name = os.path.join(self.save_dir, file_name)
        file_name = file_utils.get_incremented_path(file_name)
        return file_name


class MlflowConfigSerializer(RunConfigSerializer):
    """
    Log the configs to a mlflow. Will be executed in case chief (also local)
    and evaluator run

    For it all the configs will be flatten with : separator and then each
    parameter and value will be added to the mlflow.
    """

    def serialize_to_file(self, configs_to_log):
        """
        Will write all the parameters in flattened view to mlflow. Nested
        Parameters will be flattened with mlflow_utils.PARAMETER_SEPARATOR

        Parameters
        ----------
        configs_to_log
            configs to log obtained from config_logger
        """
        if not run_utils.is_chief() and not run_utils.is_evaluator_run():
            return
        logger = logging.getLogger(__name__)
        logger.info("Save run config to mlflow")
        configs_to_log = self.format_configs_to_log(configs_to_log, True)
        configs_to_log_flatten = nest_utils.flatten_nested_struct(
            configs_to_log, separator=mlflow_utils.PARAMETER_SEPARATOR)
        cluster_spec = run_utils.get_cluster_spec()
        mlflow_utils.log_config_parameter_to_mlflow(
            "CLUSTER_SPEC", cluster_spec)
        for each_param_name, each_param_value in sorted(
                configs_to_log_flatten.items()):
            mlflow_utils.log_config_parameter_to_mlflow(
                each_param_name, each_param_value)


class ArtifactsSerializer(Serializer, metaclass=abc.ABCMeta):
    """
    Serializer for artifacts

    Parameters
    ----------
    artifact
        artifact to serialize
    """

    # pylint: disable=too-few-public-methods
    # this is a small interface
    # pylint: disable=abstract-method
    # this class is also an abstract class
    def __init__(self, name: str, save_dir: str, artifact):
        super(ArtifactsSerializer, self).__init__(name=name, save_dir=save_dir)
        self.artifact = artifact


class JSONArtifactsSerializer(ArtifactsSerializer):
    """
    Serialize the artifact to json
    """

    # pylint: disable=too-few-public-methods
    # this is a small interface
    def serialize(self):
        logger = logging.getLogger(__name__)
        if not self.save_dir:
            logger.warning("No save_dir provided to config serializer!"
                           "Set the project if you want to save the configs")
            return
        save_file_name = self._get_save_file_name()
        with open(save_file_name, "w", encoding='utf8') as file:
            json.dump(self.artifact, file, indent=4, sort_keys=True,
                      default=lambda o: RunConfigSerializer.NOT_SERIALIZABLE)

    def _get_save_file_name(self) -> str:
        file_name = "{}.json".format(self.name)
        file_name = os.path.join(self.save_dir, file_name)
        file_name = file_utils.get_incremented_path(file_name)
        return file_name


# pylint: disable=invalid-name
# this is a type constant, not a class
_SERIALIZER_TYPE = Union[List[Serializer], Serializer]
_CONFIG_SERIALIZER_TYPE = Union[List[ConfigSerializer], ConfigSerializer]


# pylint: enable=invalid-name

def serialize_run_configs(
        run_name: str, save_dir_attr_name: Optional[str] = None,
        single_config_names: Optional[list] = None,
        serializers_to_use: _CONFIG_SERIALIZER_TYPE = RunConfigSerializer
) -> Callable:
    """
    Serialize the run configs using serializers_to_use
    inside of the save_dir from object.{save_dir_attr_name} or to the
    artifacts directory of the active run

    Parameters
    ----------
    run_name
        name of the run
    save_dir_attr_name
        name of the attribute inside of the object with save directory inside;
        if not provided and nucleus7 active project exists, will use artifacts
        directory from the project as save directory
    single_config_names
        list of config names, where only single config is allowed; in that case,
        only first item of the config list will be used
    serializers_to_use
        single serializer class or multiple serializer classes to use

    Returns
    -------
    wrapper
        wrapped method
    """

    def wrapper(function):
        @wraps(function)
        def wrapped(self, *args, **kwargs):
            save_dir = _get_serializer_save_dir(self, save_dir_attr_name)
            serialize_all_serializers(run_name, save_dir, serializers_to_use,
                                      single_config_names=single_config_names)
            return function(self, *args, **kwargs)

        return wrapped

    return wrapper


def serialize_all_serializers(name: str,
                              save_dir: str,
                              serializers_to_use: _SERIALIZER_TYPE,
                              **serializer_kwargs):
    """
    Serialize all serializers

    Parameters
    ----------
    name
        name for serializer (same name for all)
    save_dir
        directory for serializer (same directory for all)
    serializers_to_use
        classes of serializers to use
    **serializer_kwargs
        kwargs to pass to serializer constructor

    """
    if not isinstance(serializers_to_use, list):
        serializers = [serializers_to_use]
    else:
        serializers = serializers_to_use
    for each_serializer_cls in serializers:
        serializer = each_serializer_cls(save_dir=save_dir, name=name,
                                         **serializer_kwargs)
        serializer.serialize()


def _format_list_configs_to_dict_with_name(configs: Union[list, dict]) -> dict:
    """
    Format the list of configs to the dict with keys as name from the config
    or the class name from the config

    Parameters
    ----------
    configs
        list of configs or a dict of configs

    Returns
    -------
    result
        configs formatted
    """
    if isinstance(configs, dict):
        return configs

    result = {}
    no_name_ind = 0
    for each_config in configs:
        name = each_config.get("name")
        class_name = each_config.get("class_name")
        if class_name and not name:
            name = class_name.split(".")[-1]
        if not class_name and not name:
            name = str(no_name_ind)
            no_name_ind += 1
        if name in result:
            existing_config = result[name]
            if isinstance(existing_config, list):
                existing_config.append(each_config)
            else:
                result[name] = []
                result[name].append(existing_config)
                result[name].append(each_config)
        else:
            result[name] = each_config
    return result


def _get_serializer_save_dir(self, save_dir_attr_name: Optional[str] = None
                             ) -> Optional[str]:
    save_dir = None
    if save_dir_attr_name is not None:
        save_dir = object_utils.recursive_getattr(
            self, save_dir_attr_name)
    else:
        if project.get_active_project():
            save_dir = project.get_active_artifacts_directory()
    return save_dir


def _replace_namedtuples_with_dict(nested):
    if isinstance(nested, tuple) and hasattr(nested, "_asdict"):
        nested = nested._asdict()
    if isinstance(nested, dict):
        new_nested = {}
        for each_key, each_value in nested.items():
            new_nested[each_key] = _replace_namedtuples_with_dict(each_value)
    elif isinstance(nested, (tuple, list)):
        new_nested = []
        for each_item in nested:
            new_nested.append(_replace_namedtuples_with_dict(each_item))
    else:
        new_nested = nested
    return new_nested


def _add_mode_to_configs(list_of_configs: List[dict]) -> dict:
    obj_configs_with_mode = {}
    for single_config in list_of_configs:
        mode = _get_mode_from_config(single_config)
        obj_configs_with_mode.setdefault(mode, [])
        obj_configs_with_mode[mode].append(single_config)
    return obj_configs_with_mode


def _filter_configs(list_of_configs: list) -> list:
    configs_filtered = []
    for single_config in list_of_configs:
        single_config_filtered = {k: v for k, v in single_config.items()
                                  if k not in _EXCLUDE_CONFIG_FIELDS}
        configs_filtered.append(single_config_filtered)
    return configs_filtered


def _collapse_modes(configs_with_mode: dict) -> Union[List[dict], dict]:
    if all([each_mode is None for each_mode in configs_with_mode.keys()]):
        return configs_with_mode[None]
    infer_mode = tf.estimator.ModeKeys.PREDICT
    if all([each_mode == infer_mode for each_mode in configs_with_mode.keys()]):
        return configs_with_mode[infer_mode]
    return configs_with_mode


def _add_suffix_to_config_name_space(name_scope: str,
                                     single_config_names: list) -> str:
    suffix = ("config" if name_scope in single_config_names else "configs")
    name_scope_with_suffix = "_".join([name_scope, suffix])
    return name_scope_with_suffix


def _collapse_single_configs(
        list_of_configs: list, name_scope: str,
        single_config_names: list
) -> Union[List[dict], dict]:
    if name_scope in single_config_names:
        if len(list_of_configs) > 1:
            msg = ("Found more than one config for name scope "
                   "{}: {}").format(name_scope, list_of_configs)
            raise ValueError(msg)

        return list_of_configs[0]
    return list_of_configs


def _get_mode_from_config(config: dict) -> Optional[str]:
    if config_logger.GET_MODE_CONFIG_NAME in config:
        get_mode_fn = config[config_logger.GET_MODE_CONFIG_NAME]
        mode = get_mode_fn()
        return mode
    return None
