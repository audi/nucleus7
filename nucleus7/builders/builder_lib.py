# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Builder library
"""
import ast
import copy
import logging
from typing import Callable
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Union

from nucleus7.core import project_global_config
from nucleus7.core import register
from nucleus7.utils import deprecated
from nucleus7.utils import nucleotide_utils
from nucleus7.utils import object_utils

PluggedConfigKeyAndBuildFn = NamedTuple(
    "PluggedConfigKeyAndBuildFn",
    [("config_key", str), ("build_fn", Callable[..., object]),
     ("add_to_config", bool)])


def log_build_process(function):
    """
    Decorator to add the logs before and after cls.build() method

    Parameters
    ----------
    function
        method to decorate with log
    """

    def wrapped(base_cls=None,
                class_name=None,
                default_cls=None,
                **object_kwargs):
        """
        Wrapped method with log

        Parameters
        ----------
        base_cls
            base class
        class_name
            class name to build
        default_cls
            default class, if no class with class name was registered
        object_kwargs
            kwargs to pass to the object constructor

        Returns
        -------
        built_object
            built object
        """
        obj_name = object_kwargs.get('name', class_name)
        if not obj_name:
            obj_name = class_name or default_cls.__name__
        logger = logging.getLogger(__name__)
        logger.info("Build %s: start", obj_name)
        obj = function(base_cls=base_cls,
                       class_name=class_name,
                       default_cls=default_cls,
                       **object_kwargs)
        logger.info("Build %s: done", obj_name)
        return obj

    return wrapped


@deprecated.replace_deprecated_parameter(
    "factory_name", "class_name", required=False)
@log_build_process
def build_registered_object(*,
                            base_cls: Union[type, None] = None,
                            class_name: Union[str, None] = None,
                            default_cls: Union[type, None] = None,
                            **object_kwargs):
    """
    Construct class with registered name by passing config_kwargs to it.
    Also calls object.build() if build method exists in that object

    Parameters
    ----------
    base_cls
        base class; will check if the class with class_name will inherits from
        this base_cls
    class_name
        class name for the object
    default_cls
        default class to use if no 'class_name' provided inside of config
    **object_kwargs
        dictionary with configuration of constructed object

    Raises
    ------
    KeyError
        if no class was registered with that name
    ValueError
        if super_cls is provided and retrieved class does not inherit from it

    Returns
    -------
    obj
        built object
    """
    assert class_name is not None or default_cls is not None, (
        "class_name or default_cls must be provided!")
    if class_name is None:
        cls = default_cls
    else:
        class_name = nucleotide_utils.get_class_name_and_register_package(
            class_name=class_name)
        cls = register.retrieve_from_register(class_name, base_cls)
    object_kwargs = project_global_config.add_defaults_to_nucleotide(
        cls, object_kwargs)
    obj = cls(**object_kwargs)
    if hasattr(obj, 'build'):
        obj.build()
    return obj


def build_registered_object_from_config(config: dict,
                                        base_cls: Union[type, None] = None,
                                        default_cls: Union[type, None] = None):
    """
    Construct class with registered name by passing config as kwargs to it.
    Also calls object.build() if build method exists in that object

    Parameters
    ----------
    config
        dictionary with configuration of constructed object
    base_cls
        base class; will check if the class with class_name will inherits from
        this base_cls
    default_cls
        default class to use if no 'class_name' provided inside of config

    Raises
    ------
    KeyError
        if no class was registered with that name
    ValueError
        if super_cls is provided and retrieved class does not inherit from it

    Returns
    -------
    obj
        built object
    """
    return build_registered_object(base_cls=base_cls,
                                   default_cls=default_cls, **config)


def build_chain_of_objects(chain_config: list, build_fn: Callable) -> list:
    """
    Build a list of objects based on its configs

    Parameters
    ----------
    chain_config
        list of configurations of objects
    build_fn
        function which takes only one argument (object config) and builds object
        based on the config

    Returns
    -------
    objects
        list of build objects
    """
    if not isinstance(chain_config, list):
        chain_config = [chain_config]
    objects = []
    for each_config in chain_config:
        obj = build_fn(each_config)
        objects.append(obj)
    return objects


def build_config_object(config_fn: Callable,
                        main_config: dict,
                        additional_search_key: Union[str, None] = None,
                        search_in_root: bool = True,
                        remove_built_config_from_main: bool = True,
                        **config_kwargs):
    """
    Build configuration using config_fn by providing arguments from
    main_config directly or from particular section in it

    Parameters
    ----------
    config_fn
        function to build the config
    main_config
        dict holding configuration to use for build; can have also other
        parameters, which are not used in the build
    additional_search_key
        name of the section (key) inside of main_config where to look for
        parameters except in root
    search_in_root
        if the parameters should be read also in the root
    remove_built_config_from_main
        if the arguments from config_fn found in section or in the root of
        main_config, should be removed from main_config
    **config_kwargs
        optional parameters to pass to config_fn

    Returns
    -------
    config_built
        configuration built using config_fn and extracted parameters
    """
    if search_in_root:
        kwargs_extracted = object_utils.select_fn_arguments_from_dict(
            config_fn, main_config)
    else:
        kwargs_extracted = dict()

    if additional_search_key:
        config_from_section = main_config.get(additional_search_key, {})
        kwargs_extracted_from_section = (
            object_utils.select_fn_arguments_from_dict(
                config_fn, config_from_section))
        configs_intersection = (set(kwargs_extracted) &
                                set(kwargs_extracted_from_section))
        if configs_intersection:
            raise ValueError("Following parameters are provided multiple "
                             "times: {}".format(configs_intersection))
        kwargs_extracted.update(kwargs_extracted_from_section)
        if remove_built_config_from_main:
            main_config.pop(additional_search_key, None)

    if remove_built_config_from_main:
        for k in kwargs_extracted:
            main_config.pop(k, None)
    config_built = config_fn(**kwargs_extracted, **config_kwargs)
    return config_built


def build_with_plugged_objects(
        config: dict,
        build_fn: Callable[..., object],
        plugged_config_keys_and_builds: List[PluggedConfigKeyAndBuildFn],
        build_callbacks: Optional[List[Callable]] = None):
    """
    Build the object using its config and build_fn. If config has keys from
    plugged_configs_and_builds, then that plugged configs will be replaced by
    built objects

    Parameters
    ----------
    config
        config to build the object from
    build_fn
        function to build the object from its config; plugged objects will
        be passed as kwargs with argument name as plugged config_key
    plugged_config_keys_and_builds
        list of tuples with plugged config keys and corresponding build
        functions
    build_callbacks
        list of callbacks to perform after build_fn on the built object; inputs
        to this callbacks are built_object and plugged_objects as kwargs, e.g.
        `def callback_fn(built_object, **plugged_objects): pass`

    Returns
    -------
    built_object
        built object using build_fn
    """
    build_callbacks = build_callbacks or []
    config = copy.deepcopy(config)
    build_plugged_objects = {}
    for each_plugged_config_key_and_build in plugged_config_keys_and_builds:
        (plugged_config_key, plugged_build_fn, add_plugged_to_config
         ) = each_plugged_config_key_and_build
        if plugged_config_key not in config:
            continue
        config_for_plugged_object = config.pop(plugged_config_key)
        built_plugged_object = plugged_build_fn(config_for_plugged_object)
        if add_plugged_to_config:
            config[plugged_config_key] = built_plugged_object
        else:
            build_plugged_objects[plugged_config_key] = built_plugged_object
    built_object = build_fn(config)
    for each_build_callback in build_callbacks:
        built_object = each_build_callback(
            built_object, **build_plugged_objects)
    return built_object


def eval_config(config: dict) -> dict:
    """
    perform literal evaluation of all values inside of config

    Parameters
    ----------
    config

    Returns
    -------
    config
        evaluated config

    """
    for each_key, each_value in config.items():
        try:
            v_eval = ast.literal_eval(each_value)
            config[each_key] = v_eval
        except (ValueError, SyntaxError):
            continue
    return config
