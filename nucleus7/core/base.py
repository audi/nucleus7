# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Base class and meta class for registering and logging
"""

import logging

from nucleus7.core import config_logger
from nucleus7.core import register


class MetaLogAndRegister(type):
    """
    Meta class which will register new classes and also add their constructor
    parameters to the _ConfigLogger

    Following attributes can be set inside of the class to control the register
    and config logging:

        * register_name_scope - name scope for register; if not set, original
          class name will be used (will be searched also inside of bases)
        * register_name - name of the class inside of register; if not set,
          class name will be used (will be searched only in the class)
        * log_name_scope - name scope for config logger; defaults to
          register_name_scope (will be searched also inside of bases)
        * exclude_args_from_log - list of the constructor arguments that must be
          not logged (will be searched also inside of bases)
        * exclude_from_register - flag if this class must be excluded from
          register, e.g. for interfaces and base classes
          (will be searched only in the class)
        * exclude_from_log - flag if this class must be excluded from
          constructor log, e.g. for intermediate classes, defaults to
          exclude_from_register (will be searched only in the class)
    """

    def __init__(cls, class_name, bases, attributes_dict):
        cls._register_name = MetaLogAndRegister._get_register_name(
            class_name, attributes_dict)
        cls._register_name_scope = MetaLogAndRegister._get_register_name_scope(
            class_name, bases, attributes_dict)
        cls._log_name_scope = MetaLogAndRegister._get_log_name_scope(
            class_name, bases, attributes_dict)
        cls._exclude_args_from_log = (
            MetaLogAndRegister._get_exclude_args_from_log(
                bases, attributes_dict))
        cls._exclude_from_register = (
            MetaLogAndRegister._get_exclude_from_register(attributes_dict))
        cls._exclude_from_log = (
            MetaLogAndRegister._get_exclude_from_log(attributes_dict))
        if not cls._exclude_from_register:
            register.register_to_name_scope(cls._register_name_scope, cls,
                                            name=cls._register_name)
        super().__init__(class_name, bases, attributes_dict)

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        if not cls._exclude_from_log:
            config_logger.add_constructor_parameters_to_log(
                instance, cls._log_name_scope, parameters_args=args,
                parameters_kwargs=kwargs,
                exclude_args=cls._exclude_args_from_log)
        return instance

    @staticmethod
    def _get_register_name(class_name, attributes_dict):
        return attributes_dict.get('register_name', class_name)

    @staticmethod
    def _get_exclude_from_register(attributes_dict):
        return attributes_dict.get('exclude_from_register', False)

    @staticmethod
    def _get_exclude_from_log(attributes_dict):
        exclude_from_register = MetaLogAndRegister._get_exclude_from_register(
            attributes_dict)
        return attributes_dict.get('exclude_from_log', exclude_from_register)

    @staticmethod
    def _get_register_name_scope(class_name, bases, attributes_dict):
        name_scope = (
            MetaLogAndRegister._get_attr_from_cls_or_from_bases(
                bases, attributes_dict, "register_name_scope")
            or class_name)
        return name_scope

    @staticmethod
    def _get_log_name_scope(class_name, bases, attributes_dict):
        register_name_scope = MetaLogAndRegister._get_register_name_scope(
            class_name, bases, attributes_dict)
        log_name_scope = (
            MetaLogAndRegister._get_attr_from_cls_or_from_bases(
                bases, attributes_dict, "log_name_scope")
            or register_name_scope)
        return log_name_scope

    @staticmethod
    def _get_exclude_args_from_log(bases, attributes_dict):
        exclude_from_log = (
            MetaLogAndRegister._get_attr_from_cls_or_from_bases(
                bases, attributes_dict, "exclude_args_from_log")
            or [])
        return exclude_from_log

    @staticmethod
    def _get_attr_from_cls_or_from_bases(bases: tuple, attributes_dict: dict,
                                         attr_name: str):
        attr = None
        try:
            attr = attributes_dict[attr_name]
        except KeyError:
            # pylint: disable=not-an-iterable
            # is iterable, but pylint does not recognize it
            for each_base in bases:
                attr = getattr(each_base, attr_name, None)
                if attr is not None:
                    break
        return attr


class BaseClass:
    """
    Base class to inherit from for nucleus7 objects

    Includes built property and build method which can rename / modify
    parameters depending on defaults property

    """

    def __init__(self):
        self._built = False

    def build(self):
        """
        Build the object by updating the parameters with defaults

        Returns
        -------
        self
            self object for chaining
        """
        logger = logging.getLogger(__name__)
        if self._built:
            logger.debug("Object %s is already built!", self.__class__.__name__)
            return self
        for key in self.defaults:
            self._update_argument_with_defaults(key)
        self._built = True
        return self

    @property
    def built(self) -> bool:
        """
        Flag if object was built

        Returns
        -------
        built
            True if object was built, False otherwise
        """
        return self._built

    @property
    def defaults(self):
        """
        Define default values for nucleotide parameters as a dict mapping from
        attribute name to its default value. Attribute name (key in returned
        dict) must be the same as attribute name in constructor

        Returns
        -------
        default_parameters : dict
        """
        return {}

    def _update_argument_with_defaults(self, name):
        """
        Update attribute by using default values.

        If the value is a dict and default value is also a dict, default value
        will be updated with provided value and directly set as an attribute
        with same name. Otherwise defaults will be used. If provided value is
        None, default value will be used

        Parameters
        ----------
        name : str
            name of attribute

        """
        if name not in self.defaults:
            return

        value = getattr(self, name)
        default_value = self.defaults[name]
        if callable(default_value):
            default_fn = default_value
            value = default_fn(value)
        elif value is None:
            value = default_value
        elif isinstance(value, dict) and isinstance(default_value, dict):
            all_keys = set(value) | set(default_value)
            value = {k: value.get(k, default_value[k]) for k in all_keys}
        setattr(self, name, value)
