# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Register class and API
"""

from typing import Optional
from typing import Union
import warnings

from nucleus7.utils import deprecated
from nucleus7.utils.warnings import RegisterWarning


class _Register:
    _register_name_scopes = {}
    _used_register_names = set()

    @staticmethod
    def register(name_scope: str, cls: type, name: Optional[str] = None):
        """
        Register nucleotide class

        Parameters
        ----------
        name_scope
            name scope to register to
        name
            register name
        cls
            class to register

        Warnings
        --------
        RegisterWarning
            if class name already in use; in that case new class will be used
            inside of register
        """
        name = name or cls.__name__
        if name in _Register._used_register_names:
            msg = "Name {} for class {} is already in use!".format(name, cls)
            warnings.warn(msg, RegisterWarning, stacklevel=2)

        _Register._register_name_scopes.setdefault(name_scope, {})
        _Register._register_name_scopes[name_scope][name] = cls
        _Register._used_register_names.add(name)

    @staticmethod
    def retrieve(name: str, super_cls: Optional = None) -> type:
        """
        Retrieve object by name from register

        Parameters
        ----------
        name
            name of registered class
        super_cls
            super class

        Raises
        ------
        ValueError
            if no class was registered with that name
        ValueError
            if super_cls is provided and retrieved class does not inherit from
            it

        Returns
        -------
        cls
            class from register
        """
        register_all = {}
        for crm in _Register._register_name_scopes.values():
            register_all.update(crm)
        try:
            cls = register_all[name]
        except KeyError:
            msg = ("Name {} is not inside of nucleus7 register. "
                   "Did you forget to include your library?").format(name)
            raise ValueError(msg)

        if super_cls is not None:
            if not issubclass(cls, super_cls):
                msg = ("Class {} with name {} is is not instance of "
                       "{}!!!").format(cls.__name__, name, super_cls)
                raise ValueError(msg)
        return cls

    @staticmethod
    def get() -> dict:
        """
        Get register
        """
        return _Register._register_name_scopes

    @staticmethod
    def reset():
        """
        Reset register
        """
        _Register._register_name_scopes = {}
        _Register._used_register_names = set()

    @staticmethod
    def remove_from_name_scope(name_scope: str, cls_or_name: Union[str, type]):
        """
        Remove registered cls from name scope

        Parameters
        ----------
        name_scope
            name scope
        cls_or_name
            class to remove or registered name
        """
        if name_scope not in _Register._register_name_scopes:
            return
        name_scope_register = _Register._register_name_scopes[name_scope]
        if isinstance(cls_or_name, str):
            if cls_or_name in name_scope_register:
                del name_scope_register[cls_or_name]
            if not name_scope_register:
                del _Register._register_name_scopes[name_scope]
            return
        name_to_remove = None
        for each_name, each_cls in name_scope_register.items():
            if each_cls is cls_or_name:
                name_to_remove = each_name
                break
        if name_to_remove is not None:
            del name_scope_register[name_to_remove]
        if not name_scope_register:
            del _Register._register_name_scopes[name_scope]


@deprecated.warn_deprecated_method(
    additional_instructions=("Remove this decorator from nucleus7 classes. "
                             "Register is automatically performed by "
                             "inheriting. See MetaLogAndRegister for more "
                             "details how to control it"))
def register(name: Optional[str] = None):
    # pylint: disable=unused-argument
    # the API must be the same as deprecated to not cause the failures
    """
    DEPRECATED
    """

    def _register(cls):
        return cls

    return _register


def register_to_name_scope(name_scope: str, cls: type,
                           name: Optional[str] = None):
    """
    Register nucleotide class

    Parameters
    ----------
    name_scope
        name scope to register to
    name
        register name
    cls
        class to register

    Raises
    ------
    ValueError
        if class name already in use
    """
    return _Register.register(name_scope, cls, name)


def retrieve_from_register(name: str, base_cls: Optional = None) -> type:
    """
    Retrieve object by name from register

    Parameters
    ----------
    name
        name of registered class
    base_cls
        super class

    Raises
    ------
    KeyError
        if no class was registered with that name
    ValueError
        if super_cls is provided and retrieved class does not inherit from it

    Returns
    -------
    cls
        class from register
    """
    return _Register.retrieve(name, base_cls)


def remove_from_name_scope(name_scope: str, cls_or_name: Union[str, type]):
    """
    Remove registered cls from name scope

    Parameters
    ----------
    name_scope
        name scope
    cls_or_name
        class to remove or registered name
    """
    _Register.remove_from_name_scope(name_scope, cls_or_name)


def get_register() -> dict:
    """
    Get register

    Returns
    -------
    register
        register with name scopes and registered classes inside
    """
    return _Register.get()


def reset_register():
    """
    Reset register
    """
    _Register.reset()
