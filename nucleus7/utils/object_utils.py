# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Utils to work with objects
"""

from functools import wraps
import inspect
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple


def get_parent_method(cls: Optional[Callable] = None,
                      method: Optional[Callable] = None
                      ) -> Optional[List[Callable]]:
    """
    Get the parent class if method is not specified otherwise get all parent
    methods which are overridden by method

    Parameters
    ----------
    cls
        class to get the parents
    method
        method of the class to search for parents

    Returns
    -------
    parent
        parent class or method
    """
    if cls is None:
        return None
    parents = inspect.getmro(cls)[1:]
    if method is None:
        return list(parents)
    parent_methods = [getattr(cls, method.__name__)
                      for cls in parents
                      if hasattr(cls, method.__name__)
                      and cls is not object]
    return parent_methods


def get_method_signatures(cls: Optional[Callable] = None,
                          method: Optional[Callable] = None,
                          include_parents: bool = True
                          ) -> List[inspect.Signature]:
    """
    Get signature of the method together with signatures of the parents
    Parameters
    ----------
    cls
        class to get the parents
    method
        method to search for

    include_parents
        if the parents must be included in the signatures

    Returns
    -------
    list_of_signatures
        signatures in hierarchical order (method is first)

    """
    signatures = [inspect.signature(method)]
    if include_parents:
        parent_methods = get_parent_method(cls, method)
        for each_parent_method in parent_methods:
            parent_signature = inspect.signature(each_parent_method)
            signatures.append(parent_signature)
    return signatures


def get_object_name(obj):
    """
    Return object name or class name if object has no __name__ field. If it is \
    a class, return class name

    Parameters
    ----------
    obj
        Input to get name from

    Returns
    -------
    name
        The requested name
    """
    try:
        name_str = obj.__repr__()
    except TypeError:
        return str(obj)
    if name_str.find('object at') == -1:
        return name_str
    if hasattr(obj, '__name__'):
        return obj.__name__
    return obj.__class__.__name__


def assert_is_built(function: Callable):
    """
    Decorator to check if self._built == True else raises ValueError
    with name of object and message to build it

    Parameters
    ----------
    function
        method to decorate

    Returns
    -------
    decorated_method
        decorated method

    Raises
    ------
    ValueError
        if self.built != True
    """

    @wraps(function)
    def wrapped(self, *args, **kwargs):
        assert_object_is_built(self)
        return function(self, *args, **kwargs)

    return wrapped


def assert_object_is_built(obj: object):
    """
    Check if object is built, e.g. self.built == True

    Parameters
    ----------
    obj
        object to check

    Returns
    -------
    decorated_method
        decorated method

    Raises
    ------
    ValueError
        if object.built != True
    """
    if not obj.built:
        class_name = obj.__class__.__name__
        msg = "Object {} is not built! Call self.build() first".format(
            class_name)
        raise ValueError(msg)


# pylint: disable=invalid-name
# this will be used as a property decorator, so has same name pattern
class classproperty(property):
    """
    Add the class property to class

    Usage is same as a @property, but this properties first argument is class
    itself, not a object
    """

    # pylint: disable=no-member
    # __get__ method is a method of fget, since it is a descriptor
    def __get__(self, self_object, self_cls):
        if self_object is None:
            return self.fget.__get__(self_cls)()
        return self.fget.__get__(self_object)()


# pylint: enable=invalid-name


def select_fn_arguments_from_dict(function: Callable, config: dict) -> dict:
    """
    Select matching to function signature keyword parameters from config

    Parameters
    ----------
    function
        function to get the parameters for
    config
        configuration

    Returns
    -------
    kwargs
        keyword arguments for function

    """
    fn_parameters = set(inspect.signature(function).parameters.keys())
    fn_kwargs = {k: config[k] for k in fn_parameters if k in config}
    return fn_kwargs


def assert_property_is_defined(property_name: str):
    """
    Check if object property is not None

    Parameters
    ----------
    property_name
        name of the property

    Raises
    ------
    ValueError
        if obj.[property] is None
    """

    def wrapper(function):
        @wraps(function)
        def wrapped(self, *args, **kwargs):
            if getattr(self, property_name) is None:
                class_name = self.__class__.__name__
                object_name = getattr(self, 'name', None)
                msg = ("Property {} of object {} with class {} is "
                       "not set!").format(property_name, object_name,
                                          class_name)
                raise ValueError(msg)
            return function(self, *args, **kwargs)

        return wrapped

    return wrapper


def get_full_method_signature(cls: Optional[Callable] = None,
                              method: Optional[Callable] = None
                              ) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    This method gets all arguments of the current method as well as its
    parents implementations. This is useful when creating a child class and
    passing the constructor arguments to the parent using `**kwargs`

    Parameters
    ----------
    cls
        class to get the parents
    method
        method of the class to search for parents

    Returns
    -------
    args
        list of args, e.g. ordered arguments
    kwargs
        list of keyword only arguments
    defaults
        dict with mapping of argument name to its default value
    """
    parent_methods = get_parent_method(cls, method)
    if parent_methods is not None:
        methods_to_inspect = [method or cls] + list(parent_methods)
    else:
        methods_to_inspect = [method or cls]

    args = []
    kwargs = []
    defaults = {}

    check_args = True
    check_kwargs = True
    for each_method in methods_to_inspect:
        if not check_args and not check_kwargs:
            break
        # pylint: disable=unidiomatic-typecheck
        # this is needed because otherwise the metaclass.__call__ method
        # will be inspected and not method.__init__
        if inspect.isclass(each_method) and not type(each_method) is type:
            each_method = each_method.__init__
        full_arg_scope = inspect.getfullargspec(each_method)

        method_signature = inspect.signature(each_method)
        params_order = full_arg_scope.args + full_arg_scope.kwonlyargs

        for each_param_name in params_order:
            if each_param_name == 'self':
                continue
            each_param = method_signature.parameters[each_param_name]
            if each_param.kind == inspect.Parameter.KEYWORD_ONLY:
                if (check_kwargs
                        and each_param_name not in kwargs
                        and each_param_name not in args):
                    kwargs.append(each_param_name)
            else:
                if (check_args
                        and each_param_name not in kwargs
                        and each_param_name not in args):
                    args.append(each_param_name)
            # pylint: disable=protected-access
            # _empty must be used since it is a part of inspect
            if each_param.default is not inspect._empty:
                defaults[each_param_name] = each_param.default
            # pylint: enable=protected-access

        check_args = full_arg_scope.varargs is not None
        check_kwargs = full_arg_scope.varkw is not None

    return args, kwargs, defaults


def get_class_name_with_module(obj) -> str:
    """
    Get the full class name, e.g. module name and class name out of the object
    class

    Parameters
    ----------
    obj
        object

    Returns
    -------
    module_and_class_name
        module (if exist) and class name
    """
    module_name = obj.__class__.__module__
    class_name = obj.__class__.__name__
    if not module_name:
        return class_name
    return '.'.join([module_name, class_name])


def recursive_getattr(obj: object, attr_name: str, separator: str = "."):
    """
    Recursive getattr to get the attr of the attr, e.g. if object has attribute
    a and it is a object that holds a attribute b with value V, then you
    can access it with `recursive_getattr(obj, 'a.b')`

    Parameters
    ----------
    obj
        object with attributes
    attr_name
        name of the attribute to get; levels are separated using separator
    separator
        separator to separate different attribute levels

    Returns
    -------
    value
        value of the recursive attribute of the object
    """
    attr_split = attr_name.split(separator)
    value = obj
    for each_attr in attr_split:
        value = getattr(value, each_attr)
    return value
