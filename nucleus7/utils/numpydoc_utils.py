# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Utils to work with numpydoc
"""

from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union

from numpydoc.docscrape import NumpyDocString

from nucleus7.utils.object_utils import get_full_method_signature
from nucleus7.utils.object_utils import get_parent_method


def get_docs_of_method(cls: Optional[Callable] = None,
                       method: Optional[Callable] = None,
                       include_parents=True) -> dict:
    """
    Read docs from method in NumpyDocFormat and optional add parent docs

    Parameters
    ----------
    cls
        class with method to read the docs
    method
        method to read the docs
    include_parents
        if parent methods should be traced too

    Returns
    -------
    docs
        dict with args as keys and their descriptions as values
    """
    args_with_descriptions = _get_args_with_no_description(
        cls=cls, method=method)

    if method is not None:
        descriptions_from_method = get_dict_from_doc(method, 'Parameters')
    else:
        descriptions_from_method = get_dict_from_doc(cls, 'Parameters')

    if not descriptions_from_method:
        return args_with_descriptions

    args_with_descriptions.update(
        {k: v for k, v in descriptions_from_method.items()
         if k in args_with_descriptions})
    if include_parents and cls is not None:
        parent_methods = get_parent_method(cls, method)
        for each_parent_method in parent_methods:
            parent_method_descriptions = get_dict_from_doc(
                each_parent_method, 'Parameters')
            args_with_descriptions.update(
                {k: v for k, v in parent_method_descriptions.items()
                 if k in args_with_descriptions})
    return args_with_descriptions


def get_dict_from_doc(method: Callable, section: str) -> dict:
    """
    Generate dictionary from the doc section

    Parameters
    ----------
    method
        method with docstring
    section
        section name to get the parameters

    Returns
    -------
    doc_as_dict
        dictionary with parameters as keys and description as values
    """
    doc = doc_from_method(method)
    if doc:
        return {param_doc[0]: ''.join(param_doc[2])
                for param_doc in doc[section]}
    return doc


def get_dict_from_doc_with_lists(method: Callable, section: str,
                                 bullet: str = '*') -> Dict[str, dict]:
    """
    Generate dictionary from the doc section if description is a written
    as list with bullets

    >>> method.__doc__ = '''
        Attributes
        ----------
        incoming_keys : list
            * inputs1 : description1
            * inputs2 : description2
        generated_keys : list
            * predictions1 : description1
            * predictions2 : description2
        '''
    >>> get_dict_from_doc_with_lists(method, "Attributes")
    {'incoming_keys': {'inputs1': description1,
                       'inputs2': description2},
     'generated_keys': {'predictions1': description1,
                        'predictions2': description2}}

    Parameters
    ----------
    method
        method with docstring
    section
        section name to get the parameters
    bullet
        bullet for list

    Returns
    -------
    parameters_with_descriptions
        dictionary with parameters as keys and description as values
    """
    doc = doc_from_method(method)
    result = {}
    for field in doc.get(section, []):
        result[field[0]] = {}
        field_attrs = ' '.join(field[-1]).split(bullet)
        for field_attr in field_attrs:
            if not field_attr:
                continue
            attr_spl_ind = field_attr.find(':')
            if attr_spl_ind < 0:
                attr, description = field_attr.strip(), ''
            else:
                attr, description = (field_attr[:attr_spl_ind].strip(),
                                     field_attr[attr_spl_ind + 1:].strip())
            description = " ".join(description.split())
            result[field[0]][attr] = description
    return result


def doc_from_method(method: Callable) -> Union[NumpyDocString, dict]:
    """
    Read the docstring from method to NumpyDocString format

    Parameters
    ----------
    method
        callable to inspect

    Returns
    -------
    docstring
        docstring
    """
    doc = method.__doc__
    if doc is None:
        return {}

    if not doc[:1] == '\n':
        doc = '\n    ' + doc
    doc = NumpyDocString(doc)
    return doc


def _get_args_with_no_description(cls: type, method: Callable) -> dict:
    args, kwargs, _ = get_full_method_signature(cls=cls, method=method)
    all_args = list(args) + list(kwargs)
    args_with_no_description = {
        each_key: "no description" for each_key in all_args}
    return args_with_no_description
