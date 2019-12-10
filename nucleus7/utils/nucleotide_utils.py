# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Utils to work with nucleotides
"""

from collections import namedtuple
import importlib
import inspect
import pprint
from typing import Optional
from typing import Tuple
from typing import Union

from nucleus7.core import config_logger
from nucleus7.core import register
from nucleus7.core.fields import NucleotideKeyFields
from nucleus7.utils import deprecated
from nucleus7.utils import nest_utils
from nucleus7.utils import numpydoc_utils
from nucleus7.utils import object_utils

NucleotideInfo = namedtuple(
    "NucleotideInfo",
    ("nucleotide", "class_name", "base_classes", "constructor_parameters",
     "description", "incoming_keys_description", "generated_keys_description",
     "dynamic_incoming_keys", "dynamic_generated_keys"))

ALL_OTHERS_KEY = "*"

_NESTED_KEY_SEPARATOR = ":"


def get_nucleotide_signature(nucleotide) -> Tuple[dict, dict]:
    """
    Return signature of nucleotide.

    Arguments are taken from `self.incoming_keys` and returns from
    `self.generated_keys`. If docstring exists to main method (predict or
    process), then it is parsed to get descriptions of nodes. Docstring in
    that case should be structured in numpydoc format.

    Parameters
    ----------
    nucleotide
        nucleotide

    Returns
    -------
    arguments
        dictionary with input names as keys and its help as value
    return values
        dictionary with return value names as keys and its help as value

    Warnings
    --------
    if there are no return values inside of return section of docstring or
    inside of parameters section
    """
    if hasattr(nucleotide, 'incoming_keys'):
        arguments = {k: 'no description' for k in nucleotide.incoming_keys_all}
    else:
        arguments = {}
    if hasattr(nucleotide, 'generated_keys'):
        returns = {k: 'no description' for k in nucleotide.generated_keys_all}
    else:
        returns = {}

    if nucleotide.__doc__:
        class_input_and_return_keys = (
            numpydoc_utils.get_dict_from_doc_with_lists(nucleotide,
                                                        'Attributes'))
        class_input_and_return_keys = {
            k: v for k, v in class_input_and_return_keys.items()
            if k in ['incoming_keys', 'generated_keys']}
        class_input_keys = class_input_and_return_keys.get(
            'incoming_keys', {})
        class_return_keys = class_input_and_return_keys.get(
            'generated_keys', {})
        arguments = {k: class_input_keys.get(k, v)
                     for k, v in arguments.items()}
        returns = {k: class_return_keys.get(k, v)
                   for k, v in returns.items()}

    # pylint: disable=protected-access
    # now there is no other way to access _process_method_name; will be changed
    # in the future, when _process_method_name will be obsolete
    if hasattr(nucleotide, nucleotide._process_method_name):
        method = getattr(nucleotide, nucleotide._process_method_name)
    else:
        method = None
    # pylint: enable=protected-access
    if method is None or method.__doc__ is None:
        return arguments, returns

    arguments_ = numpydoc_utils.get_dict_from_doc(method, 'Parameters')
    arguments = {k: arguments_.get(k, v) for k, v in arguments.items()}
    returns_ = numpydoc_utils.get_dict_from_doc(method, 'Returns')
    returns = {k: returns_.get(k, v) for k, v in returns.items()}
    return arguments, returns


def get_nucleotide_info(
        *, class_name: Optional[str] = None,
        nucleotide_cls: Optional[type] = None,
        nucleotide: Optional = None) -> NucleotideInfo:
    """
    Print statistics like incoming and generated keys and class
    inheritance pattern

    Parameters
    ----------
    class_name
        complete name of the nucleotide included the package where it is
    nucleotide_cls
        nucleotide class
    nucleotide
        nucleotide itself

    Returns
    -------
    nucleotide_info
        nucleotide information

    Raises
    ------
    ValueError
        if both class_name and nucleotide_cls are provided
    """
    wrong_args_msg = (
        "Either nucleotide, nucleotide class or class name should be "
        "provided, not many of them!")
    if class_name and nucleotide_cls:
        raise ValueError(wrong_args_msg)
    if nucleotide and (class_name or nucleotide_cls):
        raise ValueError(wrong_args_msg)

    if class_name:
        _, nucleotide_cls = _register_and_retrieve_nucleotide(class_name)

    if nucleotide:
        nucleotide_cls = nucleotide.__class__

    (incoming_keys_description, generated_keys_description
     ) = get_nucleotide_signature(nucleotide_cls)
    for each_key in nucleotide_cls.incoming_keys_optional:
        description = incoming_keys_description[each_key]
        incoming_keys_description['{} (optional)'.format(each_key)] = (
            description)
    for each_key in nucleotide_cls.generated_keys_optional:
        description = generated_keys_description[each_key]
        generated_keys_description['{} (optional)'.format(each_key)] = (
            description)

    base_classes = inspect.getmro(nucleotide_cls)
    constructor_parameters = numpydoc_utils.get_docs_of_method(
        nucleotide_cls)
    description = numpydoc_utils.doc_from_method(
        nucleotide_cls).get('Summary', ["No description found"])

    nucleotide_info = NucleotideInfo(
        nucleotide=nucleotide_cls,
        class_name=class_name,
        base_classes=base_classes,
        constructor_parameters=constructor_parameters,
        description=description,
        incoming_keys_description=incoming_keys_description,
        generated_keys_description=generated_keys_description,
        dynamic_generated_keys=nucleotide_cls.dynamic_generated_keys,
        dynamic_incoming_keys=nucleotide_cls.dynamic_incoming_keys)

    if nucleotide is not None:
        nucleotide_info = _add_logged_args_to_nucleotide_info(
            nucleotide, nucleotide_info)

    return nucleotide_info


@deprecated.replace_deprecated_parameter("factory_name", "class_name")
def get_class_name_and_register_package(class_name: Union[str, None] = None):
    """
    Return only class name from full name and register components from its
    package

    Parameters
    ----------
    class_name
        full path to object class including package which should be inside
        PYTHONPATH

    Returns
    -------
    class_name
        class name without package to retrieve the class from register
    """
    package = '.'.join(class_name.split('.')[:-1])
    class_name = class_name.split('.')[-1]
    if package:
        importlib.import_module(package)
    return class_name


def collapse_inputs(list_of_inputs: list) -> dict:
    """
    Collapse the list of dicts to one dict
    If one key appears in multiple dicts, then the value in resulted dict
    with this key will be the list of values from same key of inputs
    otherwise input dicts are just combined

    >>> collapse_inputs([{'a': 1, 'b': 2}, {'a': 3, 'c': 4}])
    {'a': [1, 3], 'b': 2, 'c': 4}
    >>> collapse_inputs([{'alist': [[1], 2], 'dict': {'a': 10}},
                          {'alist': [3], 'dict': {'c': 4}}])
    {'alist': [[1], 2, 3], 'dict': {'a': 10, 'c': 4}}

    Parameters
    ----------
    list_of_inputs
        list of dicts with values to collapse

    Returns
    -------
    collapsed_dict
        collapsed version of the dict

    Raises
    ------
    ValueError
        if provided combination of the types for same key in the collapsed
        dicts is not possible, e.g. it is not possible to collapse dict with
        list or value
    ValueError
        if there is a collision of the keys in collapsing of the dicts
    """

    def _collapse_dicts(list_of_dicts):
        collapsed_dict = {}
        for each_dict in list_of_dicts:
            intersection = set(each_dict).intersection(set(collapsed_dict))
            if intersection:
                raise ValueError(
                    "Following keys intersect during collapse: {}".format(
                        intersection))
            collapsed_dict.update(each_dict)
        return collapsed_dict

    def _collapse_lists(list_of_lists):
        collapsed_list = []
        for each_list in list_of_lists:
            collapsed_list.extend(each_list)
        return collapsed_list

    def _collapse_other(list_of_values):
        if len(list_of_values) == 1:
            return list_of_values[0]
        return list(list_of_values)

    def _collapse_list_of_values(list_of_values):
        value_types = {type(v) for v in list_of_values}
        if value_types == {list}:
            return _collapse_lists(list_of_values)
        if value_types == {dict}:
            return _collapse_dicts(list_of_values)
        if value_types == {dict, list}:
            msg = ("Provided combination of input types cannot be collapsed! "
                   "{}".format(value_types))
            raise ValueError(msg)
        return _collapse_other(list_of_values)

    result_keys = {each_key for each_list in list_of_inputs
                   for each_key in each_list}

    collapsed = {}
    for each_result_key in result_keys:
        key_values = [each_list[each_result_key] for each_list in list_of_inputs
                      if each_result_key in each_list]
        key_values = _collapse_list_of_values(key_values)
        collapsed[each_result_key] = key_values

    return collapsed


def remap_and_collapse_nucleotide_inputs(
        inputs: dict, nucleotide_names: Optional[list] = None,
        nucleotide_keys_mapping: Optional[dict] = None,
) -> Optional[dict]:
    """
    Remap and collapse nucleotide inputs according to the mapping

    Parameters
    ----------
    inputs
        dict of inputs to the nucleotide, where highest level is the nucleotide
        names mapping to corresponding data,
        e.g. {nucleotide1: {key1: 1}, nucleotide2: {key2: 2}}
    nucleotide_names
        nucleotide names from inputs to use; if not specified, all the inputs
        will be used
    nucleotide_keys_mapping
        mapping for keys for each nucleotide from nucleotide_names to remap,
        e.g. {nucleotide1: {key1: new_key1}, nucleotide2: {key2: new_key2}}

    Returns
    -------
    remapped_and_collapsed_inputs
        collapsed version of the remapped inputs

    See
    ---
    :obj:`collapse_inputs`
    """
    nucleotide_keys_mapping = nucleotide_keys_mapping or {}
    nucleotide_names = (nucleotide_names if nucleotide_names is not None
                        else sorted(inputs.keys()))
    inputs_list = [inputs[each_name] for each_name in nucleotide_names]
    if None in inputs_list:
        return None

    incoming_keys_mapping_list = []
    for each_name in nucleotide_names:
        single_nucleotide_mapping = nucleotide_keys_mapping.get(each_name, {})
        if (ALL_OTHERS_KEY in nucleotide_keys_mapping
                and ALL_OTHERS_KEY not in single_nucleotide_mapping):
            single_nucleotide_mapping[ALL_OTHERS_KEY] = (
                nucleotide_keys_mapping[ALL_OTHERS_KEY])
        incoming_keys_mapping_list.append(single_nucleotide_mapping)
    remapped_and_collapsed_results = remap_and_collapse_inputs(
        inputs_list, incoming_keys_mapping_list)
    return remapped_and_collapsed_results


def remap_and_collapse_inputs(list_of_inputs: list,
                              list_of_mappings: list) -> dict:
    """
    Remaps the inputs according to mapping and collapse them

    Parameters
    ----------
    list_of_inputs
        list of dicts with values to collapse; can be nested
    list_of_mappings
        list of dicts with corresponding mappings; each mapping can be
        flat version of nested mapping with separator ':'

    Returns
    -------
    remapped_and_collapsed_inputs
        collapsed version of the remapped inputs

    See
    ---
    :obj:`collapse_inputs`
    """

    list_of_inputs_remapped = []
    for each_input, each_mapping in zip(list_of_inputs, list_of_mappings):
        inp_remapped = remap_single_input(each_input, each_mapping)
        list_of_inputs_remapped.append(inp_remapped)

    inputs_collapsed = collapse_inputs(list_of_inputs_remapped)
    return inputs_collapsed


def remap_single_input(inputs: dict, mapping: Optional[dict] = None) -> dict:
    """
    Remap single input keys according to mapping

    Parameters
    ----------
    inputs
        dict with inputs, where keys should be remapped
    mapping
        mapping of old keys to new keys; if some key was not present, it will
        be passed as is; if new key is "_", it will be ignored in remapped
        result

    Returns
    -------
    remapped_inputs
        inputs with remapped keys
    """
    inputs_remapped_flat = {}
    mapping = mapping or {}
    inputs_flat = nest_utils.flatten_nested_struct(
        inputs, separator=_NESTED_KEY_SEPARATOR)
    for old_name, value in sorted(inputs_flat.items()):
        remapped_new_names = _get_new_key_for_nested_input_and_map(
            old_name, mapping)
        for each_new_name in remapped_new_names:
            if each_new_name == NucleotideKeyFields.IGNORE_KEY:
                continue
            inputs_remapped_flat[each_new_name] = value
    inputs_remapped = (
        nest_utils.unflatten_dict_to_nested(
            inputs_remapped_flat, separator=_NESTED_KEY_SEPARATOR))
    return inputs_remapped


def get_sample_config_for_nucleotide(class_name: str) -> dict:
    """
    Generate a default json for the class

    Parameters
    ----------
    class_name
        complete name of the nucleotide included the package where it is

    Returns
    -------
    json_dict
        dict containing a default the same structure as the json config
    """
    nucleotide_name, nucleotide_cls = _register_and_retrieve_nucleotide(
        class_name)
    incoming_keys, _ = get_nucleotide_signature(nucleotide_cls)
    incoming_keys_mapping = _make_incoming_keys_sample_mapping(incoming_keys)
    node_msg = 'TODO_SET_CORRECT_INBOUND_NODES'
    inbound_nodes = [node_msg for _ in incoming_keys]

    args_to_skip = ['inbound_nodes', 'incoming_keys_mapping']

    (nucleotide_args, nucleotide_kwargs, nucleotide_defaults
     ) = object_utils.get_full_method_signature(nucleotide_cls)

    out_dict = {}

    for argument in nucleotide_args + nucleotide_kwargs:
        if argument in args_to_skip or argument in nucleotide_defaults:
            continue
        out_dict[argument] = 'TODO_SET_CORRECT_VALUE'

    out_dict.update(nucleotide_defaults)
    out_dict['incoming_keys_mapping'] = incoming_keys_mapping
    out_dict['inbound_nodes'] = inbound_nodes
    out_dict['class_name'] = class_name
    out_dict['name'] = nucleotide_name
    return out_dict


def format_nucleotide_info(nucleotide_info: NucleotideInfo,
                           width: int = 100, indent: int = 4) -> str:
    """
    Print nucleotide statistics

    Parameters
    ----------
    nucleotide_info
        nucleotide info
    width
        max width of the text
    indent
        indent to use for dict data

    Returns
    -------
    nucleotide_info_as_text
        formatted string with nucleotide info
    """
    incoming_keys_description = nucleotide_info.incoming_keys_description
    generated_keys_description = nucleotide_info.generated_keys_description
    base_class_names = [str(c).replace("<class ", "").replace(">", "")
                        for c in nucleotide_info.base_classes[:-1]]
    formatted_info = "\n".join([
        'Inheritance schema:\n'
        "\t" + " <=\n\t^<= ".join(base_class_names) + "\n",
        "Description: ",
        '\n\t' + "\n\t".join(nucleotide_info.description) + "\n",
        "Parameters: ",
        pprint.pformat(nucleotide_info.constructor_parameters, width=width,
                       indent=indent)
    ])
    formatted_info += "\n" * 2

    if not incoming_keys_description:
        formatted_info += "Nucleotide has no static incoming keys\n\n"
    else:
        formatted_info += "\n".join([
            "Static incoming keys:",
            pprint.pformat(incoming_keys_description, width=width,
                           indent=indent)
        ])
        formatted_info += "\n" * 2
    if nucleotide_info.dynamic_incoming_keys:
        formatted_info += "Nucleotide has dynamic incoming keys\n"

    if not generated_keys_description:
        formatted_info += "Nucleotide has no static generated keys\n"
    else:
        formatted_info += "\n".join([
            "Generated keys:",
            pprint.pformat(generated_keys_description, width=width,
                           indent=indent)
        ])

    if nucleotide_info.dynamic_generated_keys:
        formatted_info += "\n" * 2
        formatted_info += "Nucleotide has dynamic generated keys"
    return formatted_info


def _register_and_retrieve_nucleotide(class_name: str) -> Tuple[str, type]:
    nucleotide_name = get_class_name_and_register_package(
        class_name=class_name)

    try:
        nucleotide_cls = register.retrieve_from_register(nucleotide_name)
    # pylint: disable=invalid-name
    # is common practice to call exceptions as e
    except KeyError as e:
        msg = str(e)
        msg += "Class {} does not inherit from nucleotide".format(class_name)
        raise KeyError(msg)
    return nucleotide_name, nucleotide_cls


def _make_incoming_keys_sample_mapping(incoming_keys):
    sub_dict = {}
    for index, incoming_key in enumerate(incoming_keys):
        sub_dict['TODO_SET_INPUT_PLUGIN_NAME_{}'.format(index)] = {
            'TODO_SET_INPUT_PLUGIN_OUTPUT_KEY_NAME_{}'.format(
                index): incoming_key}
    return sub_dict


def _add_logged_args_to_nucleotide_info(
        nucleotide, nucleotide_info: NucleotideInfo) -> NucleotideInfo:
    nucleotide_logged_args = _get_logged_nucleotide_args(nucleotide)
    if not nucleotide_logged_args:
        return nucleotide_info

    def _add_value_to_description(param_description: str, used_value) -> str:
        used_value_str = str(used_value)
        description_with_value = (
            "USED_VALUE: {value}; DESCRIPTION: {description}".format(
                value=used_value_str, description=param_description))
        return description_with_value

    new_constructor_parameters = {
        each_param_name: _add_value_to_description(
            each_param_description,
            nucleotide_logged_args.get(each_param_name,
                                       "value excluded from log"))
        for each_param_name, each_param_description in
        nucleotide_info.constructor_parameters.items()}

    nucleotide_info = nucleotide_info._replace(
        constructor_parameters=new_constructor_parameters)
    return nucleotide_info


def _get_logged_nucleotide_args(nucleotide):
    logged_configs = config_logger.get_logged_configs()
    nucleotide_log_name_scope = getattr(
        nucleotide, "_log_name_scope", nucleotide.__class__.__name__)
    nucleotide_logged_args = logged_configs.get(
        nucleotide_log_name_scope, {}).get(nucleotide.name)
    if not nucleotide_logged_args:
        return None

    nucleotide_logged_args = nucleotide_logged_args[-1]
    return nucleotide_logged_args


def _get_new_key_for_nested_input_and_map(flat_key: str, mapping: dict):
    flat_key_split = flat_key.split(_NESTED_KEY_SEPARATOR)
    flat_key_levels = {_NESTED_KEY_SEPARATOR.join(flat_key_split[:i + 1])
                       for i in range(len(flat_key_split))}

    new_names = _apply_explicit_key_mappings(flat_key, flat_key_levels, mapping)

    if not new_names:
        new_name_from_other_key = _apply_all_other_key_mappings(
            flat_key, mapping)
        if new_name_from_other_key:
            new_names = [new_name_from_other_key]
        else:
            new_names = [flat_key]
    return new_names


def _apply_explicit_key_mappings(flat_key, flat_key_levels, mapping):
    new_names = []
    for each_old_key, each_new_key in mapping.items():
        if each_old_key in flat_key_levels:
            if each_new_key == NucleotideKeyFields.IGNORE_KEY:
                new_names.append(NucleotideKeyFields.IGNORE_KEY)
                continue
            new_names.append(flat_key.replace(each_old_key, each_new_key))
    return new_names


def _apply_all_other_key_mappings(flat_key: str, mapping: dict
                                  ) -> Optional[str]:
    flat_key_split = flat_key.split(_NESTED_KEY_SEPARATOR)
    flat_key_levels_with_other_keys = [
        _NESTED_KEY_SEPARATOR.join(flat_key_split[:i] + [ALL_OTHERS_KEY])
        for i in range(len(flat_key_split) - 1, -1, -1)]
    indices_from_other_mapping = []
    for each_old_key, each_new_key in mapping.items():
        if each_old_key in flat_key_levels_with_other_keys:
            indices_from_other_mapping.append(
                (flat_key_levels_with_other_keys.index(each_old_key),
                 each_old_key, each_new_key))
    if not indices_from_other_mapping:
        return None

    old_key, new_key = sorted(indices_from_other_mapping)[0][1:]
    if new_key == NucleotideKeyFields.IGNORE_KEY:
        return NucleotideKeyFields.IGNORE_KEY

    if old_key == ALL_OTHERS_KEY:
        return _NESTED_KEY_SEPARATOR.join([new_key, flat_key])

    old_key_without_other = old_key[:-(len(ALL_OTHERS_KEY)
                                       + len(_NESTED_KEY_SEPARATOR))]
    return flat_key.replace(old_key_without_other, new_key)
