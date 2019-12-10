# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Utils for tensorflow
"""

from functools import wraps
import json
import logging
import os
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
import warnings

from matplotlib import pyplot as plt
import tensorflow as tf
# pylint: disable=no-name-in-module
# python and core are a part of tensorflow
from tensorflow.python.client import device_lib

# pylint: enable=no-name-in-module
from nucleus7.utils import io_utils
from nucleus7.utils import nest_utils
from nucleus7.utils import np_utils
from nucleus7.utils import tf_collections_utils
from nucleus7.utils import tf_ops
from nucleus7.utils import tf_varscopes_utils

# pylint: disable=invalid-name
# this is type constant, not a class
_NESTED_TYPE = Dict[str, Union[tf.Tensor, Dict[str, tf.Tensor]]]


# pylint: enable=invalid-name


def figure_to_summary(fig: plt.Figure,
                      vis_placeholder: tf.Tensor,
                      vis_summary: tf.summary.Summary,
                      summary_writer: tf.summary.FileWriter,
                      global_step: tf.Tensor):
    """
    Write matplotlib figure to summary

    Parameters
    ----------
    fig
        matplotlib figure
    vis_placeholder
        placeholder to evaluate the summary
    vis_summary
        summary to use
    summary_writer
        summary writer
    global_step
        global step

    """
    image = np_utils.fig2rgb_array(fig)
    summary_writer.add_summary(
        vis_summary.eval(feed_dict={vis_placeholder: image}),
        global_step=global_step)


def count_trainable_params(graph: Optional[tf.Graph] = None) -> int:
    """
    Count number of trainable parameters inside of `tf.trainable_variables`

    Parameters
    ----------
    graph
        tensorflow graph

    Returns
    -------
    number_of_parameters
        number of trainable parameters
    """
    graph = graph or tf.get_default_graph()
    total_parameters = 0
    for variable in graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


def split_dict_to_list_of_dict(
        dict_with_tensors: Dict[str, tf.Tensor],
        num_of_splits: int,
        axis=0,
        pad_to_batch=False) -> List[Dict[str, tf.Tensor]]:
    """
    Split the dict of tensors to list of dicts with same structure

    Parameters
    ----------
    dict_with_tensors
        dict with tensors
    num_of_splits
        number of splits
    axis
        axis to split the tensors
    pad_to_batch
        whether to pad

    Returns
    -------
    list_of_dicts
        list of split dicts

    """
    if pad_to_batch:
        split_fn = tf_ops.split_with_pad
    else:
        split_fn = tf.split
    dict_split = {k: split_fn(inp, num_of_splits, axis)
                  for k, inp in dict_with_tensors.items()}
    dict_list_split = []
    for i in range(num_of_splits):
        subdict = {k: inp_spl[i] for k, inp_spl in dict_split.items()}
        dict_list_split.append(subdict)
    return dict_list_split


def get_available_gpus() -> List[str]:
    """
    Returns the names of available gpus

    Returns
    -------
    gpu_names
        names of available gpus
    """
    try:
        # pylint: disable=no-member
        # pycuda.driver has cuda member
        # pylint: disable=unused-import
        # pycuda.autoinit must be imported to initialize cuda
        # pylint: disable=import-error
        # pycuda is basically not a must
        import pycuda.driver as cuda
        try:
            import pycuda.autoinit
            num_devices = cuda.Device.count()
            devices = ['/device:GPU:{}'.format(i) for i in range(num_devices)]
            return devices
        except cuda.RuntimeError:
            return []
        # pylint: enable=no-member
        # pylint: enable=unused-import
    except ImportError:
        with tf.Session(config=tf.ConfigProto(
                gpu_options={"allow_growth": True})):
            local_device_protos = device_lib.list_local_devices()
            return [x.name for x in local_device_protos
                    if x.device_type == 'GPU']


def get_connected_inputs_to_predictions(inputs: Dict[str, tf.Tensor],
                                        predictions: Dict[str, tf.Tensor],
                                        graph: tf.Graph
                                        ) -> Dict[str, tf.Tensor]:
    """
    Get inputs that have connections to predictions

    Parameters
    ----------
    inputs
        nested structure of inputs
    predictions
        nested structure of predictions
    graph
        instance of tensorflow graph with inputs and predictions

    Returns
    -------
    inputs_connected
        same structure dict as inputs, but without values that have no
        connection to predictions
    """

    def _get_predecessors(operation):
        return set(each_input.op for each_input in operation.inputs)

    def _get_all_predecessors(operation, list_of_predecessors=None):
        list_of_predecessors = list_of_predecessors or set()
        preds = _get_predecessors(operation)
        if not preds:
            return list_of_predecessors
        for op_pred in preds:
            children_ = _get_all_predecessors(
                op_pred, list_of_predecessors=list_of_predecessors)
            list_of_predecessors.update(children_)
        return list_of_predecessors

    inputs_tensors_dict = nest_utils.flatten_nested_struct(inputs)
    inputs_ops = {k: t.op for k, t in inputs_tensors_dict.items()}

    predictions_tensors = list(
        nest_utils.flatten_nested_struct(predictions).values())
    predictions_ops = [t.op for t in predictions_tensors]

    subgraph_def = tf.graph_util.extract_sub_graph(
        graph.as_graph_def(),
        [n.name for n in predictions_ops]
    )
    # pylint: disable=not-context-manager
    # tf.Graph().as_default() is a context manager
    with tf.Graph().as_default():
        tf.import_graph_def(subgraph_def, name="")
        subgraph_op_names = {
            op.name for op in tf.get_default_graph().get_operations()}
    # pylint: enable=not-context-manager
    for k, each_op in inputs_ops.items():
        if each_op.name not in subgraph_op_names:
            del inputs_tensors_dict[k]

    inputs_connected = nest_utils.unflatten_dict_to_nested(inputs_tensors_dict)
    return inputs_connected


def remove_tag_from_variable_name(name: str) -> str:
    """
    Remove variable tag from variable name

    Parameters
    ----------
    name
        variable name

    Returns
    -------
    name_without_tag
        name without tag

    Examples
    --------
    >>> remove_tag_from_variable_name('model/variable/name:1')
    'model/variable/name'

    >>> remove_tag_from_variable_name('model/variable/name')
    'model/variable/name'

    """
    name_spl = name.split(':')
    if len(name_spl) > 1:
        name_spl = name_spl[:-1]
    name_without_tag = ':'.join(name_spl)
    return name_without_tag


def combine_multiple_graphs_from_meta_and_checkpoint(
        meta_graph_paths: dict, checkpoint_paths: dict,
        save_dir: str):
    """
    Combine multiple graphs and their checkpoints
    Each graph will be imported under its variable_scope, so no collisions
    of variables will happen

    Parameters
    ----------
    meta_graph_paths
        dict with keys as names of the graphs and afterwards the variable scope
        names where this graph is imported and values are paths to corresponding
        graph.meta files
    checkpoint_paths
        dict with same keys as meta_graph_paths and values pointing to the
        corresponding checkpoints
    save_dir
        directory where to save the combined graph

    """
    # TODO(oleksandr.vorobiov@audi.de): split to multiple methods

    assert set(meta_graph_paths) == set(checkpoint_paths), (
        "Meta graphs and checkpoints must have same keys ({}, {})".format(
            meta_graph_paths.keys(), checkpoint_paths.keys()))
    for each_graph_name in meta_graph_paths:
        assert "//" not in each_graph_name and "::" not in each_graph_name, (
            "Symbols '//' and '::' are not allowed inside of graph names "
            "({})".format(each_graph_name))

    tf.reset_default_graph()

    logger = logging.getLogger(__name__)

    save_file_name_main = "combined_" + "-".join(meta_graph_paths.keys())
    save_file_name = os.path.join(
        save_dir, save_file_name_main + '.chpt')

    variables_loaded = set()
    savers = {}

    # load and merge graphs
    for graph_name in sorted(meta_graph_paths):
        saver = _load_graph(graph_name, meta_graph_paths, variables_loaded)
        savers[graph_name] = saver

    tf.train.export_meta_graph(save_file_name + '.meta')

    saver_complete = tf.train.Saver()

    io_utils.maybe_mkdir(save_dir)
    with tf.Session() as sess:
        for graph_name, saver in savers.items():
            checkpoint_path = checkpoint_paths[graph_name]
            if saver is not None:
                saver.restore(sess, checkpoint_path)

        logger.info("Save combined checkpoint and graph to %s", save_file_name)
        saver_complete.save(sess, save_file_name, write_meta_graph=False)


def _load_graph(graph_name: str, meta_graph_paths: dict, variables_loaded: set
                ) -> tf.train.Saver:
    # TODO(oleksandr.vorobiov@audi.de): refactor to have CollectionNames import
    #  in the main imports
    from nucleus7.model.fields import CollectionNames

    logger = logging.getLogger(__name__)
    meta_graph_path = meta_graph_paths[graph_name]
    other_graph_names = [name for name in meta_graph_paths
                         if name != graph_name]
    logger.info("Load graph %s from %s", graph_name, meta_graph_path)
    _ = tf.train.import_meta_graph(
        meta_graph_path, clear_devices=True, import_scope=graph_name)
    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope=graph_name)
    if not variables:
        add_variables_from_graph_without_collection(
            graph=tf.get_default_graph())
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                      scope=graph_name)
    saver_var_list = {}
    for each_variable in variables:
        if each_variable not in variables_loaded:
            original_variable_name = (
                tf_varscopes_utils.remove_scope_from_name(
                    each_variable.name, graph_name)
            )
            original_variable_name = remove_tag_from_variable_name(
                original_variable_name)
            saver_var_list[original_variable_name] = each_variable
            variables_loaded.add(each_variable)
    if saver_var_list:
        saver = tf.train.Saver(var_list=saver_var_list)
    else:
        saver = None
        logger.warning("Graph with name %s and path %s has no variables "
                       "to restore!", graph_name, meta_graph_path)
    logger.info("Add level to collection names with graph name %s",
                graph_name)
    tf_collections_utils.add_prefix_to_collection_with_prefix(
        graph_name, CollectionNames.INPUTS,
        collection_names_to_leave=other_graph_names)
    tf_collections_utils.add_prefix_to_collection_with_prefix(
        graph_name, CollectionNames.PREDICTIONS,
        collection_names_to_leave=other_graph_names)
    return saver


def save_input_output_node_names(
        path: str,
        inputs_collection_name: str,
        outputs_collection_names: str,
        parameter_placeholders=None,
        graph: tf.Graph = None):
    """
    Save json file with input and output node names and its shapes

    Resulted file has following format
    {'inputs': {'name': , 'shape': }, 'outputs': {'name': , 'shape': }}

    Parameters
    ----------
    path
        path to save
    inputs_collection_name
        name of collection for inputs
    outputs_collection_names
        name of collection for outputs
    parameter_placeholders
        optional list of DefaultPlaceholderInfo specifying all the default
        placeholders of the model
    graph
        graph to retrieve inputs and outputs
    """
    def _get_shape(tensor):
        if tensor.shape == tf.TensorShape(None):
            return None
        return tensor.shape.as_list()

    graph = graph if graph is not None else tf.get_default_graph()
    inputs = tf_collections_utils.collection2nested(
        inputs_collection_name, graph=graph)
    outputs = tf_collections_utils.collection2nested(
        outputs_collection_names, graph=graph)
    outputs = nest_utils.flatten_nested_struct(outputs)
    input_output_nodes = dict()
    input_output_nodes['inputs'] = {
        each_input_key: {"name": each_input_value.name,
                         "shape": _get_shape(each_input_value),
                         "dtype": each_input_value.dtype.name}
        for each_input_key, each_input_value in inputs.items()}
    input_output_nodes['outputs'] = {
        each_output_key: {"name": each_output_value.name,
                          "shape": _get_shape(each_output_value),
                          "dtype": each_output_value.dtype.name}
        for each_output_key, each_output_value in outputs.items()}

    if parameter_placeholders:
        input_output_nodes['parameter_placeholders'] = {
            each_item.full_name: {
                "name": each_item.placeholder.name,
                "shape": _get_shape(each_item.placeholder),
                "default_value": each_item.default_value,
                "dtype": each_item.placeholder.dtype.name}
            for each_item in parameter_placeholders
        }
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(input_output_nodes, file, indent=4, sort_keys=True,
                  ensure_ascii=False)


def add_variables_from_graph_without_collection(graph: tf.Graph = None):
    """
    Add the variables to the GLOBAL_VARIABLES collection
    if no variables there exist, e.g. no restoring of the model is possible
    This may happen when you save the checkpoint, and seems to be a tensorflow
    bug

    Parameters
    ----------
    graph
        tensorflow graph

    """
    logger = logging.getLogger(__name__)
    logger.info("Add variables to GLOBAL_VARIABLES collection. It may "
                "be sign, that your checkpoint has no variables collection "
                "inside. Check it")
    graph = graph if graph is not None else tf.get_default_graph()
    all_operations = graph.get_operations()
    variable_operations = [v for v in all_operations if v.type == 'VariableV2']
    variables_ = [v.values() for v in variable_operations]
    variables = []
    global_variables = [v.name for v in
                        graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
    for each_variable in variables_:
        if isinstance(each_variable, (list, tuple)):
            variables.extend(each_variable)
        else:
            variables.append(each_variable)
    for each_variable in variables:
        if each_variable.name in global_variables:
            continue
        graph.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, each_variable)


def replace_outputs_with_named_identity(function: Callable[..., _NESTED_TYPE]
                                        ) -> Callable[..., _NESTED_TYPE]:
    """
    Replaces all the values inside of method outputs with identity tensor
    applied on these values with name equal to key of the value in the
    unflatten view, e.g. the name for value b inside {'a': {'b': 10}} will be
    'a//b'

    If the value is not a tf.Tensor, it will be passed as is

    Returns
    -------
    decorated_method
        decorated method
    """

    @wraps(function)
    def wrapped(self, *args, **kwargs):
        output = function(self, *args, **kwargs)
        result_with_identity = _replace_with_named_identity(output)
        return result_with_identity

    return wrapped


def _replace_with_named_identity(inputs: _NESTED_TYPE) -> _NESTED_TYPE:
    """
    Replaces all the values inside of inputs with identity tensor applied on
    these values with name equal to key of the value in the unflatten view,
    e.g. the name for value b inside {'a': {'b': 10}} will be 'a//b'

    If the value is not a tf.Tensor, it will be passed as is in the result

    Parameters
    ----------
    inputs
        possibly nested structure of tensors and other values

    Returns
    -------
    nested_dict_with_identity
        same structured as inputs, but using identity nodes for all tensor
        values with name as key name in unflatten view
    """
    result_with_identity = {}
    inputs_flatten = nest_utils.flatten_nested_struct(inputs)

    for each_key, each_value in inputs_flatten.items():
        if not isinstance(each_value, tf.Tensor):
            result_with_identity[each_key] = each_value
            continue
        identity_value = tf.identity(each_value, name=each_key)
        result_with_identity[each_key] = identity_value

    result_with_identity = nest_utils.unflatten_dict_to_nested(
        result_with_identity)
    return result_with_identity


def filter_variables_by_pattern(variables: List[tf.Variable],
                                pattern: str,
                                var_scope: Optional[str] = None
                                ) -> List[tf.Variable]:
    """
    Filter the variables which have pattern in the name

    Parameters
    ----------
    variables
        list of variables
    pattern
        pattern to look for
    var_scope
        variable scope name to remove from variable name before matching

    Returns
    -------
    filtered_variables
        filtered variables
    """
    match_names_to_vars = {each_var.name: each_var for each_var in variables}
    if var_scope:
        match_names_to_vars = {
            tf_varscopes_utils.remove_scope_from_name(
                each_name, var_scope): each_var
            for each_name, each_var in match_names_to_vars.items()}

    vars_filtered = [
        each_var for each_name, each_var in match_names_to_vars.items()
        if pattern in each_name]
    if not vars_filtered:
        msg = ("No variables with {} inside of name under var_scope {} "
               "are found inside {}").format(pattern, variables, var_scope)
        warnings.warn(msg, Warning)
    return vars_filtered
