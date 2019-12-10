# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Predictors for inference
"""
import logging
from typing import List
from typing import Optional
from typing import Union

import tensorflow as tf
# pylint: disable=no-name-in-module
# is not part of the tensorflow API, but is needed here
from tensorflow.contrib.predictor.predictor import Predictor

from nucleus7.coordinator.configs import InferenceLoadConfig
from nucleus7.coordinator.configs import TensorrtConfig
from nucleus7.core.nucleotide import Nucleotide
from nucleus7.model.fields import CollectionNames
from nucleus7.model.fields import ScopeNames
from nucleus7.utils import nest_utils
from nucleus7.utils import tensorrt_utils
from nucleus7.utils import tf_collections_utils
from nucleus7.utils import tf_utils


# pylint: enable=no-name-in-module


class PredictorNucleotide(Nucleotide):  # pylint: disable=abstract-method
    """
    Class to use as predictor nucleotide.

    Is needed only to differentiate the nucleotides which are used as predictors
    """

    def __init__(self,
                 inbound_nodes: Optional[Union[list, dict, str]] = None,
                 name: str = "predictor"):
        super(PredictorNucleotide, self).__init__(
            inbound_nodes=inbound_nodes, name=name)


class _GraphAndCheckpointPredictor(Predictor):
    """
    Predictor from meta graph and checkpoint

    Parameters
    ----------
    meta_graph_path
        path to meta graph
    checkpoint_path
        path to checkpoint for meta graph
    graph
        graph to use
    config
        session run config

    """

    def __init__(self, meta_graph_path: str,
                 checkpoint_path: str,
                 graph: tf.Graph = None,
                 config: tf.ConfigProto = None):
        self._graph = graph or tf.Graph()
        with self._graph.as_default():
            tf.train.import_meta_graph(meta_graph_path, clear_devices=True)
            try:
                saver = tf.train.Saver()
            except ValueError:
                tf_utils.add_variables_from_graph_without_collection()
                saver = tf.train.Saver()
            self._session = tf.Session(config=config)
            saver.restore(self._session, checkpoint_path)
            inputs = tf_collections_utils.collection2nested(
                CollectionNames.INPUTS)
            predictions = tf_collections_utils.collection2nested(
                CollectionNames.PREDICTIONS)

        self._feed_tensors = nest_utils.flatten_nested_struct(inputs)
        self._fetch_tensors = nest_utils.flatten_nested_struct(predictions)


class _TensorrtPredictor(Predictor):
    """
    Predictor, that takes saved model and converts it to tensorrt and uses
    tensorrt graph for it.

    If it cannot use tensorrt, it will raise an error

    Parameters
    ----------
    saved_model_dir
        saved model directory
    config
        session run config
    tensorrt_config
        tensorrt config

    Raises
    ------
    ValueError
        if tensorrt cannot be imported
    """

    def __init__(self, saved_model_dir: str,
                 tensorrt_config: Optional[TensorrtConfig],
                 config: tf.ConfigProto = None):
        self.saved_model_dir = saved_model_dir
        input_tensors, output_tensors, trt_graph_def = (
            tensorrt_utils.convert_saved_model_to_tensorrt(
                saved_model_dir, tensorrt_config, session_config=config))
        tf.reset_default_graph()
        self._graph = tf.Graph()
        feed_tensors, fetch_tensors = self._import_graph_def(
            input_tensors, output_tensors, trt_graph_def)
        self._session = tf.Session(graph=self._graph, config=config)
        self._feed_tensors = feed_tensors
        self._fetch_tensors = fetch_tensors

    def _import_graph_def(self, input_tensors, output_tensors, trt_graph_def):
        # pylint: disable=not-context-manager
        # bug of pylint and/or tensorflow, since it is a context manager
        with self._graph.as_default():
            feed_tensors, input_map = self._create_feed_tensors_and_input_map(
                input_tensors)
            output_keys, output_tensors_list = zip(*output_tensors.items())
            output_tensor_names = [
                each_tensor.name for each_tensor in output_tensors_list]
            fetch_tensors_list = tf.import_graph_def(
                graph_def=trt_graph_def,
                input_map=input_map,
                return_elements=output_tensor_names
            )
            fetch_tensors = dict(zip(output_keys, fetch_tensors_list))
        return feed_tensors, fetch_tensors

    @staticmethod
    def _create_feed_tensors_and_input_map(input_tensors):
        input_placeholders = {
            each_key: tf.placeholder(each_tensor.dtype, shape=each_tensor.shape)
            for each_key, each_tensor in input_tensors.items()}
        input_map = {input_tensors[each_key].name: input_placeholders[each_key]
                     for each_key in input_tensors}
        return input_placeholders, input_map


def predictor_from_load_config(
        load_config: InferenceLoadConfig,
        tensorrt_config: Optional[TensorrtConfig] = None,
        session_config: Optional[tf.ConfigProto] = None,
        postprocessors_to_use: Optional[List[str]] = None,
        model_parameters: Optional[dict] = None) -> Predictor:
    """
    Create instance of :obj:`tf.contrib.predictor.Predictor` from load_config.

    If saved_model was provided and tensorrt_config.use_tensorrt, it will try
    to create Predictor using tensorrt. If constructor of tensorrt fails,
    saved model predictor will be used.

    Parameters
    ----------
    load_config
        load config
    tensorrt_config
        tensorrt config
    session_config
        session configuration
    postprocessors_to_use
        which postprocessors to use
    model_parameters
        model parameters to feed in nested view; only names will be used to get
        the placeholders and add to predictor feed_tensors

    Returns
    -------
    predictor
        predictor
    """
    logger = logging.getLogger(__name__)
    if load_config.saved_model:
        saved_model_path = load_config.saved_model
        if tensorrt_config and tensorrt_config.use_tensorrt:
            _predictor = _TensorrtPredictor(
                saved_model_dir=saved_model_path,
                tensorrt_config=tensorrt_config, config=session_config)
            logger.info("Using tensorrt predictor")
        else:
            _predictor = tf.contrib.predictor.from_saved_model(
                saved_model_path, config=session_config)
    else:
        if tensorrt_config and tensorrt_config.use_tensorrt:
            logger.warning(
                "currently tensorrt can be used only with saved_model")
        # create predictor from meta graph
        meta_graph_path = load_config.meta_graph
        checkpoint_path = load_config.checkpoint
        _predictor = _GraphAndCheckpointPredictor(
            meta_graph_path, checkpoint_path, config=session_config)
    if postprocessors_to_use:
        # pylint: disable=protected-access
        # there is no property setter and so only way to assign it
        fetch_tensors_unflatten = nest_utils.unflatten_dict_to_nested(
            _predictor._fetch_tensors)
        fetch_tensors_unflatten_filtered = {
            k: v for k, v in fetch_tensors_unflatten.items()
            if k in postprocessors_to_use
        }
        fetch_tensors_flatten_filtered = nest_utils.flatten_nested_struct(
            fetch_tensors_unflatten_filtered)
        _predictor._fetch_tensors = fetch_tensors_flatten_filtered

    if model_parameters:
        _predictor = _add_model_parameters_to_predictor(
            _predictor, model_parameters)
    return _predictor


def represent_predictor_through_nucleotides(
        predictor: Predictor,
        incoming_keys_mapping: Optional[dict] = None) -> List[Nucleotide]:
    """
    Represent predictor feed tensors structure as list of nucleotides

    When predictor has following fetch'_tensors structure but in flatten
    `{'postprocessor_nucleotide1': {'out1': ..., 'out2': ...},
    'postprocessor_nucleotide2': {'out3': ..., 'out4': ...}}`,
    then it will return list of nodes
    `[postprocessor_nucleotide1, postprocessor_nucleotide2]` with generated keys
    `['out1', 'out2']` and `['out3', 'out4']` respectively

    Parameters
    ----------
    predictor
        predictor with fetch_tensors property defined
    incoming_keys_mapping
        mapping of the incoming keys to the predictor

    Returns
    -------
    predictor_nucleotides
        list of nucleotides which mimic the data structure of fetched predictor
        tensors
    """
    fetch_tensors_flatten = predictor.fetch_tensors
    fetch_tensors = nest_utils.unflatten_dict_to_nested(fetch_tensors_flatten)
    feeded_tensors_without_parameters_flatten = {
        each_key: each_tensor
        for each_key, each_tensor in predictor.feed_tensors.items()
        if each_tensor.op.type != 'PlaceholderWithDefault'}
    feeded_tensors_without_parameters = nest_utils.unflatten_dict_to_nested(
        feeded_tensors_without_parameters_flatten)
    incoming_keys = sorted(list(feeded_tensors_without_parameters))

    predictor_nucleotides = []
    for each_nucleotide_name_key, each_nucleotide_outputs in sorted(
            fetch_tensors.items()):
        nucleotide = PredictorNucleotide(name=each_nucleotide_name_key,
                                         inbound_nodes="dataset")
        nucleotide.incoming_keys = incoming_keys
        if incoming_keys_mapping is not None:
            nucleotide.incoming_keys_mapping = {
                "dataset": incoming_keys_mapping}
        nucleotide.generated_keys = sorted(list(each_nucleotide_outputs))
        predictor_nucleotides.append(nucleotide)
    return predictor_nucleotides


@nest_utils.flatten_nested_inputs("inputs")
@nest_utils.unflatten_nested_outputs
def predict_using_predictor(predictor: Predictor, *, inputs: dict,
                            model_parameters: Optional[dict] = None) -> dict:
    """
    Make predictions using predictor

    Parameters
    ----------
    predictor
        predictor to use
    inputs
        inputs for predictor
    model_parameters
        model parameters to feed in nested view

    Returns
    -------
    predictions
        predictions
    """
    model_parameters = model_parameters or {}
    model_parameters_flat = nest_utils.flatten_nested_struct(
        model_parameters, flatten_lists=False)
    inputs_filtered = {k: v for k, v in inputs.items()
                       if k in predictor.feed_tensors}
    inputs_filtered.update(model_parameters_flat)
    predictions = predictor(inputs_filtered)
    return predictions


def _add_model_parameters_to_predictor(predictor: Predictor,
                                       model_parameters: dict) -> Predictor:
    logger = logging.getLogger(__name__)
    graph = predictor.graph
    model_parameters_flat = nest_utils.flatten_nested_struct(
        model_parameters, flatten_lists=False)
    model_parameters_placeholders = {}
    for each_parameter_name, each_value in model_parameters_flat.items():
        tensor = _get_tensor_by_parameter_name(graph, each_parameter_name)
        logger.info("Set parameter %s to %s (tensor name: %s)",
                    each_parameter_name, each_value, tensor.name)
        model_parameters_placeholders[each_parameter_name] = tensor

    # pylint: disable=protected-access
    # there is no property setter and so only way to assign it
    predictor._feed_tensors.update(model_parameters_placeholders)
    return predictor


def _get_tensor_by_parameter_name(graph: tf.Graph, parameter_name: str
                                  ) -> tf.Tensor:
    nucleotide_var_scopes_with_parameters = [
        ScopeNames.MODEL,
        ScopeNames.POSTPROCESSING,
    ]
    nucleotide_var_scope = parameter_name.split("//", maxsplit=1)[0]
    tensor_name = ":".join([parameter_name, "0"])
    tensor = None
    for each_var_scope in nucleotide_var_scopes_with_parameters:
        tensor_name_with_var_scope = "/".join(
            [each_var_scope, nucleotide_var_scope, tensor_name])
        try:
            tensor = graph.get_tensor_by_name(tensor_name_with_var_scope)
            break

        except KeyError:
            continue
    if tensor is None:
        raise ValueError(
            "Model parameter with name {} does not exist!".format(
                parameter_name))

    if tensor.op.type != 'PlaceholderWithDefault':
        msg = ("Model parameter tensor operation should be of type "
               "PlaceholderWithDefault! (found {})").format(tensor.op.type)
        raise ValueError(msg)
    return tensor
