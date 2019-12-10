# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Utils to use tensorrt
"""

from typing import Dict
from typing import Optional
from typing import Tuple

import tensorflow as tf

try:
    from tensorflow.contrib import tensorrt as trt
except (ImportError, tf.errors.NotFoundError):
    trt = None

from nucleus7.utils import tf_utils
from nucleus7.coordinator.configs import TensorrtConfig


def convert_saved_model_to_tensorrt(
        saved_model_dir: str,
        tensorrt_config: TensorrtConfig = None,
        session_config: Optional[tf.ConfigProto] = None
) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor], tf.GraphDef]:
    """
    Convert saved model to tensorrt.

    Uses default tag and signature_def

    Parameters
    ----------
    saved_model_dir
        directory with saved model inside
    tensorrt_config
        tensorrt config which holds all the tensorrt parameters
    session_config
        session config to use

    Returns
    -------
    input_tensors
        dict holding input tensors from saved model signature_def
    output_tensors
        dict holding output tensors from saved model signature_def
    trt_graph
        graph_def with tensorrt graph with variables

    Raises
    ------
    ValueError
        if tensorrt import was unsuccessful
    """
    if trt is None:
        raise ImportError(
            "No tensorrt is found under tensorflow.contrib.tensorrt")
    tensorrt_kwargs = (
        tensorrt_config._asdict() if tensorrt_config is not None else {})
    tensorrt_kwargs.pop("use_tensorrt", None)
    (input_tensors, output_tensors, frozen_graph_def
     ) = _load_saved_model_as_frozen_graph(saved_model_dir)
    output_tensors_list = list(output_tensors.values())

    trt_graph = trt.create_inference_graph(
        input_graph_def=frozen_graph_def,
        outputs=output_tensors_list,
        session_config=session_config,
        **tensorrt_kwargs)
    return input_tensors, output_tensors, trt_graph


def _load_saved_model_as_frozen_graph(
        saved_model_dir: str
) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor], tf.GraphDef]:
    tag = tf.saved_model.tag_constants.SERVING
    signature_def_tag = (
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        meta_graph_def = tf.saved_model.loader.load(
            sess, [tag],
            export_dir=saved_model_dir, clear_devices=True
        )
        inputs = meta_graph_def.signature_def[signature_def_tag].inputs
        outputs = meta_graph_def.signature_def[signature_def_tag].outputs
        output_names = [
            tf_utils.remove_tag_from_variable_name(each_node.name)
            for each_node in outputs.values()]
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), output_names)
    input_tensors = {
        k: graph.get_tensor_by_name(v.name) for k, v in inputs.items()}
    output_tensors = {
        k: graph.get_tensor_by_name(v.name) for k, v in outputs.items()}
    return input_tensors, output_tensors, frozen_graph_def
