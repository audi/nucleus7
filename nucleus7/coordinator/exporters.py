# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Exporters for Trainer
"""

import json
import logging
import os
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import tensorflow as tf

from nucleus7.coordinator import configs as coord_configs
from nucleus7.kpi.kpi_evaluator import KPIEvaluator
from nucleus7.model import Model
from nucleus7.model.fields import CollectionNames
from nucleus7.utils import io_utils
from nucleus7.utils import kpi_utils
from nucleus7.utils import mlflow_utils
from nucleus7.utils import nest_utils
from nucleus7.utils import project_utils
from nucleus7.utils import tf_utils
from nucleus7.utils.kpi_utils import filter_kpi_values


class ModelExporter(tf.estimator.LatestExporter):
    """
    Exporter for model

    If any kpi evaluator callbacks were provided, it will store it result inside
    of {evaluator_name}.json file to exports_dir

    If eval_result is not None, it will store it to {eval_result}.json inside of
    exports_dir; also it will create file with name global_step_{global_step}

    Parameters
    ----------
    exports_dir
        directory to save exports
    kpi_evaluators
        kpi evaluator
    """

    def __init__(self, name: str,
                 serving_input_receiver_fn: Callable,
                 exports_dir: str,
                 kpi_evaluators: Optional[List[KPIEvaluator]] = None,
                 assets_extra=None,
                 as_text=False, exports_to_keep: Optional[int] = None):
        # pylint: disable=too-many-arguments
        # not possible to have less arguments without more complexity
        super().__init__(name, serving_input_receiver_fn,
                         assets_extra, as_text, exports_to_keep)
        self._exports_dir = exports_dir
        self._kpi_evaluators = kpi_evaluators or []

    def export(self, estimator, export_path, checkpoint_path, eval_result,
               is_the_final_export):
        """
        Adds export of global step and KPI values to the saved_models directory

        overridden from :obj:`tf.estimator.LatestExporter`.
        See its documentation for more information
        """
        # pylint: disable=too-many-arguments
        # default signature from tensorflow
        saved_model_path = super().export(
            estimator, self._exports_dir, checkpoint_path, eval_result,
            is_the_final_export)
        # pylint: disable=no-value-for-parameter
        # saved_model_load_fn passed by patch
        global_step = eval_result.get('global_step', -1)
        mlflow_utils.log_saved_model(saved_model_path, global_step)
        if isinstance(saved_model_path, bytes):
            saved_model_path = saved_model_path.decode('utf-8')
        for each_evaluator in self._kpi_evaluators:
            save_fname = os.path.join(saved_model_path,
                                      each_evaluator.name + ".json")
            kpi_filtered, _ = kpi_utils.filter_kpi_values(
                each_evaluator.last_kpi)
            with open(save_fname, 'w') as file:
                json.dump(kpi_filtered, file, indent=2, sort_keys=True)

        if eval_result:
            project_utils.log_exported_model_info(
                os.path.realpath(saved_model_path), global_step)

            del eval_result['global_step']
            eval_result_filtered, _ = filter_kpi_values(eval_result, True)
            save_fname = os.path.join(saved_model_path, 'eval_result.json')
            with open(save_fname, 'w', encoding='utf-8') as file:
                json.dump(eval_result_filtered, file, indent=4, sort_keys=True,
                          ensure_ascii=False)
            _log_eval_result_to_mlflow(eval_result_filtered)
        return saved_model_path


class ServingInputReceiverFnBuilder:
    """
    Create serving_input_receiver_fn to be passed to Exporter.

    Inside of serving_input_receiver_fn on first run also
    `graph_inference.meta` file with meta graph for inference is generated
    together with `saved_models/input_output_names.json` file with names
    of inputs / outputs for it inside of saved_models/ directory.

    Result function has placeholders for all dataset items that connected
    to predictions. To understand, which inputs connect to predictions,
    it builds whole inference graph with all input keys and then stores
    only connected inputs to `saved_models/input_output_names.json` and
    resets the graph. Afterwards, if it reads input names from
    `saved_models/input_output_names.json` and puts only that inputs to
    serving_input_receiver_fn. By next run, it checks if that file already
    exists and if so, it just reads input names from it and does not build
    inference graph anymore.

    Parameters
    ----------
    model
        instance of model
    dataset_fn
        callable without arguments to create dataset
    save_dir_inference_graph
        path to save the inference graph
    save_dir_inputs_outputs_mapping
        path to save the inputs outputs mapping
    inference_inputs_have_variable_shape
        controls if the inputs inside of inference graph should have
        variable shapes; defaults to True
    """

    # pylint: disable=too-few-public-methods
    # serves more as a container and not as an interface

    def __init__(self,
                 model: Model,
                 save_dir_inference_graph: str,
                 save_dir_inputs_outputs_mapping: str,
                 dataset_fn: Callable[[], tf.data.Dataset],
                 inference_inputs_have_variable_shape: bool = False):
        self.model = model
        self.dataset_fn = dataset_fn
        self.save_dir_inputs_outputs_mapping = save_dir_inputs_outputs_mapping
        self.save_dir_inference_graph = save_dir_inference_graph
        self.inference_inputs_have_variable_shape = (
            inference_inputs_have_variable_shape)

    def get(self):
        """
        Returns
        -------
        serving_input_receiver_fn : callable
            serving_input_receiver_fn
        """
        return self._serving_input_receiver_fn

    def _serving_input_receiver_fn(self) -> callable:
        # pylint: disable=not-context-manager
        # bug of pylint and/or tensorflow, since it is a context manager
        with tf.Graph().as_default():
            data = self.dataset_fn()
        # pylint: enable=not-context-manager
        input_shapes, input_types = data.output_shapes, data.output_types

        shape_fn = (_get_undefined_shapes_except_last
                    if self.inference_inputs_have_variable_shape
                    else _get_defined_shapes_with_batch)

        input_output_fname = os.path.join(
            self.save_dir_inputs_outputs_mapping,
            coord_configs.INPUT_OUTPUT_NAMES_FILE_NAME)

        self._maybe_create_inference_graph_and_save_input_output(
            input_output_fname, input_shapes, input_types, shape_fn)

        input_names = io_utils.load_json(input_output_fname)['inputs']
        inputs = {k: tf.placeholder(dtype=input_types[k],
                                    shape=shape_fn(input_shapes[k]),
                                    name="input_{}".format(k))
                  for k in input_names}
        inputs_from_default_placeholders = {
            each_item.full_name: each_item.placeholder
            for each_item in self.model.default_placeholders}
        inputs.update(inputs_from_default_placeholders)
        return tf.estimator.export.ServingInputReceiver(inputs, inputs)

    def _maybe_create_inference_graph_and_save_input_output(
            self, input_output_fname: str, input_shapes, input_types,
            shape_fn):
        """
        Create inference graph and save input output mappings
        and afterwards reset the graph
        """
        if os.path.isfile(input_output_fname):
            return
        inference_graph = tf.Graph()
        self.model.reset_tf_graph()
        # pylint: disable=not-context-manager
        # bug of pylint and/or tensorflow, since it is a context manager
        with inference_graph.as_default():
            logger = logging.getLogger(__name__)
            features = {k: tf.placeholder(dtype=input_types[k],
                                          shape=shape_fn(input_shapes[k]),
                                          name="input_{}".format(k))
                        for k in input_shapes}
            logger.info('build inference graph and select connected inputs '
                        'to %s', input_output_fname)
            self.model.build_inference_graph(features)
            logger.info('export input-output node mappings '
                        'to %s', input_output_fname)
            tf_utils.save_input_output_node_names(
                input_output_fname,
                parameter_placeholders=self.model.default_placeholders,
                inputs_collection_name=CollectionNames.INPUTS,
                outputs_collection_names=CollectionNames.PREDICTIONS)

            inference_graph_fname = os.path.join(
                self.save_dir_inference_graph,
                coord_configs.INFERENCE_GRAPH_FILE_NAME)
            if not os.path.isfile(inference_graph_fname):
                logger.info('export inference graph to %s',
                            inference_graph_fname)
                tf.train.export_meta_graph(inference_graph_fname,
                                           graph=tf.get_default_graph())
        # pylint: enable=not-context-manager
        del inference_graph


def _get_inference_batch_size(tensor_shape: tf.TensorShape) -> Union[None, int]:
    """
    Returns batch size from tensor_shape.
    If tensor_shape[0] == None, returns None, otherwise returns 1
    """
    if not tensor_shape:
        return None
    if tensor_shape.as_list()[0] is None:
        return None
    return 1


def _get_undefined_shapes_except_last(tensor_shape) -> List[Union[int, None]]:
    """
    Modify all dimensions between batch dimension and last one to
    be None and change batch dimension to 1 or None
    """
    batch_size = _get_inference_batch_size(tensor_shape)
    shape = ([batch_size] +
             [None for _ in range(len(tensor_shape) - 2)] +
             tensor_shape.as_list()[1:][-1:])
    return shape


def _get_defined_shapes_with_batch(tensor_shape) -> List[Union[int, None]]:
    """
    Change batch dimension to 1 or None and leave other dimensions as is
    """
    batch_size = _get_inference_batch_size(tensor_shape)
    shape = [batch_size] + tensor_shape.as_list()[1:]
    return shape


def _log_eval_result_to_mlflow(eval_result_filtered: dict):
    eval_result_filtered_flatten = nest_utils.flatten_nested_struct(
        eval_result_filtered, separator="--")
    for each_eval_name, each_eval_value in (
            eval_result_filtered_flatten.items()):
        mlflow_utils.log_metric_to_mlflow(
            each_eval_name, each_eval_value)
