# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Dummy and mock model parts
"""
from typing import Dict
from typing import List

import numpy as np
import tensorflow as tf

from nucleus7.data import Dataset
from nucleus7.data import FileList
from nucleus7.data.processor import RandomAugmentationTf
from nucleus7.data.reader import DataReader
from nucleus7.data.reader import TfRecordsDataReader
from nucleus7.kpi import KPIAccumulator
from nucleus7.kpi import KPIPlugin
from nucleus7.model import Model
from nucleus7.model import ModelLoss
from nucleus7.model import ModelMetric
from nucleus7.model import ModelPlugin
from nucleus7.model import ModelPostProcessor
from nucleus7.model import ModelSummary


# pylint: disable=missing-docstring,abstract-method
# are test dummies, so docstring is not needed
# not all abstract methods must be overridden

class DummyPluginCNN(ModelPlugin):
    exclude_from_register = True
    incoming_keys = ['inputs_cnn']
    generated_keys = ['predictions']

    def __init__(self, **kwargs):
        super(DummyPluginCNN, self).__init__(**kwargs)

    def predict(self, inputs_cnn) -> dict:
        """
        Parameters
        ----------
        inputs_cnn : tensor_like
            inputs to convolution

        Returns
        -------
        predictions : tensor_like
            predictions after convolution
        """
        # pylint: disable=arguments-differ
        # super predict has more generic signature
        return {'predictions': tf.layers.conv2d(
            inputs_cnn, 10, 3, padding='same',
            activation=self.activation)}


class DummyPluginFlatten(ModelPlugin):
    """
    Attributes
    ----------
    incoming_keys : list
        * inputs_flatten : inputs to flatten
    generated_keys : list
        predictions : flattened inputs
    """
    exclude_from_register = True
    incoming_keys = ['inputs_flatten']
    generated_keys = ['predictions']

    def __init__(self, *args, **kwargs):
        super(DummyPluginFlatten, self).__init__(*args, **kwargs)

    def predict(self, inputs_flatten):
        # pylint: disable=arguments-differ
        # super predict has more generic signature
        return {'predictions': tf.layers.flatten(inputs_flatten)}


class DummyPluginMLP(ModelPlugin):
    exclude_from_register = True
    incoming_keys = ['inputs_mlp']
    generated_keys = ['predictions']

    def __init__(self, *args, **kwargs):
        super(DummyPluginMLP, self).__init__(*args, **kwargs)
        self.num_classes = kwargs.get('num_classes', 10)

    def predict(self, inputs_mlp):
        """
        Parameters
        ----------
        inputs_mlp : tensor_like
            inputs to dense

        Returns
        -------
        predictions : tensor_like
            inputs after dense layer
        """
        # pylint: disable=arguments-differ
        # super predict has more generic signature
        return {'predictions': tf.layers.dense(inputs_mlp, self.num_classes)}


class DummyPluginMLPKeras(ModelPlugin):
    exclude_from_register = True
    incoming_keys = ['inputs_mlp']
    generated_keys = ['predictions']

    def __init__(self, *args, **kwargs):
        super(DummyPluginMLPKeras, self).__init__(*args, **kwargs)
        self.num_classes = kwargs.get('num_classes', 10)
        self.dense_layer = None  # type: tf.keras.layers.Dense

    def build(self):
        super(DummyPluginMLPKeras, self).build()
        dense_layer = tf.keras.layers.Dense(self.num_classes)
        self.dense_layer = self.add_keras_layer(dense_layer)
        return self

    def predict(self, inputs_mlp):
        """
        Parameters
        ----------
        inputs_mlp : tensor_like
            inputs to dense

        Returns
        -------
        predictions : tensor_like
            inputs after dense layer
        """
        # pylint: disable=arguments-differ
        # super predict has more generic signature
        second_dense_layer = self.add_keras_layer(
            tf.keras.layers.Dense(self.num_classes), "second_dense_layer")
        first_layer_output = self.dense_layer(inputs_mlp)
        predictions = second_dense_layer(first_layer_output)
        return {'predictions': predictions}


class DummyPluginWithDummyParameter(ModelPlugin):
    exclude_from_register = True
    incoming_keys = ['inputs_mlp']
    generated_keys = ['predictions']

    def __init__(self, dummy_parameter, *args, **kwargs):
        super(DummyPluginWithDummyParameter, self).__init__(*args, **kwargs)
        self.num_classes = kwargs.get('num_classes', 10)
        self.dummy_parameter = dummy_parameter

    def predict(self, is_training, inputs_mlp):
        """
        Parameters
        ----------
        inputs_mlp : tensor_like
            inputs to dense

        Returns
        -------
        predictions : tensor_like
            inputs after dense layer
        """
        # pylint: disable=arguments-differ
        # super predict has more generic signature
        # pylint: disable=unused-argument
        # is_training is there because of interface signature
        return {'predictions': tf.layers.dense(inputs_mlp, self.num_classes)}


class DummyPlugin2Layers(ModelPlugin):
    exclude_from_register = True
    incoming_keys = ['inputs']
    generated_keys = ['outputs']

    def __init__(self, *args, second_layer_trainable=True, **kwargs):
        # pylint: disable=arguments-differ
        # super __init__ has more generic signature
        super(DummyPlugin2Layers, self).__init__(*args, **kwargs)
        self.second_layer_trainable = second_layer_trainable
        self.layer1 = None
        self.layer2 = None

    def create_keras_layers(self):
        self.layer1 = self.add_keras_layer(tf.keras.layers.Dense(5))
        self.layer2 = self.add_keras_layer(
            tf.keras.layers.Dense(5, trainable=self.second_layer_trainable))

    def predict(self, inputs):
        # pylint: disable=arguments-differ
        # super predict has more generic signature
        return {'outputs': self.layer2(self.layer1(inputs))}


class DummySoftmaxLoss(ModelLoss):
    exclude_from_register = True
    incoming_keys = ['labels', 'logits']
    generated_keys = ['loss']

    def __init__(self, *args, **kwargs):
        super(DummySoftmaxLoss, self).__init__(*args, **kwargs)

    def process(self, labels, logits):
        # pylint: disable=arguments-differ
        # super predict has more generic signature
        return {'loss': tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)}


class DummyPostProcessor(ModelPostProcessor):
    exclude_from_register = True
    incoming_keys = ['predictions']
    generated_keys = ['predictions_pp']

    def process(self, predictions):
        # pylint: disable=arguments-differ
        # super predict has more generic signature
        return {'predictions_pp': tf.identity(predictions)}


class DummySummary(ModelSummary):
    exclude_from_register = True
    incoming_keys = ['labels', 'predictions']
    generated_keys = ['image_predictions_class',
                      'image_labels_class']

    def process(self, labels, predictions):
        # pylint: disable=arguments-differ
        # super predict has more generic signature
        return {'image_predictions_class': tf.argmax(predictions, -1),
                'image_labels_class': tf.identity(labels)}


class DummyMetric(ModelMetric):
    exclude_from_register = True
    incoming_keys = ['labels', 'predictions']
    generated_keys = ['metric']

    def process(self, labels, predictions):
        # pylint: disable=arguments-differ
        # super predict has more generic signature
        return {'metric': tf.reduce_mean(
            tf.to_float(labels) - predictions)}


class DummyTpFpTnFnKPIPlugin(KPIPlugin):
    """
    Plugin to calculate true and false positives and also true and false
    negatives
    """
    exclude_from_register = True
    incoming_keys = [
        "labels",
        "predictions"
    ]
    generated_keys = [
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
    ]

    def process(self, labels, predictions):
        # pylint: disable=arguments-differ
        # super evaluate_sample has more generic signature
        (true_positives, false_positives, true_negatives, false_negatives
         ) = 0, 0, 0, 0
        if labels == 1:
            true_positives += (labels == predictions)
            false_negatives += (labels != predictions)
        if labels == 0:
            true_negatives += (labels == predictions)
            false_positives += (labels != predictions)
        return {"true_positives": true_positives,
                "false_positives": false_positives,
                "true_negatives": true_negatives,
                "false_negatives": false_negatives}


class DummyF1KPIAccumulator(KPIAccumulator):
    """
    Generates F1 score out of true positives, false positives and
    false negatives
    """
    exclude_from_register = True
    incoming_keys = [
        "true_positives",
        "false_positives",
        "false_negatives",
    ]
    generated_keys = [
        "f1_score",
        "precision",
        "recall",
    ]

    def process(self, true_positives, false_positives, false_negatives):
        # pylint: disable=arguments-differ
        # super evaluate_sample has more generic signature
        true_positives_sum = np.sum(true_positives)
        false_positives_sum = np.sum(false_positives)
        false_negatives_sum = np.sum(false_negatives)
        precision = (true_positives_sum
                     / (true_positives_sum + false_positives_sum))
        recall = (true_positives_sum
                  / (true_positives_sum + false_negatives_sum))
        f1_score = 2 * precision * recall / (precision + recall)
        kpi = {'precision': precision,
               'recall': recall,
               'f1_score': f1_score}
        return kpi


class DummyRandomAugmentationTf(RandomAugmentationTf):
    exclude_from_register = True
    incoming_keys = ["data"]
    generated_keys = ["data"]
    random_variables_keys = ["noise"]

    def create_random_variables(self) -> Dict[str, tf.Tensor]:
        noise = tf.random_uniform([], 0, 1, seed=self.get_random_seed())
        return {"noise": noise}

    def augment(self, *, data) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # super method has more generic signature
        return {"data": data + self.random_variables["noise"]}


class FileListDummy(FileList):
    exclude_from_register = True

    def __init__(self,
                 file_names: Dict[str, List[str]],
                 **kwargs):
        super().__init__(file_names, **kwargs)

    def match(self):
        return


class DatasetDummy(Dataset):
    exclude_from_register = True

    def __init__(self, data, **kwargs):
        self.data = data
        super().__init__(**kwargs)

    def create_initial_data(self):
        return tf.data.Dataset.from_tensor_slices(self.data)


class ModelMock(Model):
    exclude_from_register = True

    def __init__(self, num_classes, **kwargs):
        super().__init__(plugins=[], losses=[], **kwargs)
        self.num_classes = num_classes
        self.keras_dense_layer = tf.keras.layers.Dense(self.num_classes)
        self.trainable_vars = []

    def __call__(self, inputs_from_dataset):
        results = super(ModelMock, self).__call__(inputs_from_dataset)
        self.trainable_vars = self.trainable_vars or tf.trainable_variables()
        return results

    @property
    def trainable_variables(self) -> List[tf.Variable]:
        return tf.trainable_variables()

    def reset_tf_graph(self):
        self.keras_dense_layer.built = False

    def forward_pass(self, *, inputs_from_dataset: Dict[str, tf.Tensor]
                     ) -> Dict[str, tf.Tensor]:
        data = inputs_from_dataset['dataset']['data']
        result = tf.layers.dense(data, units=10, activation=tf.nn.relu)
        logits = self.keras_dense_layer(result)
        return {'predictions_raw': logits}

    def postprocess_predictions(self, *,
                                inputs_from_dataset: Dict[str, tf.Tensor],
                                predictions_raw: Dict[str, tf.Tensor]):
        # pylint: disable=unused-argument
        logits = predictions_raw['predictions_raw']
        classes = {'classes': tf.argmax(logits, -1)}
        return classes

    def calculate_losses(self, *,
                         inputs_from_dataset: Dict[str, tf.Tensor],
                         predictions_raw: Dict[str, tf.Tensor]):
        # pylint: disable=unused-argument
        labels = inputs_from_dataset['dataset']['labels']
        logits = predictions_raw['predictions_raw']
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits)
        losses = {'loss': loss,
                  'total_loss': loss}
        losses = self._add_regularization(losses)
        return losses

    def get_summaries(self, *,
                      inputs_from_dataset: Dict[str, tf.Tensor],
                      predictions_raw: Dict[str, tf.Tensor],
                      predictions: Dict[str, tf.Tensor]):
        # pylint: disable=unused-argument
        labels = inputs_from_dataset['dataset']['labels']
        pred_class = predictions['classes']
        summary = {'scalar_labels': labels[0],
                   'scalar_classes': pred_class[0]}
        return summary

    def get_metrics(self, *,
                    inputs_from_dataset: Dict[str, tf.Tensor],
                    predictions_raw: Dict[str, tf.Tensor],
                    predictions: Dict[str, tf.Tensor]):
        # pylint: disable=unused-argument
        metrics = {'metric': tf.constant([0])}
        return metrics

    def get_test_inputs(self, batch_size, data_dim):
        inputs_np = {
            'data': np.random.randn(batch_size, data_dim).astype(
                np.float32),
            'labels': np.random.randint(
                0, self.num_classes, [batch_size], dtype=np.int64),
            'temp': np.ones(batch_size, np.float32)}
        inputs_tf = {k: tf.constant(v) for k, v in inputs_np.items()}
        return inputs_np, inputs_tf


class DataReaderDummyNP(DataReader):
    """
    transforms file_list with keys {data1, data2} by casting its values to float
    """
    exclude_from_register = True
    file_list_keys = ["data1", "data2"]
    generated_keys = ["data1", "data2"]

    def read(self, **data):
        return {k: np.float32(v) for k, v in data.items()}


class DataReaderDummyTF(DataReader):
    """
    transforms file_list with keys {data1, data2} by casting its values to float
    """
    exclude_from_register = True
    is_tensorflow = True
    file_list_keys = ["data1", "data2"]
    generated_keys = ["data1", "data2"]

    def read(self, **data):
        return {k: tf.string_to_number(v, tf.float32) for k, v in data.items()}


class TfRecordsDataReaderDummy(TfRecordsDataReader):
    """
    decodes data from tfrecords with following shapes:
        {"data1": [1],
         "data2": [None, 20],
         "data3": [None, 1],
         "data_default": [1]}
    """
    exclude_from_register = True
    generated_keys = ["data1", "data2", "data3", "data_default"]

    def get_tfrecords_features(self):
        keys = ["data1", "data2", "data3", "data_default"]
        return {k: tf.FixedLenFeature((), tf.string,
                                      np.zeros([1], np.float32).tostring())
                for k in keys}

    def get_tfrecords_output_types(self):
        keys = ["data1", "data2", "data3", "data_default"]
        return {k: tf.float32 for k in keys}

    def postprocess_tfrecords(self, data1, data2, data3, data_default):
        # pylint: disable=arguments-differ
        # super method has more generic signature
        return {
            "data1": tf.reshape(data1, [1]),
            "data2": tf.reshape(data2, [-1, 20]),
            "data3": tf.reshape(data3, [-1, 1]),
            "data_default": tf.reshape(data_default, [1]),
        }
