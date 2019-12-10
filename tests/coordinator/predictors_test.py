# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from absl.testing import parameterized
import tensorflow as tf

from nucleus7.coordinator.predictors import predict_using_predictor
from nucleus7.coordinator.predictors import (
    represent_predictor_through_nucleotides)
from nucleus7.core.nucleotide import Nucleotide
from nucleus7.utils import nest_utils


class TestPredictors(parameterized.TestCase):

    def test_represent_predictor_through_nucleotides(self):
        class PredictorMock(object):
            def __init__(self_):
                self_._fetch_tensors = None

            @property
            def fetch_tensors(self_):
                return self_._fetch_tensors

            @property
            def feed_tensors(self_):
                return self_._feed_tensors

        tf.reset_default_graph()
        predictor = PredictorMock()
        fetch_tensors = {
            'nucleotide1': {'output1': 'value1',
                            'output2': 'value2'},
            'nucleotide2': {'output3': 'value3',
                            'output4': 'value4'}
        }
        feed_tensors = {
            "data1": tf.placeholder(tf.float32),
            "data2": tf.placeholder(tf.float32),
            "parameter1": tf.placeholder_with_default(10, [])
        }
        fetch_tensors_flatten = nest_utils.flatten_nested_struct(fetch_tensors)
        predictor._fetch_tensors = fetch_tensors_flatten
        predictor._feed_tensors = feed_tensors
        nucleotides = represent_predictor_through_nucleotides(predictor)
        nucleotide1_must = Nucleotide(name='nucleotide1')
        nucleotide1_must.generated_keys = ['output1', 'output2']
        nucleotide1_must.incoming_keys = ['data1', 'data2']
        nucleotide2_must = Nucleotide(name='nucleotide2')
        nucleotide2_must.generated_keys = ['output3', 'output4']
        nucleotide2_must.incoming_keys = ['data1', 'data2']
        nucleotides_must = [nucleotide1_must, nucleotide2_must]
        for nucleotide, nucleotide_must in zip(nucleotides, nucleotides_must):
            self.assertEqual(nucleotide_must.name,
                             nucleotide.name)
            self.assertSetEqual(set(nucleotide_must.generated_keys),
                                set(nucleotide.generated_keys))
            self.assertSetEqual(set(nucleotide_must.incoming_keys),
                                set(nucleotide.incoming_keys))

    @parameterized.parameters({"with_model_parameters": True},
                              {"with_model_parameters": False})
    def test_predict_using_predictor(self, with_model_parameters):
        class PredictorMock(object):
            def __init__(self_, fetch_tensors, feed_tensors):
                self_.fetch_tensors = fetch_tensors
                self_.feed_tensors = feed_tensors

            def __call__(self_, inputs: dict):
                return {k + '_out': v for k, v in inputs.items()}

        data = {'node1': {'out1': 10, 'out2': 20},
                'node2': {'out3': 30, 'out4': 40},
                'node3': {'out5': 50}}
        data_for_predictor = {k: v for k, v in data.items()
                              if k in ['node1', 'node2']}
        if with_model_parameters:
            model_parameters = {"nucleotide1": {"parameter1": 10},
                                "nucleotide3": {"parameter2": 20,
                                                "parameter3": [30, 40]}}
        else:
            model_parameters = None

        data_for_predictor_flatten = nest_utils.flatten_nested_struct(
            data_for_predictor, flatten_lists=False)
        feed_tensors = nest_utils.flatten_nested_struct(
            data_for_predictor, flatten_lists=False)
        predictor_out_flatten_must = {
            k + '_out': v for k, v in data_for_predictor_flatten.items()}

        predictor_out_must = nest_utils.unflatten_dict_to_nested(
            predictor_out_flatten_must)
        if with_model_parameters:
            predictor_out_must.update(
                {"nucleotide1": {"parameter1_out": 10},
                 "nucleotide3": {"parameter2_out": 20,
                                 "parameter3_out": [30, 40]}})

        predictor = PredictorMock(feed_tensors=feed_tensors,
                                  fetch_tensors=None)

        if with_model_parameters:
            result = predict_using_predictor(predictor, inputs=data,
                                             model_parameters=model_parameters)
        else:
            result = predict_using_predictor(predictor, inputs=data)
        self.assertDictEqual(predictor_out_must, result)

    def test_predictor_from_load_config(self):
        pass
