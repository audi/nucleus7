# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from nucleus7.coordinator.buffer_callback import BufferCallback
from nucleus7.coordinator.configs import RunIterationInfo


class _DummySummarizerCallback(BufferCallback):
    exclude_from_register = True
    incoming_keys = ["data1", "data2"]

    def process_buffer(self, **buffer_data):
        data_mean = {k + "_mean": np.mean(v, 0) for k, v in buffer_data.items()}
        data_std = {k + "_std": np.std(v, 0) for k, v in buffer_data.items()}
        result = {**data_mean, **data_std}
        return result


class TestBufferCallback(tf.test.TestCase, parameterized.TestCase):

    def setUp(self):
        self.batch_size = 5
        self.num_batches = 3
        self.data = [{"data1": np.random.rand(self.batch_size, 5),
                      "data2": np.random.rand(self.batch_size, 3, 3)}
                     for _ in range(self.num_batches)]
        self.evaluate = [[0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0],
                         [0, 0, 1, 0, 0]]
        self.data_concat = {
            k: np.concatenate([each_batch[k] for each_batch in self.data], 0)
            for k in ["data1", "data2"]}

    @parameterized.parameters({"evaluate": False},
                              {"evaluate": True})
    def test_on_iteration_end(self, evaluate):
        buffer_callback = _DummySummarizerCallback(inbound_nodes=[])
        results = []
        for i, each_batch in enumerate(self.data):
            if i == self.num_batches - 1:
                buffer_callback.iteration_info = RunIterationInfo(
                    is_last_iteration=True)
            if evaluate:
                evaluate_batch = self.evaluate[i]
            else:
                evaluate_batch = None
            results.append(buffer_callback.on_iteration_end(
                evaluate=evaluate_batch, **each_batch))

        if not evaluate:
            result_last_must = self._get_result_must_in_interval()
            results_must = [None, None, result_last_must]
        else:
            result_last_list = [self._get_result_must_in_interval(7, 13),
                                self._get_result_must_in_interval(13, None)]
            result_last = {k: np.concatenate(
                [each_res[k] for each_res in result_last_list], 0)
                for k in result_last_list[0]}
            results_must = [None,
                            self._get_result_must_in_interval(0, 7),
                            result_last]

        for each_result, each_result_must in zip(results, results_must):
            if each_result_must is None:
                self.assertIsNone(each_result)
            else:
                self.assertAllClose(each_result,
                                    each_result_must)

    def _get_result_must_in_interval(self, start_ind=0, last_ind=None):
        interval = slice(start_ind, last_ind)
        data_mean = {
            k + "_mean": np.mean(self.data_concat[k][interval],
                                 0, keepdims=True)
            for k in ["data1", "data2"]
        }
        data_std = {
            k + "_std": np.std(self.data_concat[k][interval],
                               0, keepdims=True)
            for k in ["data1", "data2"]
        }
        result = {**data_mean, **data_std}
        return result
