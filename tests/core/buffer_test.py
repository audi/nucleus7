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

from nucleus7.core.buffer import AdditiveBuffer
from nucleus7.core.buffer import AverageBuffer
from nucleus7.core.buffer import BufferProcessor
from nucleus7.core.buffer import SamplesBuffer


class TestSamplesBuffer(tf.test.TestCase):

    def setUp(self):
        self.sample1 = {"data1": 10, "data2": {"sub1": 2, "sub2": 3}}
        self.sample2 = {"data1": 3, "data2": {"sub1": 5, "sub2": 4}}

    def test_main(self):
        buffer = SamplesBuffer()
        buffer.add(**self.sample1)
        buffer.add(**self.sample2)
        buffer_must = {"data1": [10, 3],
                       "data2": {"sub1": [2, 5],
                                 "sub2": [3, 4]}}
        self.assertAllClose(buffer_must,
                            buffer.get())
        self.assertAllClose(buffer[0],
                            self.sample1)
        self.assertAllClose(buffer[1],
                            self.sample2)
        self.assertEqual(2,
                         len(buffer))
        buffer_flat_must = {"data1": [10, 3],
                            "data2//sub1": [2, 5],
                            "data2//sub2": [3, 4]}
        self.assertEqual(buffer_flat_must,
                         buffer.get_flat())
        del buffer[-1]
        self.assertEqual(1,
                         len(buffer))
        buffer1_must = {"data1": [10],
                        "data2": {"sub1": [2],
                                  "sub2": [3]}}
        self.assertAllClose(buffer1_must,
                            buffer.get())
        buffer.clear()
        self.assertEqual(0,
                         len(buffer))
        self.assertDictEqual({},
                             buffer.get())
        self.assertDictEqual({},
                             buffer.get_flat())

    def test_add_with_new_keys(self):
        sample3 = {"data1": 6, "data2": {"sub1": 50, "sub3": 1}, "data3": 4}
        buffer = SamplesBuffer()
        buffer.add(**self.sample1)
        buffer.add(**self.sample2)
        buffer.add(**sample3)
        buffer_must = {"data1": [10, 3, 6],
                       "data2": {"sub1": [2, 5, 50],
                                 "sub2": [3, 4, None],
                                 "sub3": [None, None, 1]},
                       "data3": [None, None, 4]}
        self.assertAllEqual(buffer_must,
                            buffer.get())


class TestAdditiveBuffer(tf.test.TestCase):

    def setUp(self):
        self.sample1 = {"data1": 10, "data2": {"sub1": 2, "sub2": 3}}
        self.sample2 = {"data1": 3, "data2": {"sub1": 5, "sub2": 4}}
        self.sample3 = {"data1": 1, "data2": {"sub1": 10, "sub2": 40}}

    def test_main(self):
        buffer = AdditiveBuffer()
        buffer.add(**self.sample1)
        self.assertEqual(1,
                         len(buffer))
        self.assertAllClose(self.sample1,
                            buffer.get())

        buffer.add(**self.sample2)
        buffer_must = {"data1": 13,
                       "data2": {"sub1": 7,
                                 "sub2": 7}}
        self.assertEqual(2,
                         len(buffer))
        self.assertAllClose(buffer_must,
                            buffer.get())
        buffer.add(**self.sample3)
        buffer_must = {"data1": 14,
                       "data2": {"sub1": 17,
                                 "sub2": 47}}
        self.assertAllClose(buffer_must,
                            buffer.get())
        self.assertEqual(3,
                         len(buffer))

        buffer.clear()
        self.assertEqual(0,
                         len(buffer))

    def test_wrong_samples(self):
        buffer = AdditiveBuffer()
        buffer.add(**self.sample1)
        with self.assertRaises(ValueError):
            buffer.add(data2={"sub1": 5, "sub2": 4})

        with self.assertRaises(ValueError):
            buffer.add(data1=5, data2={"sub1": 5, "sub2": 4, "sub3": 5})

        with self.assertRaises(ValueError):
            buffer.add(data1=5, data2={"sub1": 5, "sub2": np.ndarray([5, 5])})


class TestAverageBuffer(tf.test.TestCase):

    def setUp(self):
        self.sample1 = {"data1": 10, "data2": {"sub1": 2, "sub2": 3}}
        self.sample2 = {"data1": 3, "data2": {"sub1": 5, "sub2": 4}}
        self.sample3 = {"data1": 1, "data2": {"sub1": 10, "sub2": 40}}

    def test_main(self):
        buffer = AverageBuffer()
        buffer.add(**self.sample1)
        self.assertEqual(1,
                         len(buffer))
        self.assertAllClose(self.sample1,
                            buffer.get())

        buffer.add(**self.sample2)
        buffer_must = {"data1": 13 / 2,
                       "data2": {"sub1": 7 / 2,
                                 "sub2": 7 / 2}}
        self.assertEqual(2,
                         len(buffer))
        self.assertAllClose(buffer_must,
                            buffer.get())
        buffer.add(**self.sample3)
        buffer_must = {"data1": 14 / 3,
                       "data2": {"sub1": 17 / 3,
                                 "sub2": 47 / 3}}
        self.assertAllClose(buffer_must,
                            buffer.get())
        self.assertEqual(3,
                         len(buffer))

        buffer.clear()
        self.assertEqual(0,
                         len(buffer))


class TestBufferProcessor(parameterized.TestCase, tf.test.TestCase):

    def setUp(self):
        def _process_buffer(data1, data2):
            mean_data1 = np.mean(data1)
            mean_data2 = np.mean(data2)
            return {"mean_data12": (mean_data1 + mean_data2) / 2}

        self.process_buffer_fn = _process_buffer

    @parameterized.parameters({"not_batch_keys": ["data3"]},
                              {"not_batch_keys": None})
    def test_split_batch_to_samples(self, not_batch_keys):
        batch_size = 10
        processor = BufferProcessor(None, not_batch_keys=not_batch_keys)
        data = {"data1": {"sub1": np.arange(batch_size),
                          "sub2": np.arange(batch_size, batch_size * 2)},
                "data2": np.arange(batch_size * 2, batch_size * 3),
                "data3": np.arange(batch_size * 3, batch_size * 4)}
        sample_inputs = processor.split_batch_to_samples(**data)
        result_must = []
        for i in range(batch_size):
            result_sample_i = {
                "data2": data["data2"][i],
                "data1": {k: v[i] for k, v in data["data1"].items()}}
            if not_batch_keys:
                result_sample_i["data3"] = data["data3"]
            else:
                result_sample_i["data3"] = data["data3"][i]
            result_must.append(result_sample_i)
        self.assertAllClose(result_must,
                            sample_inputs)

    def test_combine_samples_to_batch(self):
        processor = BufferProcessor(None)
        list_of_sample_results = [None, None]
        self.assertIsNone(processor.combine_samples_to_batch(
            list_of_sample_results))

        sample1 = {"data1": 10, "data2": {"sub1": 2, "sub2": 3}}
        sample2 = {"data1": 3, "data2": {"sub1": 5, "sub2": 4}}

        list_of_sample_results = [None, sample1]
        result_must = {"data1": np.array([10]),
                       "data2": {"sub1": np.array([2]),
                                 "sub2": np.array([3])}}
        result = processor.combine_samples_to_batch(list_of_sample_results)
        self.assertAllClose(result_must,
                            result)

        list_of_sample_results = [None, sample1, sample2]
        result_must = {"data1": np.array([10, 3]),
                       "data2": {"sub1": np.array([2, 5]),
                                 "sub2": np.array([3, 4])}}
        result = processor.combine_samples_to_batch(list_of_sample_results)
        self.assertAllClose(result_must,
                            result)

    @parameterized.parameters({"clear_buffer_after_evaluate": True},
                              {"clear_buffer_after_evaluate": False})
    def test_process_single_sample(self, clear_buffer_after_evaluate):
        samples = [{"data1": np.random.rand(2), "data2": np.random.rand(10)}
                   for _ in range(10)]
        evaluate = [False] * 10
        evaluate[1] = True
        evaluate[6] = True
        processor = BufferProcessor(
            self.process_buffer_fn,
            clear_buffer_after_evaluate=clear_buffer_after_evaluate)
        last_acc_ind = 0
        for sample_i, (evaluate_sample, sample_data) in enumerate(
                zip(evaluate, samples)):
            result = processor.process_single_sample(
                evaluate=evaluate_sample,
                **sample_data)
            if evaluate_sample:
                accumulated_samples_must = samples[last_acc_ind: sample_i + 1]
                data1_mean = np.mean([each_sample["data1"] for each_sample in
                                      accumulated_samples_must])
                data2_mean = np.mean([each_sample["data2"] for each_sample in
                                      accumulated_samples_must])
                result_must = {"mean_data12": (data1_mean + data2_mean) / 2}
            else:
                result_must = None
            if result_must is None:
                self.assertIsNone(result)
            else:
                self.assertAllClose(result_must,
                                    result)
            if evaluate_sample and clear_buffer_after_evaluate:
                last_acc_ind = sample_i + 1
        buffer_last_must = {k: [each_sample[k]
                                for each_sample in samples[last_acc_ind:]]
                            for k in ["data1", "data2"]}
        self.assertAllClose(buffer_last_must,
                            processor.buffer.get())

    def test_process_batch(self):
        batch_size1 = 5
        batch_size2 = 10
        evaluate1 = np.zeros([batch_size1])
        evaluate2 = np.zeros([batch_size2])
        evaluate2[0] = 1
        evaluate2[5] = 1
        data1 = {"data1": np.random.rand(batch_size1, 2) * 2,
                 "data2": np.random.rand(batch_size1, 5) * 3,
                 "evaluate": evaluate1}
        data2 = {"data1": np.random.rand(batch_size2, 2) * 4,
                 "data2": np.random.rand(batch_size2, 5) * 5,
                 "evaluate": evaluate2}

        processor = BufferProcessor(self.process_buffer_fn)
        result1 = processor.process_batch(**data1)
        result2 = processor.process_batch(**data2)
        data_concat = {k: np.concatenate([data1[k], data2[k]], 0)
                       for k in data1}
        slices = [slice(0, batch_size1 + 1),
                  slice(batch_size1 + 1, batch_size1 + 6)]
        data_mean = [{k: np.mean(data_concat[k][each_slice])
                      for k in ["data1", "data2"]}
                     for each_slice in slices]
        result2_must = {
            "mean_data12": np.array([np.mean(list(each_data_mean.values()))
                                     for each_data_mean in data_mean])}
        buffer_last_must = {k: data_concat[k][batch_size1 + 6:]
                            for k in ["data1", "data2"]}
        self.assertIsNone(result1)
        self.assertAllClose(result2_must,
                            result2)
        self.assertAllClose(buffer_last_must,
                            processor.buffer.get())
