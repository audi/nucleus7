# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import os
import random
from unittest.mock import MagicMock
from unittest.mock import call as mock_call

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from nucleus7.data.data_pipe import DataPipe
from nucleus7.data.data_pipe import DataPipeMixin
from nucleus7.data.processor import DataProcessor
from nucleus7.data.reader import TfRecordsDataReader
from nucleus7.test_utils.model_dummies import DataReaderDummyNP
from nucleus7.test_utils.model_dummies import DataReaderDummyTF
from nucleus7.test_utils.model_dummies import TfRecordsDataReaderDummy
from nucleus7.test_utils.test_utils import write_tf_records


class _DataProcessorNP(DataProcessor):
    exclude_from_register = True
    incoming_keys = ["data1"]
    generated_keys = ["data1_p"]

    def process(self, data1):
        return {"data1_p": data1 + 20}


class _DataProcessorTF(DataProcessor):
    exclude_from_register = True
    incoming_keys = ["data1"]
    generated_keys = ["data1_p"]
    is_tensorflow = True

    def process(self, data1):
        return {"data1_p": data1 + 20}


class _DataProcessorNP2(DataProcessor):
    exclude_from_register = True
    incoming_keys = ["data2"]
    generated_keys = ["data2_p"]

    def process(self, data2):
        return {"data2_p": data2 + 200}


class _DataProcessorTF2(DataProcessor):
    exclude_from_register = True
    incoming_keys = ["data2"]
    generated_keys = ["data2_p"]
    is_tensorflow = True

    def process(self, data2):
        return {"data2_p": data2 + 200}


class TestDataPipe(tf.test.TestCase):

    def setUp(self):
        self.num_samples = 10

        self.file_list_with_floats = [
            {"data1_fl": "%05d" % random.randint(0, 100),
             "data2_fl": "%d" % random.randint(10, 40),
             "data3_fl": "%08d" % random.randint(100, 1000)}
            for _ in range(self.num_samples)]
        self.tfrecords_file_name = os.path.join(
            self.get_temp_dir(), "data.tfrecords")
        self.data_tfrecords = [
            {"data1": np.random.rand(1).astype(np.float32),
             "data2": np.random.rand(
                 np.random.randint(1, 5), 20).astype(np.float32),
             "data3": np.random.rand(
                 np.random.randint(1, 10), 1).astype(np.float32)}
            for _ in range(self.num_samples)]

    def test_build_mixed_types(self):
        reader_np = DataReaderDummyNP().build()
        processor_np = _DataProcessorNP().build()
        reader_tf = DataReaderDummyTF().build()
        processor_tf = _DataProcessorTF().build()
        data_pipe = DataPipe(readers=reader_np, processors=processor_tf)
        with self.assertRaises(ValueError):
            data_pipe.build()
        data_pipe = DataPipe(readers=reader_tf, processors=processor_np)
        with self.assertRaises(ValueError):
            data_pipe.build()
        data_pipe = DataPipe(readers=reader_tf, processors=processor_tf)
        data_pipe.build()
        self.assertTrue(data_pipe.is_tensorflow_pipe)

        data_pipe = DataPipe(readers=reader_np, processors=processor_np)
        data_pipe.build()
        self.assertFalse(data_pipe.is_tensorflow_pipe)

    def test_build_tfrecords_readers(self):
        reader_tf = DataReaderDummyTF().build()
        reader_tfrecords = TfRecordsDataReader().build()
        data_pipe = DataPipe(readers=[reader_tf, reader_tfrecords])
        with self.assertRaises(ValueError):
            data_pipe.build()
        data_pipe = DataPipe(readers=[reader_tfrecords])
        data_pipe.build()
        self.assertTrue(data_pipe.read_from_tfrecords)
        data_pipe = DataPipe(readers=[reader_tf])
        data_pipe.build()
        self.assertFalse(data_pipe.read_from_tfrecords)

    def test_call_np(self):
        reader = DataReaderDummyNP(
            name="reader",
            file_list_keys_mapping={"data1_fl": "data1",
                                    "data2_fl": "data2"}).build()
        processor1 = _DataProcessorNP(
            inbound_nodes=["reader"], name="processor1").build()
        processor2 = _DataProcessorNP2(
            inbound_nodes=["reader"], name="processor2").build()
        data_pipe = DataPipe(readers=[reader],
                             processors=[processor1, processor2]).build()

        reader.read = MagicMock(wraps=reader.read)
        processor1.process = MagicMock(wraps=processor1.process)
        processor2.process = MagicMock(wraps=processor2.process)

        data_pipe.build_dna()
        data_pipe.save_target = self.get_temp_dir()
        for each_sample_file_list in self.file_list_with_floats:
            result = data_pipe(**each_sample_file_list)
            result_must = {
                "reader": {"data1": float(each_sample_file_list["data1_fl"]),
                           "data2": float(each_sample_file_list["data2_fl"])},
                "processor1": {
                    "data1_p": float(each_sample_file_list["data1_fl"]) + 20},
                "processor2": {
                    "data2_p": float(each_sample_file_list["data2_fl"]) + 200}
            }
            self.assertAllClose(result_must,
                                result)

        reader_call_data_must = [
            mock_call(**{k[:5]: v for k, v in each_sample_file_list.items()
                         if k in ["data1_fl", "data2_fl"]})
            for each_sample_file_list in self.file_list_with_floats]
        processor1_call_data_must = [
            mock_call(data1=float(each_sample_file_list["data1_fl"]))
            for each_sample_file_list in self.file_list_with_floats]
        processor2_call_data_must = [
            mock_call(data2=float(each_sample_file_list["data2_fl"]))
            for each_sample_file_list in self.file_list_with_floats]
        saver_save_call_data_must = [
            mock_call(data1_p=float(each_sample_file_list["data1_fl"]) + 20,
                      data2_p=float(each_sample_file_list["data2_fl"]) + 200)
            for each_sample_file_list in self.file_list_with_floats]

        reader.read.assert_has_calls(reader_call_data_must)
        processor1.process.assert_has_calls(processor1_call_data_must)
        processor2.process.assert_has_calls(processor2_call_data_must)

    def test_call_tf(self):
        tf.reset_default_graph()
        reader = DataReaderDummyTF(
            name="reader",
            file_list_keys_mapping={"data1_fl": "data1",
                                    "data2_fl": "data2"}).build()
        processor1 = _DataProcessorTF(
            inbound_nodes=["reader"], name="processor1").build()
        processor2 = _DataProcessorTF2(
            inbound_nodes=["reader"], name="processor2").build()
        data_pipe = DataPipe(readers=[reader],
                             processors=[processor1, processor2]).build()
        data_pipe.build_dna()

        file_list = {}
        for each_sample_file_list in self.file_list_with_floats:
            for each_key, each_value in each_sample_file_list.items():
                file_list.setdefault(each_key, [])
                file_list[each_key].append(each_value)

        file_list_data = tf.data.Dataset.from_tensor_slices(file_list)
        data_pipe_mapped = file_list_data.map(lambda x: data_pipe(**x))
        iterator = data_pipe_mapped.make_one_shot_iterator()
        result = iterator.get_next()

        for i, each_sample_file_list in enumerate(self.file_list_with_floats):
            result_eval = self.evaluate(result)
            result_must = {
                "reader": {"data1": float(each_sample_file_list["data1_fl"]),
                           "data2": float(each_sample_file_list["data2_fl"])},
                "processor1": {
                    "data1_p": float(each_sample_file_list["data1_fl"]) + 20},
                "processor2": {
                    "data2_p": float(each_sample_file_list["data2_fl"]) + 200}
            }
            self.assertAllClose(result_must,
                                result_eval)
        with self.assertRaises(tf.errors.OutOfRangeError):
            _ = self.evaluate(result)

    def test_call_tf_tfrecords(self):
        file_name = self.tfrecords_file_name
        writer = tf.python_io.TFRecordWriter(file_name)
        for each_sample in self.data_tfrecords:
            write_tf_records(each_sample, file_name, writer, False)
        writer.close()

        tf.reset_default_graph()
        reader = TfRecordsDataReaderDummy(
            name="reader",
            file_list_keys_mapping={"data": "tfrecords"}).build()
        processor1 = _DataProcessorTF(
            inbound_nodes=["reader"], name="processor1").build()
        processor2 = _DataProcessorTF2(
            inbound_nodes=["reader"], name="processor2").build()
        data_pipe = DataPipe(readers=[reader],
                             processors=[processor1, processor2]).build()
        data_pipe.build_dna()

        file_list_data = tf.data.TFRecordDataset([file_name])
        data_pipe_mapped = file_list_data.map(lambda x: data_pipe(tfrecords=x))
        iterator = data_pipe_mapped.make_one_shot_iterator()
        result = iterator.get_next()

        for i, each_sample_data in enumerate(self.data_tfrecords):
            result_eval = self.evaluate(result)
            reader_data_must = {k: v for k, v in each_sample_data.items()}
            reader_data_must["data_default"] = np.zeros([1], np.float32)
            result_must = {
                "reader": reader_data_must,
                "processor1": {
                    "data1_p": each_sample_data["data1"] + 20},
                "processor2": {
                    "data2_p": each_sample_data["data2"] + 200},
            }
            self.assertAllClose(result_must,
                                result_eval)
        with self.assertRaises(tf.errors.OutOfRangeError):
            _ = self.evaluate(result)

    def test_filter_sample_true_np(self):
        def _processor1_data_filter_true(data1_p):
            return data1_p > 80

        reader = DataReaderDummyNP(
            name="reader",
            file_list_keys_mapping={"data1_fl": "data1",
                                    "data2_fl": "data2"}).build()
        processor1 = _DataProcessorNP(
            inbound_nodes=["reader"], name="processor1").build()
        processor2 = _DataProcessorNP2(
            inbound_nodes=["reader"], name="processor2").build()
        processor1.data_filter_true = _processor1_data_filter_true
        data_pipe = DataPipe(readers=[reader],
                             processors=[processor1, processor2]).build()

        data_pipe.build_dna()
        data_pipe.save_target = self.get_temp_dir()
        for each_sample_file_list in self.file_list_with_floats:
            result = data_pipe(**each_sample_file_list)
            filter_sample_true = data_pipe.filter_sample_true(**result)
            filter_sample_true_must = result["processor1"]["data1_p"] > 80
            self.assertEqual(filter_sample_true_must,
                             filter_sample_true)

    def test_filter_sample_true_tf(self):
        def _processor1_data_filter_true(data1_p):
            return tf.greater(data1_p, 80.0)

        tf.reset_default_graph()
        reader = DataReaderDummyTF(
            name="reader",
            file_list_keys_mapping={"data1_fl": "data1",
                                    "data2_fl": "data2"}).build()
        processor1 = _DataProcessorTF(
            inbound_nodes=["reader"], name="processor1").build()
        processor2 = _DataProcessorTF2(
            inbound_nodes=["reader"], name="processor2").build()
        data_pipe = DataPipe(readers=[reader],
                             processors=[processor1, processor2]).build()
        data_pipe.build_dna()
        processor1.data_filter_true = _processor1_data_filter_true

        file_list = {}
        for each_sample_file_list in self.file_list_with_floats:
            for each_key, each_value in each_sample_file_list.items():
                file_list.setdefault(each_key, [])
                file_list[each_key].append(each_value)

        file_list_data = tf.data.Dataset.from_tensor_slices(file_list)
        data_pipe_mapped = file_list_data.map(lambda x: data_pipe(**x))
        iterator = data_pipe_mapped.make_one_shot_iterator()
        result = iterator.get_next()

        for i in range(self.num_samples):
            filter_sample_true = data_pipe.filter_sample_true(**result)
            filter_sample_true_eval, result_eval = self.evaluate(
                [filter_sample_true, result])
            filter_sample_true_must = result_eval["processor1"]["data1_p"] > 80

            self.assertEqual(filter_sample_true_must,
                             filter_sample_true_eval)


class TestDataPipeMixin(tf.test.TestCase, parameterized.TestCase):

    def setUp(self):
        reader = DataReaderDummyNP(
            name="reader",
            file_list_keys_mapping={"data1_fl": "data1",
                                    "data2_fl": "data2"}).build()
        processor1 = _DataProcessorNP(
            inbound_nodes=["reader"], name="processor1").build()
        processor1.generated_keys.append("_data_1_optional")
        processor2 = _DataProcessorNP2(
            inbound_nodes=["reader"], name="processor2").build()
        self.data_pipe = DataPipe(readers=[reader],
                                  processors=[processor1, processor2]).build()
        self.data_pipe.build_dna()
        self.output_keys_mapping = {
            "processor1": {"data1_p": "data1_p_rm",
                           "data_1_optional": "data_1_optional_rm"},
            "processor2": {"data2_p": "data2_p_rm"},
            "reader": {"data1": "data1_rm"}}

    @parameterized.parameters({"with_keys_mapping": True},
                              {"with_keys_mapping": False})
    def test_generated_keys(self, with_keys_mapping):
        self.assertListEqual([],
                             DataPipeMixin.generated_keys_required)
        self.assertListEqual([],
                             DataPipeMixin.generated_keys_optional)

        output_keys_mapping = (self.output_keys_mapping if with_keys_mapping
                               else None)
        mixin = DataPipeMixin(data_pipe=self.data_pipe,
                              output_keys_mapping=output_keys_mapping)
        generated_keys_required = mixin.generated_keys_required
        generated_keys_optional = mixin.generated_keys_optional
        if with_keys_mapping:
            generated_keys_required_must = [
                'data1_p_rm', 'data1_rm', 'data2', 'data2_p_rm']
            generated_keys_optional_must = ["data_1_optional_rm"]
        else:
            generated_keys_required_must = [
                'data1_p', 'data1', 'data2', 'data2_p']
            generated_keys_optional_must = ["data_1_optional"]

        self.assertSetEqual(set(generated_keys_required_must),
                            set(generated_keys_required))
        self.assertSetEqual(set(generated_keys_optional_must),
                            set(generated_keys_optional))

    @parameterized.parameters({"with_keys_mapping": True},
                              {"with_keys_mapping": False})
    def test_process_sample_with_pipe(self, with_keys_mapping):
        output_keys_mapping = (self.output_keys_mapping if with_keys_mapping
                               else None)
        mixin = DataPipeMixin(data_pipe=self.data_pipe,
                              output_keys_mapping=output_keys_mapping)

        data = {"data1_fl": "%05d" % random.randint(0, 100),
                "data2_fl": "%d" % random.randint(10, 40),
                "data3_fl": "%08d" % random.randint(100, 1000)}
        result = mixin.process_sample_with_pipe(**data)

        if with_keys_mapping:
            result_must = {
                "data1_rm": float(data["data1_fl"]),
                "data2": float(data["data2_fl"]),
                "data1_p_rm": float(data["data1_fl"]) + 20,
                "data2_p_rm": float(data["data2_fl"]) + 200
            }
        else:
            result_must = {
                "data1": float(data["data1_fl"]),
                "data2": float(data["data2_fl"]),
                "data1_p": float(data["data1_fl"]) + 20,
                "data2_p": float(data["data2_fl"]) + 200
            }

        self.assertAllClose(result_must,
                            result)

    def test_dynamic_generated_keys(self):
        class A:
            dynamic_generated_keys = False

        class B(DataPipeMixin, A):
            pass

        mixin = B(data_pipe=self.data_pipe)
        self.assertFalse(mixin.dynamic_generated_keys)

        self.data_pipe.all_nucleotides["reader"].dynamic_generated_keys = True
        self.assertTrue(mixin.dynamic_generated_keys)
