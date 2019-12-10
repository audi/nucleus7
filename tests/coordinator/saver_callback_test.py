# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import glob
import math
import os
from unittest.mock import MagicMock

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from nucleus7.coordinator.configs import RunIterationInfo
from nucleus7.coordinator.saver_callback import SaverCallback
from nucleus7.coordinator.saver_callback import TfRecordsSaverCallback


class TestDataSaver(tf.test.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        {"provide_save_name": True, "provide_iteration_info": True},
        {"provide_save_name": True, "provide_iteration_info": False},
        {"provide_save_name": False, "provide_iteration_info": True},
        {"provide_save_name": False, "provide_iteration_info": False})
    def test_set_save_name(self, provide_save_name, provide_iteration_info):
        saver = SaverCallback(
            inbound_nodes=[], save_prefix="prefix", save_suffix="suffix"
        ).build()
        saver.log_dir = "save_target"
        save_name = "some_new_name" if provide_save_name else None
        if provide_iteration_info:
            iteration_info = RunIterationInfo(epoch_number=5,
                                              iteration_number=10)
            saver.iteration_info = iteration_info
            saver._sample_index = 3

        saver.set_save_name(save_name)

        if provide_save_name:
            save_name_must = "save_target/prefix-some_new_name-suffix"
        else:
            if provide_iteration_info:
                save_name_must = (
                    "save_target/prefix-"
                    "epoch_{:03d}_iter_{:05d}_sample_{:03d}-suffix"
                    "".format(5, 10, 3))
            else:
                save_name_must = "save_target/prefix-suffix"

        self.assertEqual(save_name_must,
                         saver.save_name)

        saver.log_dir = self.get_temp_dir()
        saver.save_name_depth = -1
        saver.set_save_name("subdir/temp/file-654")
        self.assertTrue(os.path.exists(os.path.join(
            self.get_temp_dir(), "prefix-subdir/temp")))
        saver.set_save_name("subdir/temp/file-654")


class TestTfRecordsDataSaver(tf.test.TestCase,
                             parameterized.TestCase):
    def setUp(self):
        self.data_saver = TfRecordsSaverCallback(
            inbound_nodes=[],
            number_of_samples_per_file=5).build()
        self.data_saver.log_dir = self.get_temp_dir()
        self.num_samples = 14
        self.data = [
            {"data1": np.random.rand(10, 20).astype(np.float32),
             "data2": np.random.rand(5).astype(np.float32),
             "data3": np.random.rand(1, 5).astype(np.float32)}
            for _ in range(self.num_samples)]
        self.additional_names = [
            *["sample_part_1"] * 3,
            *["sample_part_2"] * 9,
            *["sample_part_3"] * 2,
        ]
        self.features = {'data1': tf.FixedLenFeature((), tf.string),
                         'data2': tf.FixedLenFeature((), tf.string),
                         'data3': tf.FixedLenFeature((), tf.string)}

    @parameterized.parameters({"with_additional_names": True},
                              {"with_additional_names": False})
    def test_on_iteration_end(self, with_additional_names):
        batch_size = 5
        num_batches = math.ceil(self.num_samples / batch_size)
        batches = []
        for i in range(num_batches):
            start_ind = i * batch_size
            end_ind = min((i + 1) * batch_size, self.num_samples)
            batch = {k: np.stack([self.data[jj][k]
                                  for jj in range(start_ind, end_ind)], 0)
                     for k in self.data[0].keys()}
            if with_additional_names:
                batch["save_names"] = np.array(
                    self.additional_names[start_ind: end_ind])
            batches.append(batch)

        self.data_saver.save_sample = MagicMock(
            wraps=self.data_saver.save_sample)

        for each_batch in batches:
            self.data_saver.on_iteration_end(**each_batch)
        self.data_saver.end()

        self.assertEqual(self.num_samples,
                         self.data_saver.save_sample.call_count)
        for i in range(self.num_samples):
            self.assertAllClose(
                self.data[i],
                self.data_saver.save_sample.call_args_list[i][1])

        fnames_tfrecords_generated = sorted(
            glob.glob(os.path.join(
                self.data_saver.log_dir, 'data*.tfrecords')))
        samples_in_tfrecords = []
        number_samples_inside_tfrecords = []
        for tfrecords_fname in fnames_tfrecords_generated:
            number_samples_inside = 0
            for example in tf.python_io.tf_record_iterator(tfrecords_fname):
                sample = tf.parse_single_example(example, self.features)
                sample_decoded = {k: tf.decode_raw(v, tf.float32)
                                  for k, v in sample.items()}
                samples_in_tfrecords.append(sample_decoded)
                number_samples_inside += 1
            number_samples_inside_tfrecords.append(number_samples_inside)

        if with_additional_names:
            number_samples_inside_tfrecords_must = [3, 5, 4, 2]
        else:
            number_samples_inside_tfrecords_must = [5, 5, 4]

        self.assertListEqual(number_samples_inside_tfrecords_must,
                             number_samples_inside_tfrecords)

        samples_in_tfrecords_eval = self.evaluate(samples_in_tfrecords)

        samples_eval_must = [{k: v.flatten() for k, v in single_sample.items()}
                             for single_sample in self.data]
        self.assertAllClose(samples_eval_must,
                            samples_in_tfrecords_eval)
