# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import copy
import os
import random
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import call as mock_call

from absl.testing import parameterized
from itertools import chain
import numpy as np
import tensorflow as tf

from nucleus7.data.data_filter import DataFilter
from nucleus7.data.data_pipe import DataPipe
from nucleus7.data.dataset import Dataset
from nucleus7.data.dataset import DatasetFileList
from nucleus7.data.dataset import DatasetMix
from nucleus7.data.dataset import DatasetTfRecords
from nucleus7.data.file_list import FileList
from nucleus7.data.processor import DataProcessor
from nucleus7.test_utils.model_dummies import DataReaderDummyTF
from nucleus7.test_utils.model_dummies import DatasetDummy
from nucleus7.test_utils.model_dummies import FileListDummy
from nucleus7.test_utils.model_dummies import TfRecordsDataReaderDummy
from nucleus7.test_utils.test_utils import write_tf_records
from nucleus7.utils import nest_utils


class TestDataset(parameterized.TestCase, tf.test.TestCase):

    def setUp(self):
        tf.reset_default_graph()
        self.number_of_samples = 10
        self.inputs = {'input1': ['input1_{}'.format(i)
                                  for i in range(self.number_of_samples)],
                       'input2': ['input2_{}'.format(i)
                                  for i in range(self.number_of_samples)]}

    def test_constructor(self):
        with self.assertRaises(ValueError):
            Dataset(number_of_shards=0, shard_index=0)
        with self.assertRaises(ValueError):
            Dataset(number_of_shards=2, shard_index=2)
        with self.assertRaises(ValueError):
            Dataset(shuffle_buffer_size=-1)
        with self.assertRaises(ValueError):
            Dataset(prefetch_buffer_size=-1)
        temp_dir = self.get_temp_dir()
        with self.assertRaises(FileNotFoundError):
            Dataset(cache_dir=os.path.join(temp_dir, 'cache1', 'cache2'))

    @parameterized.parameters({'number_of_shards': 1, "shard_index": 0},
                              {'number_of_shards': 4, "shard_index": 2})
    def test_shard(self, number_of_shards=4, shard_index=2):
        data = tf.data.Dataset.from_tensor_slices(self.inputs)
        dataset = Dataset(number_of_shards=number_of_shards,
                          shard_index=shard_index).build()
        data_sharded = dataset.shard(data)
        outputs = _get_data_results(data_sharded, self.test_session())
        if number_of_shards == 1:
            outputs_must = self.inputs
        else:
            index_start = shard_index
            shard_step = number_of_shards
            outputs_must = {k: v[index_start:None:shard_step]
                            for k, v in self.inputs.items()}
        self.assertDictEqual(outputs_must, outputs)

    @parameterized.parameters({'shuffle_buffer_size': 1},
                              {'shuffle_buffer_size': 100})
    def test_shuffle(self, shuffle_buffer_size):
        data = tf.data.Dataset.from_tensor_slices(self.inputs)
        dataset = Dataset(shuffle_buffer_size=shuffle_buffer_size).build()
        data_shuffled = dataset.shuffle(data)
        outputs = _get_data_results(data_shuffled, self.test_session())
        if shuffle_buffer_size == 1:
            self.assertDictEqual(self.inputs,
                                 outputs)
            return

        for each_key, each_list_value in self.inputs.items():
            self.assertNotEqual(each_list_value,
                                outputs[each_key])
            self.assertSetEqual(set(each_list_value),
                                set(outputs[each_key]))

    def test_repeat(self):
        data = tf.data.Dataset.from_tensor_slices(self.inputs)
        dataset = Dataset().build()
        data_repeat = dataset.repeat(data)
        outputs = _get_data_results(data_repeat, self.test_session(),
                                    max_iteration=self.number_of_samples * 5)
        outputs_repeated_must = {k: v * 5 for k, v in self.inputs.items()}
        self.assertDictEqual(outputs_repeated_must,
                             outputs)

    def test_cache_and_clear_cache(self):
        cache_dir = os.path.join(self.get_temp_dir(), 'cache')
        cache_file_name_pref_must = "-".join(['Dataset', 'stype', 'train'])

        data = tf.data.Dataset.from_tensor_slices(self.inputs)
        dataset = Dataset(cache_dir=cache_dir, subtype='stype').build()
        dataset.mode = 'train'
        data_cached = dataset.cache(data)
        outputs = _get_data_results(data_cached, self.test_session())
        cache_file_name = dataset._cache_file

        self.assertTrue(os.path.split(cache_file_name)[-1].startswith(
            cache_file_name_pref_must))
        self.assertDictEqual(self.inputs,
                             outputs)
        self.assertTrue(os.path.isfile(cache_file_name + '.index'))

        dataset.clear_cache()
        self.assertFalse(os.path.isfile(cache_file_name + '.index'))

    @parameterized.parameters({'mode': "train", 'number_of_shards': 1,
                               'with_cache': True},
                              {'mode': "train", 'number_of_shards': 3},
                              {'mode': "eval"})
    def test_create_features_for_single_sample(self, mode, number_of_shards=1,
                                               with_cache=False):
        def _create_initial_data():
            return self.inputs

        def _extract_features_from_initial_data(inputs):
            return _add_suffix_to_each_value_in_dict_of_lists(
                inputs, '_features')

        def _filter_sample(inputs):
            return _add_suffix_to_each_value_in_dict_of_lists(
                inputs, '_filtered')

        def _shard(inputs):
            return _add_suffix_to_each_value_in_dict_of_lists(
                inputs, '_sharded')

        def _cache(inputs):
            return _add_suffix_to_each_value_in_dict_of_lists(
                inputs, '_cached')

        def _shuffle(inputs):
            return _add_suffix_to_each_value_in_dict_of_lists(
                inputs, '_shuffled')

        def _repeat(inputs):
            return _add_suffix_to_each_value_in_dict_of_lists(
                inputs, '_repeated')

        if with_cache:
            cache_dir = os.path.join(self.get_temp_dir(), 'cache')
        else:
            cache_dir = None
        dataset = Dataset(number_of_shards=number_of_shards,
                          cache_dir=cache_dir).build()
        dataset.mode = mode
        dataset.create_initial_data = MagicMock(
            side_effect=_create_initial_data)
        dataset.shard = MagicMock(
            side_effect=_shard)
        dataset.extract_features_from_initial_data = MagicMock(
            side_effect=_extract_features_from_initial_data)
        dataset.cache = MagicMock(side_effect=_cache)
        dataset.shuffle = MagicMock(side_effect=_shuffle)
        dataset.repeat = MagicMock(side_effect=_repeat)
        dataset.add_data_filter(_DummyDataFilterOddData())
        dataset.filter_sample = MagicMock(side_effect=_filter_sample)

        outputs = dataset.create_features_for_single_sample()

        outputs_must = {k: v for k, v in self.inputs.items()}
        if number_of_shards > 1:
            outputs_must = _add_suffix_to_each_value_in_dict_of_lists(
                outputs_must, '_sharded')
        outputs_must = _add_suffix_to_each_value_in_dict_of_lists(
            outputs_must, '_features')
        if with_cache:
            outputs_must = _add_suffix_to_each_value_in_dict_of_lists(
                outputs_must, '_cached')
        if mode == 'train':
            outputs_must = _add_suffix_to_each_value_in_dict_of_lists(
                outputs_must, '_shuffled')
        outputs_must = _add_suffix_to_each_value_in_dict_of_lists(
            outputs_must, '_repeated')
        outputs_must = _add_suffix_to_each_value_in_dict_of_lists(
            outputs_must, '_filtered')
        self.assertDictEqual(outputs_must,
                             outputs)

    @parameterized.parameters({'padded': True, "fix_batch_dimension": True},
                              {'padded': True, "fix_batch_dimension": False},
                              {'padded': False, "fix_batch_dimension": True},
                              {'padded': False, "fix_batch_dimension": False})
    def test_combine_samples_to_batch(self, padded,
                                      fix_batch_dimension=True):
        batch_size = 3
        if padded:
            inputs = {'input1': tf.placeholder(tf.float32, [10, None]),
                      'input2': tf.placeholder(tf.float32, [10, 2, None])}
        else:
            inputs = {'input1': tf.placeholder(tf.float32, [10, 5]),
                      'input2': tf.placeholder(tf.float32, [10, 2, 3])}
        dataset = Dataset(fix_batch_dimension=fix_batch_dimension).build()
        data = tf.data.Dataset.from_tensor_slices(inputs)
        data_batch = dataset.combine_samples_to_batch(data, batch_size)

        batch_size_in_shape = batch_size if fix_batch_dimension else None
        if padded:
            output_shapes_must = {
                'input1': [batch_size_in_shape, None],
                'input2': [batch_size_in_shape, 2, None]
            }
        else:
            output_shapes_must = {
                'input1': [batch_size_in_shape, 5],
                'input2': [batch_size_in_shape, 2, 3]
            }

        output_shapes_as_list = {
            k: each_shape.as_list()
            for k, each_shape in data_batch.output_shapes.items()}

        self.assertDictEqual(output_shapes_must,
                             output_shapes_as_list)

    @parameterized.parameters({'prefetch_buffer_size': 0},
                              {'prefetch_buffer_size': 10})
    def test_create_batch(self, prefetch_buffer_size):
        batch_size = 3
        data = tf.data.Dataset.from_tensor_slices(self.inputs)
        data_batch = data.batch(batch_size)

        def _create_features_for_single_sample():
            return data

        def _combine_samples_to_batch(data, batch_size):
            return data_batch

        def _prefetch(data):
            return data

        dataset = Dataset(prefetch_buffer_size=prefetch_buffer_size).build()
        dataset.create_features_for_single_sample = MagicMock(
            side_effect=_create_features_for_single_sample)
        dataset.combine_samples_to_batch = MagicMock(
            side_effect=_combine_samples_to_batch)
        dataset.prefetch = MagicMock(side_effect=_prefetch)
        dataset.mode = 'train'

        create_batch_calls_mock = Mock()
        create_batch_calls_mock.attach_mock(
            dataset.create_features_for_single_sample,
            'create_features_for_single_sample_call')
        create_batch_calls_mock.attach_mock(
            dataset.combine_samples_to_batch,
            'combine_samples_to_batch_call')
        create_batch_calls_mock.attach_mock(
            dataset.prefetch,
            'prefetch_call')

        data_batch = dataset.create_batch(batch_size)

        create_batch_expected_calls = [
            mock_call.create_features_for_single_sample_call(),
            mock_call.combine_samples_to_batch_call(data, batch_size)
        ]
        if prefetch_buffer_size > 0:
            create_batch_expected_calls.append(
                mock_call.prefetch_call(data_batch))

        create_batch_calls_mock.assert_has_calls(create_batch_expected_calls)

    @parameterized.parameters({'mode': "train"},
                              {'mode': "eval"})
    def test_filter_sample(self, mode):
        batch_size = 2
        dataset = _DatasetDummy()
        dataset.add_data_filter(_DummyDataFilterOddData())
        dataset.build()
        dataset.repeat = MagicMock(side_effect=lambda x: x)
        dataset.shuffle = MagicMock(side_effect=lambda x: x)
        dataset.mode = mode

        data = dataset.create_batch(batch_size)
        iterator = data.make_one_shot_iterator()
        batch = iterator.get_next()
        results = []
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            try:
                while True:
                    results.append(sess.run(batch))
            except tf.errors.OutOfRangeError:
                pass

        results_must = [{"value": np.array([1, 3])},
                        {"value": np.array([5, 7])},
                        {"value": np.array([9])}]
        self.assertAllClose(results_must,
                            results)


class TestDatasetFileList(tf.test.TestCase,
                          parameterized.TestCase):

    def setUp(self):
        tf.reset_default_graph()
        temp_dir = self.get_temp_dir()
        self.number_of_samples = 10
        self.file_names = {
            'input1': [os.path.join(temp_dir, 'input1_{:05d}.tmp'.format(i))
                       for i in range(self.number_of_samples)],
            'input2': [os.path.join(temp_dir, 'input2_{:05d}.tmp'.format(i))
                       for i in range(self.number_of_samples)]
        }
        self.file_list = FileListDummy(file_names=self.file_names).build()

    @parameterized.parameters({'initial_shuffle': True},
                              {'initial_shuffle': False})
    def test_build(self, initial_shuffle):
        file_names_orig = copy.deepcopy(self.file_names)
        file_list = self.file_list
        dataset = DatasetFileList(file_list=file_list,
                                  initial_shuffle=initial_shuffle)
        dataset.file_list_keys = ['input_wrong']
        with self.assertRaises(ValueError):
            dataset.build()

        dataset = DatasetFileList(file_list=file_list,
                                  initial_shuffle=initial_shuffle)
        dataset.file_list_keys = ['input1']
        dataset.build()
        file_names_orig = {k: v for k, v in file_names_orig.items()
                           if k in dataset.file_list_keys}
        if initial_shuffle:
            self.assertNotEqual(file_names_orig,
                                dataset.file_list.get())
            file_names_set_must = {k: set(v)
                                   for k, v in file_names_orig.items()}
            file_names_set = {k: set(v)
                              for k, v in dataset.file_list.get().items()}
            self.assertDictEqual(file_names_set_must,
                                 file_names_set)
        else:
            self.assertDictEqual(file_names_orig,
                                 dataset.file_list.get())

    def test_create_batch(self):
        def read_raw_data_from_file(input1, input2):
            return {'shape': tf.size(input1),
                    'fname': tf.identity(input2)}

        tf.reset_default_graph()
        batch_size = 10
        file_list = self.file_list
        dataset = DatasetFileList(file_list)
        dataset.file_list_keys = ['input1', 'input2']
        dataset.build()
        dataset.read_raw_data_from_file = MagicMock(
            side_effect=read_raw_data_from_file)

        dataset.mode = 'train'
        data = dataset.create_batch(batch_size)
        iterator = data.make_one_shot_iterator()
        batch = iterator.get_next()
        self.assertSetEqual(set(batch.keys()), {'shape', 'fname'})


class TestDatasetTfRecords(tf.test.TestCase,
                           parameterized.TestCase):

    def setUp(self):
        tf.reset_default_graph()
        self.number_of_files = 10
        self.number_of_samples_per_file = 5
        total_number_of_samples = (self.number_of_files
                                   * self.number_of_samples_per_file)
        temp_dir = self.get_temp_dir()
        self.file_names = [
            os.path.join(temp_dir, 'input1_{}.tfrecords'.format(i))
            for i in range(self.number_of_files)
        ]
        self.data = {
            'input_to_decode': np.random.random([total_number_of_samples, 5]
                                                ).astype(np.float32),
            'input_number': {
                'floats': np.arange(0, total_number_of_samples,
                                    dtype=np.float32),
                'int': np.arange(1000, 1000 + total_number_of_samples,
                                 dtype=np.int64)}
        }
        self.file_names = {'data': self.file_names}
        self.file_list = FileListDummy(file_names=self.file_names).build()
        self._write_tfrecords()

    @parameterized.parameters({"use_post_processing": True},
                              {"use_post_processing": False})
    def test_extract_features_from_initial_data(self,
                                                use_post_processing=False):
        def _postprocess_tfrecords(input_to_decode,
                                   input_number):
            return {'input_to_decode': input_to_decode + 1000,
                    'input_number': {k: v + 156
                                     for k, v in input_number.items()}}

        dataset = _DatasetTfRecordsDummy(file_list=self.file_list,
                                         shuffle_buffer_size=0,
                                         initial_shuffle=False).build()
        dataset.mode = 'eval'
        if use_post_processing:
            dataset.postprocess_tfrecords = MagicMock(
                side_effect=_postprocess_tfrecords)
        initial_data = dataset.create_initial_data()
        data_extracted = dataset.extract_features_from_initial_data(
            initial_data)
        outputs = _get_data_results(data_extracted, self.test_session())

        data_must = self.data
        if use_post_processing:
            data_must['input_to_decode'] += 1000
            data_must['input_number']['floats'] += 156
            data_must['input_number']['int'] += 156

        self.assertAllClose(data_must, outputs)

    def _write_tfrecords(self):

        data_flatten = nest_utils.flatten_nested_struct(self.data,
                                                        separator='/')
        sample_index = 0
        for filename in self.file_names['data']:
            tfrecord_writer = tf.python_io.TFRecordWriter(filename)
            for _ in range(self.number_of_samples_per_file):
                sample = {key: value[sample_index]
                          for key, value in data_flatten.items()}
                sample_index += 1
                _write_sample_to_tfrecord(sample, tfrecord_writer)
            tfrecord_writer.close()


class TestDatasetMix(tf.test.TestCase,
                     parameterized.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.number_of_samples1 = 10
        self.number_of_samples2 = 5
        self.inputs1 = {'input1': ['input11_{}'.format(i)
                                   for i in range(self.number_of_samples1)],
                        'input2': np.arange(
                            self.number_of_samples1, dtype=np.float32)}
        self.inputs2 = {'input1': ['input21_{}'.format(i)
                                   for i in range(self.number_of_samples2)],
                        'input3': {'int': np.ones([self.number_of_samples2,
                                                   20], dtype=np.int32)},
                        'input4': ['input4_{}'.format(i)
                                   for i in range(self.number_of_samples2)]}

        self.dataset1 = DatasetDummy(data=self.inputs1,
                                     shuffle_buffer_size=1,
                                     subtype='data1').build()
        self.dataset1.generated_keys = ['input1', '_input2']
        self.dataset2 = DatasetDummy(data=self.inputs2,
                                     shuffle_buffer_size=1,
                                     subtype='data2').build()
        self.dataset2.generated_keys = ['input1', 'input3', 'input4']
        self.datasets_for_mix = [self.dataset1, self.dataset2]

    @parameterized.parameters({'mode': 'train'},
                              {'mode': 'eval'})
    def test_mode_setter(self, mode):
        dataset_mix = DatasetMix(datasets=self.datasets_for_mix).build()
        dataset_mix.mode = mode
        self.assertEqual(mode,
                         self.datasets_for_mix[0].mode)
        self.assertEqual(mode,
                         self.datasets_for_mix[1].mode)

    def test_dynamic_generated_keys(self):
        dataset_mix = DatasetMix(datasets=self.datasets_for_mix).build()
        self.assertFalse(dataset_mix.dynamic_generated_keys)
        self.datasets_for_mix[0].dynamic_generated_keys = True
        dataset_mix = DatasetMix(datasets=self.datasets_for_mix).build()
        self.assertTrue(dataset_mix.dynamic_generated_keys)

    def test_generated_keys(self):
        dataset_mix_cls = DatasetMix
        self.assertEmpty(dataset_mix_cls.generated_keys_required)
        self.assertEmpty(dataset_mix_cls.generated_keys_optional)
        dataset_mix = DatasetMix(datasets=self.datasets_for_mix).build()
        keys_required_must = {'input1', 'input3', 'input4'}
        keys_optional_must = {'input2',
                              'sample_mask_data1', 'sample_mask_data2'}
        keys_all_must = {'input1', 'input2', 'input3', 'input4',
                         'sample_mask_data1', 'sample_mask_data2'}
        self.assertSetEqual(keys_required_must,
                            set(dataset_mix.generated_keys_required))
        self.assertSetEqual(keys_optional_must,
                            set(dataset_mix.generated_keys_optional))
        self.assertSetEqual(keys_all_must,
                            set(dataset_mix.generated_keys_all))

    def test_build(self):
        with self.assertRaises(ValueError):
            DatasetMix(datasets=[self.dataset1]).build()
        datasets_for_mix_same_type = [
            DatasetDummy(data=self.inputs1,
                         shuffle_buffer_size=1).build(),
            DatasetDummy(data=self.inputs2,
                         shuffle_buffer_size=1).build()
        ]
        with self.assertRaises(ValueError):
            DatasetMix(datasets=datasets_for_mix_same_type).build()

        _ = DatasetMix(datasets=self.datasets_for_mix)

    @parameterized.parameters(
        {'mode': 'train', 'sampling_weights': [2.0, 1.0]},
        {'mode': 'train'},
        {'mode': 'eval', 'sampling_weights': [2.0, 1.0]},
        {'mode': 'eval'})
    def test_create_features_for_single_sample(self, mode,
                                               sampling_weights=None):
        dataset_mix = DatasetMix(datasets=self.datasets_for_mix,
                                 sampling_weights=sampling_weights).build()
        dataset_mix.mode = mode
        data_mixed = dataset_mix.create_features_for_single_sample()

        output_shapes_must = {'input1': [],
                              'input2': [],
                              'input3': {'int': [20]},
                              'input4': [],
                              'sample_mask_data1': [],
                              'sample_mask_data2': []}
        output_types_must = {'input1': tf.string,
                             'input2': tf.float32,
                             'input3': {'int': tf.int32},
                             'input4': tf.string,
                             'sample_mask_data1': tf.float32,
                             'sample_mask_data2': tf.float32}
        output_shapes_list_flatten = nest_utils.flatten_nested_struct(
            data_mixed.output_shapes)
        output_shapes_list_flatten = {
            k: v.as_list() for k, v in output_shapes_list_flatten.items()}
        output_shapes_list = nest_utils.unflatten_dict_to_nested(
            output_shapes_list_flatten)
        self.assertDictEqual(output_shapes_must,
                             output_shapes_list)
        self.assertDictEqual(output_types_must,
                             data_mixed.output_types)

        default_values_1 = {
            'input3': {'int': np.zeros([self.number_of_samples1, 20],
                                       dtype=np.int32)},
            'input4': ["" for _ in range(self.number_of_samples1)],
            'sample_mask_data1': [1.0] * self.number_of_samples1,
            'sample_mask_data2': [0.0] * self.number_of_samples1
        }
        default_values_2 = {
            'input2': [0.0 for _ in range(self.number_of_samples2)],
            'sample_mask_data1': [0.0] * self.number_of_samples1,
            'sample_mask_data2': [1.0] * self.number_of_samples1
        }

        output1_with_defaults_must = copy.deepcopy(self.inputs1)
        output1_with_defaults_must.update(default_values_1)
        output2_with_defaults_must = copy.deepcopy(self.inputs2)
        output2_with_defaults_must.update(default_values_2)

        if mode == 'train':
            iterations_number = 500
        else:
            iterations_number = 100
        output = _get_data_results(data_mixed, self.test_session(),
                                   max_iteration=iterations_number)
        if mode == 'eval':
            outputs_must = {}
            for each_key in ['input1', 'input2', 'input4',
                             'sample_mask_data1', 'sample_mask_data2']:
                outputs_must[each_key] = _interleave(
                    output1_with_defaults_must[each_key],
                    output2_with_defaults_must[each_key],
                    iterations_number
                )
            outputs_must['input3'] = {
                'int': _interleave(
                    output1_with_defaults_must['input3']['int'],
                    output2_with_defaults_must['input3']['int'],
                    iterations_number
                )
            }
            for each_key in ['input2', 'input3', 'sample_mask_data1',
                             'sample_mask_data2']:
                self.assertAllClose(outputs_must[each_key],
                                    output[each_key])
            for each_key in ['input1', 'input4']:
                self.assertListEqual(outputs_must[each_key],
                                     output[each_key])
        else:
            number_samples_per_dataset = [0, 0]
            datasets_with_defaults = [output1_with_defaults_must,
                                      output2_with_defaults_must]
            for i_sample in range(iterations_number):
                sample = _select_sample_at_index(output, i_sample)
                dataset_ind = _get_dataset_id_from_sample(
                    sample, datasets_with_defaults)
                number_samples_per_dataset[dataset_ind] += 1
            sampling_weights_output = [v / iterations_number
                                       for v in number_samples_per_dataset]
            if sampling_weights is not None:
                sampling_weights_norm_must = [v / sum(sampling_weights)
                                              for v in sampling_weights]
            else:
                sampling_weights_norm_must = [0.5, 0.5]
            atol = 10 / iterations_number
            self.assertAllClose(sampling_weights_norm_must,
                                sampling_weights_output,
                                atol=atol)

    @parameterized.parameters({'mode': 'train', 'fix_batch_dimension': True},
                              {'mode': 'train', 'fix_batch_dimension': False},
                              {'mode': 'eval', 'fix_batch_dimension': True},
                              {'mode': 'eval', 'fix_batch_dimension': False})
    def test_create_batch(self, mode, fix_batch_dimension):
        self.datasets_for_mix[0].fix_batch_dimension = fix_batch_dimension
        self.datasets_for_mix[1].fix_batch_dimension = fix_batch_dimension

        dataset_mix = DatasetMix(datasets=self.datasets_for_mix).build()

        dataset_mix.mode = mode
        batch_size = 10
        data_batch = dataset_mix.create_batch(batch_size=batch_size)

        batch_dim_must = batch_size if fix_batch_dimension else None

        output_shapes_must = {'input1': [batch_dim_must],
                              'input2': [batch_dim_must],
                              'input3': {'int': [batch_dim_must, 20]},
                              'input4': [batch_dim_must],
                              'sample_mask_data1': [batch_dim_must],
                              'sample_mask_data2': [batch_dim_must]}
        output_types_must = {'input1': tf.string,
                             'input2': tf.float32,
                             'input3': {'int': tf.int32},
                             'input4': tf.string,
                             'sample_mask_data1': tf.float32,
                             'sample_mask_data2': tf.float32}
        output_shapes_list_flatten = nest_utils.flatten_nested_struct(
            data_batch.output_shapes)
        output_shapes_list_flatten = {
            k: v.as_list() for k, v in output_shapes_list_flatten.items()}
        output_shapes_list = nest_utils.unflatten_dict_to_nested(
            output_shapes_list_flatten)
        self.assertDictEqual(output_shapes_must,
                             output_shapes_list)
        self.assertDictEqual(output_types_must,
                             data_batch.output_types)
        _ = _get_data_results(data_batch, self.test_session(),
                              max_iteration=100)

    @parameterized.parameters(
        {"merge_on_same_file_list": None, "mode": "train"},
        {"merge_on_same_file_list": [False] * 3, "mode": "train"},
        {"merge_on_same_file_list": [True] * 3, "mode": "train"},
        {"merge_on_same_file_list": [True, False, True], "mode": "train"},
        {"merge_on_same_file_list": [False, False, True], "mode": "train"},
        {"merge_on_same_file_list": None, "mode": "eval"},
        {"merge_on_same_file_list": [False] * 3, "mode": "eval"},
        {"merge_on_same_file_list": [True] * 3, "mode": "eval"},
        {"merge_on_same_file_list": [True, False, True], "mode": "eval"},
        {"merge_on_same_file_list": [False, False, True], "mode": "eval"}
    )
    def test_create_features_for_single_sample_same_file_list(
            self, mode, merge_on_same_file_list):

        merge_on_same_file_list = None
        mode = "eval"

        class _DatasetFileList1(DatasetFileList):
            file_list_keys = ["input1", "input2"]
            generated_keys = ["input1_out", "input2_out"]

            def read_raw_data_from_file(_self, **kwargs) -> dict:
                return {k + "_out": tf.identity(v) for k, v in kwargs.items()}

        class _DatasetFileList2(DatasetFileList):
            file_list_keys = ["input1", "input3"]
            generated_keys = ["input1_out", "input3_out"]

            def read_raw_data_from_file(_self, input1, input3) -> dict:
                return {"input1_out": tf.identity(input1 + "_2nd"),
                        "input3_out": tf.identity(input3 + "_2nd")}

        class _DatasetFileList3(DatasetFileList):
            file_list_keys = ["input1", "input2"]
            generated_keys = ["input1_out", "new_out"]

            def read_raw_data_from_file(_self, input1, input2) -> dict:
                return {"input1_out": tf.identity(input1 + "_3rd"),
                        "new_out": tf.constant(10)}

        tf.reset_default_graph()
        number_of_samples = 10
        file_names1 = {
            'input1': ['input1_{:02d}.tmp'.format(i)
                       for i in range(number_of_samples)],
            'input2': ['input2_{:02d}.tmp'.format(i)
                       for i in range(number_of_samples)]
        }
        file_names2 = {
            'input1': ['input1_{:02d}.tmp'.format(i)
                       for i in range(10, number_of_samples + 20)],
            'input3': ['input2_{:02d}.tmp'.format(i)
                       for i in range(10, number_of_samples + 20)]
        }
        file_list1 = FileListDummy(file_names=file_names1).build()
        file_list2 = FileListDummy(file_names=file_names2).build()
        file_list3 = FileListDummy(file_names=file_names1).build()
        dataset1 = _DatasetFileList1(
            file_list=file_list1, subtype="dataset1").build()
        dataset2 = _DatasetFileList2(
            file_list=file_list2, subtype="dataset2").build()
        dataset3 = _DatasetFileList3(
            file_list=file_list3, subtype="dataset3").build()

        sampling_weights = [0.1, 0.2, 0.3]
        dataset_mix = DatasetMix(datasets=[dataset1, dataset2, dataset3],
                                 sampling_weights=sampling_weights,
                                 merge_on_same_file_list=merge_on_same_file_list
                                 ).build()
        dataset_mix.mode = mode
        data_mixed = dataset_mix.create_features_for_single_sample()

        if (not merge_on_same_file_list
                or (merge_on_same_file_list[0]
                    and merge_on_same_file_list[2])):
            out_keys_must = ['input1_out', 'input2_out', 'input3_out',
                             'new_out', 'sample_mask_dataset1',
                             'sample_mask_dataset2']
            result_must = {'input1_out': ['input1_00.tmp', 'input1_10.tmp_2nd',
                                          'input1_01.tmp', 'input1_11.tmp_2nd'],
                           'input2_out': ['input2_00.tmp', '', 'input2_01.tmp',
                                          ''],
                           'input3_out': ['', 'input2_10.tmp_2nd', '',
                                          'input2_11.tmp_2nd'],
                           'new_out': [10, 0, 10, 0],
                           'sample_mask_dataset1': [1.0, 0.0, 1.0, 0.0],
                           'sample_mask_dataset2': [0.0, 1.0, 0.0, 1.0]}
        else:
            out_keys_must = ['input1_out', 'input2_out', 'input3_out',
                             'new_out', 'sample_mask_dataset1',
                             'sample_mask_dataset2', 'sample_mask_dataset3']
            result_must = {'input1_out': ['input1_00.tmp', 'input1_10.tmp_2nd',
                                          'input1_00.tmp_3rd', 'input1_01.tmp'],
                           'input2_out': ['input2_00.tmp', '', '',
                                          'input2_01.tmp'],
                           'input3_out': ['', 'input2_10.tmp_2nd', '', ''],
                           'new_out': [0, 0, 10, 0],
                           'sample_mask_dataset1': [1.0, 0.0, 0.0, 1.0],
                           'sample_mask_dataset2': [0.0, 1.0, 0.0, 0.0],
                           'sample_mask_dataset3': [0.0, 0.0, 1.0, 0.0]}

        self.assertSetEqual(set(out_keys_must),
                            set(data_mixed.output_types))

        result_eval = _get_data_results(data_mixed,
                                        self.test_session(), max_iteration=4)
        if mode == "eval":
            self.assertAllEqual(result_must,
                                result_eval)


class TestDatasetFromPipe(tf.test.TestCase):
    def test_call(self):
        output_keys_mapping = {"processor1": {"data11": "data11_rm"}}
        processor1 = _DataGenerator1(name="processor1").build()
        processor2 = _DataGenerator2(name="processor2").build()
        data_pipe = DataPipe(processors=[processor1, processor2]
                             ).build()
        dataset = Dataset.from_data_pipe(
            data_pipe=data_pipe, random_seed=111,
            output_keys_mapping=output_keys_mapping).build()
        dataset.mode = "train"
        result = dataset(batch_size=4)
        result_iterator = result.make_initializable_iterator()
        batch = result_iterator.get_next()
        shapes = {k: v.as_list() for k, v in result.output_shapes.items()}
        shapes_must = {
            "data11_rm": [None, 10, 5],
            "data12": [None, 1],
            "data21": [None, 1, 3],
            "data22": [None, 5]
        }
        self.assertAllEqual(shapes_must,
                            shapes)
        self.assertSetEqual(set(shapes_must),
                            set(dataset.generated_keys_all))

        self.assertEqual(111,
                         dataset.random_seed)
        self.assertIsInstance(dataset,
                              Dataset)
        self.evaluate(result_iterator.initializer)
        _ = self.evaluate(batch)

    def test_initialize_session(self):
        output_keys_mapping = {"processor1": {"data11": "data11_rm"}}
        processor1 = _DataGenerator1(name="processor1").build()
        processor2 = _DataGenerator2(name="processor2").build()
        processor1.initialize_session = MagicMock(return_value=None)
        processor2.initialize_session = MagicMock(return_value=None)
        data_pipe = DataPipe(processors=[processor1, processor2]
                             ).build()
        dataset = Dataset.from_data_pipe(
            data_pipe=data_pipe, random_seed=111,
            output_keys_mapping=output_keys_mapping).build()
        dataset.initialize_session()

        processor1.initialize_session.assert_called_once_with()
        processor2.initialize_session.assert_called_once_with()


class TestFileListDatasetFromPipe(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.num_samples = 50
        self.file_names_with_floats = {
            "data1": ["%05d" % random.randint(0, 100)
                      for _ in range(self.num_samples)],
            "data2": ["%d" % random.randint(10, 40)
                      for _ in range(self.num_samples)],
            "data3": ["%08d" % random.randint(100, 1000)
                      for _ in range(self.num_samples)]}

    def test_call(self):
        output_keys_mapping = {"processor2": {"data1_p": "data2_p"}}
        reader_tf = DataReaderDummyTF(name="reader").build()
        processor1 = _DataProcessorTF(name="processor1",
                                      inbound_nodes=["reader"]).build()
        processor2 = _DataProcessorTF(
            name="processor2",
            inbound_nodes=["reader"],
            incoming_keys_mapping={"reader": {"data2": "data1"}}
        ).build()
        data_pipe = DataPipe(processors=[processor1, processor2],
                             readers=reader_tf
                             ).build()
        file_list = FileList.from_matched_file_names(
            self.file_names_with_floats)
        dataset = Dataset.from_data_pipe(
            data_pipe=data_pipe, file_list=file_list,
            output_keys_mapping=output_keys_mapping,
            random_seed=111).build()
        dataset.mode = "train"
        result = dataset(batch_size=4)
        result_iterator = result.make_initializable_iterator()
        batch = result_iterator.get_next()

        shapes = {k: v.as_list() for k, v in result.output_shapes.items()}
        shapes_must = {
            "data1": [None],
            "data2": [None],
            "data1_p": [None],
            "data2_p": [None]
        }
        self.assertAllEqual(shapes_must,
                            shapes)
        self.assertSetEqual(set(shapes_must),
                            set(dataset.generated_keys_all))

        self.assertEqual(111,
                         dataset.random_seed)
        self.assertIsInstance(dataset,
                              DatasetFileList)
        self.evaluate(result_iterator.initializer)
        _ = self.evaluate(batch)


class TestTfRecordsDatasetFromPipe(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.num_samples = 10
        self.tfrecords_file_name = os.path.join(
            self.get_temp_dir(), "data.tfrecords")
        self.data_tfrecords = [
            {"data1": np.random.rand(1).astype(np.float32),
             "data2": np.random.rand(
                 np.random.randint(1, 5), 20).astype(np.float32),
             "data3": np.random.rand(
                 np.random.randint(1, 10), 1).astype(np.float32)}
            for _ in range(self.num_samples)]

    def test_call(self):
        file_name = self.tfrecords_file_name
        writer = tf.python_io.TFRecordWriter(file_name)
        for each_sample in self.data_tfrecords:
            write_tf_records(each_sample, file_name, writer, False)
        writer.close()

        output_keys_mapping = {"processor2": {"data1_p": "data2_p"}}
        tf.reset_default_graph()
        reader = TfRecordsDataReaderDummy(
            name="reader",
            file_list_keys_mapping={"data": "tfrecords"}).build()
        processor1 = _DataProcessorTF(
            inbound_nodes=["reader"], name="processor1",
            add_num=10.0,
        ).build()
        processor2 = _DataProcessorTF(
            inbound_nodes=["reader"], name="processor2",
            incoming_keys_mapping={"reader": {"data2": "data1"}},
            add_num=-20.0,
        ).build()
        data_pipe = DataPipe(readers=[reader],
                             processors=[processor1, processor2]).build()

        file_list = FileList.from_matched_file_names(
            {"tfrecords": [self.tfrecords_file_name]})
        dataset = Dataset.from_data_pipe(
            data_pipe=data_pipe, file_list=file_list,
            output_keys_mapping=output_keys_mapping,
            random_seed=111).build()
        dataset.mode = "train"
        result = dataset(batch_size=4)
        result_iterator = result.make_initializable_iterator()
        batch = result_iterator.get_next()

        shapes = {k: v.as_list() for k, v in result.output_shapes.items()}
        shapes_must = {
            "data_default": [None, 1],
            "data1": [None, 1],
            "data2": [None, None, 20],
            "data3": [None, None, 1],
            "data1_p": [None, 1],
            "data2_p": [None, None, 20]
        }
        self.assertAllEqual(shapes_must,
                            shapes)
        self.assertSetEqual(set(shapes_must),
                            set(dataset.generated_keys_all))

        self.assertEqual(111,
                         dataset.random_seed)
        self.assertIsInstance(dataset,
                              DatasetTfRecords)
        self.evaluate(result_iterator.initializer)
        _ = self.evaluate(batch)


class _DatasetTfRecordsDummy(DatasetTfRecords):
    exclude_from_register = True

    def get_tfrecords_features(self):
        features = {
            'input_to_decode': tf.FixedLenFeature([], tf.string),
            'input_number': {'floats': tf.FixedLenFeature([], tf.float32),
                             'int': tf.FixedLenFeature([], tf.int64)}
        }
        return features

    def get_tfrecords_output_types(self):
        return {'input_to_decode': tf.float32}


class _DummyDataFilterOddData(DataFilter):
    exclude_from_register = True

    def predicate(self, value) -> bool:
        return value % 2


class _DatasetDummy(Dataset):
    exclude_from_register = True

    def create_initial_data(self):
        inputs = {'value': np.arange(10).astype(np.float32)}
        data = tf.data.Dataset.from_tensor_slices(inputs)
        return data


class _DataGenerator1(DataProcessor):
    exclude_from_register = True
    generated_keys = ["data11", "data12"]
    is_tensorflow = True

    def process(self, **inputs):
        return {"data11": tf.random_normal([10, 5]),
                "data12": tf.random_normal([1])}


class _DataGenerator2(DataProcessor):
    exclude_from_register = True
    generated_keys = ["data21", "data22"]
    is_tensorflow = True

    def process(self, **inputs):
        return {"data21": tf.random_normal([1, 3]),
                "data22": tf.random_normal([5])}


class _DataProcessorTF(DataProcessor):
    exclude_from_register = True
    incoming_keys = ["data1"]
    generated_keys = ["data1_p"]
    is_tensorflow = True

    def __init__(self, add_num=None, **kwargs):
        super(_DataProcessorTF, self).__init__(**kwargs)
        self.add_num = add_num

    def process(self, data1):
        if self.add_num is None:
            add_num = tf.random_normal(tf.shape(data1))
        else:
            add_num = self.add_num
        return {"data1_p": data1 + add_num}


def _select_sample_at_index(data: dict, index: int):
    sample = {}
    for each_key, each_value in data.items():
        if isinstance(each_value, dict):
            sample[each_key] = _select_sample_at_index(data[each_key], index)
        else:
            sample[each_key] = data[each_key][index]
    return sample


def _get_dataset_id_from_sample(sample: dict, datas):
    for i, each_data in enumerate(datas):
        if _is_sample_in_data(sample, each_data):
            return i
    raise ValueError('Sample does not belong to data!')


def _is_sample_in_data(sample, data):
    for each_key, each_value in sample.items():
        if isinstance(each_value, dict):
            if _is_sample_in_data(each_value, data[each_key]):
                continue
            else:
                return False
        if each_value in data[each_key]:
            continue
        return False
    return True


def _interleave(input1, input2, max_length):
    if not isinstance(input1, list):
        input1 = list(input1)
    input1_repeated = input1 * max_length
    if not isinstance(input2, list):
        input2 = list(input2)
    input2_repeated = input2 * max_length
    interleaved = list(chain.from_iterable(
        zip(input1_repeated, input2_repeated)))
    return interleaved[:max_length]


def _get_data_results(data: tf.data.Dataset, session_manager,
                      max_iteration=None) -> dict:
    iterator = data.make_one_shot_iterator()
    sample = iterator.get_next()
    outputs_flatten = {}
    iteration_number = 0
    with session_manager as sess:
        while True:
            try:
                sample_out = sess.run(sample)
                sample_out_flatten = nest_utils.flatten_nested_struct(
                    sample_out)
                for k, v in sample_out_flatten.items():
                    outputs_flatten.setdefault(k, [])
                    if isinstance(v, bytes):
                        v = v.decode()
                    outputs_flatten[k].append(v)
                iteration_number += 1
            except tf.errors.OutOfRangeError:
                break

            if max_iteration is not None and iteration_number >= max_iteration:
                break
    outputs = nest_utils.unflatten_dict_to_nested(outputs_flatten)
    return outputs


def _add_suffix_to_each_value_in_dict_of_lists(inputs: dict,
                                               suffix: str) -> dict:
    outputs = {k: [v + suffix for v in v_list]
               for k, v_list in inputs.items()}
    return outputs


def _create_serialized_example_from_sample(sample: dict) -> tf.train.Example:
    """
    create a serialized tf.train.example from a sample
    """

    feature = {
        'input_to_decode': tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[sample['input_to_decode'].tostring()])),
        'input_number/floats': tf.train.Feature(
            float_list=tf.train.FloatList(
                value=[sample['input_number/floats']])
        ),
        'input_number/int': tf.train.Feature(
            int64_list=tf.train.Int64List(
                value=[sample['input_number/int']])
        )
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def _write_sample_to_tfrecord(
        sample: dict, tfrecord_writer: tf.python_io.TFRecordWriter) -> None:
    """
    sample is a dict with a key:value pair
    tfrecord_writer is an open tf.python_io.TFRecordWriter
    """
    example = _create_serialized_example_from_sample(sample)
    tfrecord_writer.write(example)
