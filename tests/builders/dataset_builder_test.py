# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from absl.testing import parameterized

import nucleus7 as nc7
from nucleus7.builders import dataset_builder
from nucleus7.test_utils.model_dummies import FileListDummy
from nucleus7.test_utils.test_utils import register_new_class
from nucleus7.test_utils.test_utils import reset_register_and_logger


class _DatasetWithFileList(nc7.data.DatasetFileList):
    file_list_keys = ['data']

    def __init__(self, parameter2, **kwargs):
        self.parameter2 = parameter2
        super().__init__(**kwargs)


class _DatasetWithoutFileList(nc7.data.Dataset):

    def __init__(self, parameter1, **kwargs):
        self.parameter1 = parameter1
        super().__init__(**kwargs)


class TestDatasetBuilder(parameterized.TestCase):

    def setUp(self):
        reset_register_and_logger()
        register_new_class("dataset_with_file_list", _DatasetWithFileList)
        register_new_class("dataset_without_file_list",
                           _DatasetWithoutFileList)
        register_new_class("FileListDummy",
                           FileListDummy)
        self.number_of_files1 = 20
        self.number_of_files2 = 10
        self.file_names1 = {'data': ['input_1_{:03d}.ext'.format(i)
                                     for i in range(self.number_of_files1)]}
        self.file_names2 = {'data': ['input2_1_{:03d}.ext'.format(i)
                                     for i in range(self.number_of_files2)]}

    def test_build_single_without_file_list(self):
        dataset_config = {'class_name': 'dataset_without_file_list',
                          'parameter1': 10}
        dataset = dataset_builder.build(dataset_config)
        self.assertIsInstance(dataset, _DatasetWithoutFileList)
        self.assertEqual(10,
                         dataset.parameter1)

    @parameterized.parameters({"with_file_list_mapping": True},
                              {"with_file_list_mapping": False})
    def test_build_single_with_file_list(self, with_file_list_mapping):
        if with_file_list_mapping:
            file_names = {"data1": self.file_names1["data"]}
        else:
            file_names = self.file_names1

        dataset_config = {'class_name': 'dataset_with_file_list',
                          'file_list': {'file_names': file_names,
                                        'class_name': "FileListDummy"},
                          'parameter2': 20}
        if with_file_list_mapping:
            dataset_config["file_list_keys_mapping"] = {"data1": "data"}
        dataset = dataset_builder.build(dataset_config)
        self.assertIsInstance(dataset, _DatasetWithFileList)
        self.assertDictEqual(self.file_names1,
                             dataset.file_list.get())
        self.assertEqual(20,
                         dataset.parameter2)

    def test_build_mix(self):
        dataset_configs = [
            {'class_name': 'dataset_without_file_list',
             'sampling_weight': 2.0,
             'merge_on_same_file_list': False,
             'parameter1': 10,
             'subtype': 'subtype1'},
            {'class_name': 'dataset_with_file_list',
             "file_list": {'file_names': self.file_names1,
                           'class_name': "FileListDummy"},
             'parameter2': 30,
             'subtype': 'subtype2'}

        ]
        dataset = dataset_builder.build(dataset_configs)
        self.assertIsInstance(dataset,
                              nc7.data.DatasetMix)
        self.assertListEqual([2.0, 1.0],
                             dataset.sampling_weights)
        self.assertListEqual([False, True],
                             dataset.merge_on_same_file_list)
        self.assertIsInstance(dataset.datasets[0],
                              _DatasetWithoutFileList)
        self.assertIsInstance(dataset.datasets[1],
                              _DatasetWithFileList)
        self.assertDictEqual(self.file_names1,
                             dataset.datasets[1].file_list.get())
        self.assertEqual(10,
                         dataset.datasets[0].parameter1)
        self.assertEqual(30,
                         dataset.datasets[1].parameter2)
