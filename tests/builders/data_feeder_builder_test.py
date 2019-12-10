# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from absl.testing import parameterized

import nucleus7 as nc7
from nucleus7.builders import data_feeder_builder
from nucleus7.test_utils.model_dummies import FileListDummy
from nucleus7.test_utils.test_utils import register_new_class
from nucleus7.test_utils.test_utils import reset_register_and_logger


class _DataFeederWithFileList(nc7.data.DataFeederFileList):
    file_list_keys = ['data']

    def __init__(self, parameter1, **kwargs):
        self.parameter1 = parameter1
        super().__init__(**kwargs)


class _DataFeederWithoutFileList(nc7.data.DataFeeder):

    def __init__(self, parameter1, **kwargs):
        self.parameter1 = parameter1
        super().__init__(**kwargs)


class TestDataFeederBuilder(parameterized.TestCase):

    def setUp(self):
        reset_register_and_logger()
        register_new_class("datafeeder_with_file_list",
                           _DataFeederWithFileList)
        register_new_class("datafeeder_without_file_list",
                           _DataFeederWithoutFileList)
        register_new_class("FileListDummy",
                           FileListDummy)
        self.number_of_files = 20
        self.file_names = {'data': ['input_1_{:03d}.ext'.format(i)
                                    for i in range(self.number_of_files)]}

    def test_build_without_file_list(self):
        data_feeder_config = {'class_name': 'datafeeder_without_file_list',
                              'parameter1': 10}
        data_feeder = data_feeder_builder.build(data_feeder_config)
        self.assertIsInstance(data_feeder, _DataFeederWithoutFileList)
        self.assertEqual(10,
                         data_feeder.parameter1)

    @parameterized.parameters({"with_file_list_mapping": True},
                              {"with_file_list_mapping": False})
    def test_build_with_file_list(self, with_file_list_mapping):
        if with_file_list_mapping:
            file_names = {"data1": self.file_names["data"]}
        else:
            file_names = self.file_names
        data_feeder_config = {"class_name": 'datafeeder_with_file_list',
                              "file_list": {'file_names': file_names,
                                            'class_name': "FileListDummy"},
                              "parameter1": 20}
        if with_file_list_mapping:
            data_feeder_config["file_list_keys_mapping"] = {"data1": "data"}
        data_feeder = data_feeder_builder.build(data_feeder_config)
        self.assertIsInstance(data_feeder, _DataFeederWithFileList)
        self.assertDictEqual(self.file_names,
                             data_feeder.file_list.get())
        self.assertEqual(20,
                         data_feeder.parameter1)
