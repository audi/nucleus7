# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import copy

from absl.testing import parameterized

import nucleus7 as nc7
from nucleus7.builders import file_list_builder
from nucleus7.data.data_filter import DataFilter
from nucleus7.test_utils.model_dummies import FileListDummy
from nucleus7.test_utils.test_utils import register_new_class
from nucleus7.test_utils.test_utils import reset_register_and_logger


class TestFileListBuilder(parameterized.TestCase):

    def setUp(self):
        reset_register_and_logger()
        self.number_of_files1 = 20
        self.number_of_files2 = 10
        self.number_of_files3 = 10
        self.file_names1 = {'input1': ['input_1_{:03d}.ext'.format(i)
                                       for i in range(self.number_of_files1)],
                            'input2': ['input_2_{:03d}.ext'.format(i)
                                       for i in range(self.number_of_files1)]}
        self.file_names2 = {'input1': ['input2_1_{:03d}.ext'.format(i)
                                       for i in range(self.number_of_files2)],
                            'input2': ['input2_2_{:03d}.ext'.format(i)
                                       for i in range(self.number_of_files2)]}

        self.file_names3 = {'input1': ['input3_1_{:03d}.ext'.format(i)
                                       for i in range(self.number_of_files3)],
                            'input3': ['input_3_{:03d}.ext'.format(i)
                                       for i in range(self.number_of_files3)]}
        self.file_names_1_2_3 = self._add_file_names()

    def test_build_single_file_list(self):
        register_new_class('file_list1', FileListDummy)
        config_file_list = {'class_name': 'file_list1',
                            'file_names': self.file_names1}
        file_list = file_list_builder.build(config_file_list)
        self.assertIsInstance(file_list, nc7.data.FileList)
        self.assertEqual(self.number_of_files1,
                         len(file_list))
        self.assertDictEqual(self.file_names1,
                             file_list.get())

    @parameterized.parameters({"single_data_filter": True},
                              {"single_data_filter": False})
    def test_build_file_list_with_filter(self, single_data_filter):
        register_new_class('file_list1', FileListDummy)

        class _DummyDataFilter1(DataFilter):
            pass

        class _DummyDataFilter2(DataFilter):
            pass

        if single_data_filter:
            data_filter_config = {
                "data_filter": {"class_name": "_DummyDataFilter1"}}
        else:
            data_filter_config = {
                "data_filter": [{"class_name": "_DummyDataFilter1"},
                                {"class_name": "_DummyDataFilter2"},
                                {"class_name": "_DummyDataFilter2"}]}

        config_file_list = {"class_name": "file_list1",
                            "file_names": self.file_names1}
        config_file_list.update(data_filter_config)
        file_list = file_list_builder.build(config_file_list)

        number_of_data_filters_must = single_data_filter and 1 or 3
        self.assertEqual(number_of_data_filters_must,
                         len(file_list.data_filters))

        data_filter_types_must = (
                single_data_filter and [_DummyDataFilter1] or
                [_DummyDataFilter1, _DummyDataFilter2, _DummyDataFilter2])
        for each_data_filter, each_data_filter_type_must in zip(
                file_list.data_filters, data_filter_types_must):
            self.assertIsInstance(each_data_filter, each_data_filter_type_must)

    def test_build_list_of_file_list(self):
        register_new_class('file_list2', FileListDummy)
        register_new_class('file_list3', FileListDummy)
        config_file_list = [{'class_name': 'file_list2',
                             'file_names': self.file_names1},
                            {'class_name': 'file_list2',
                             'file_names': self.file_names2},
                            {'class_name': 'file_list3',
                             'file_names': self.file_names3}
                            ]
        file_list = file_list_builder.build(config_file_list)
        self.assertIsInstance(file_list, nc7.data.FileList)
        self.assertEqual(self.number_of_files1
                         + self.number_of_files2
                         + self.number_of_files3,
                         len(file_list))
        file_names_1_2_3 = {}
        for each_key in self.file_names1.keys():
            file_names_1_2_3[each_key] = (self.file_names1[each_key]
                                          + self.file_names2[each_key])
        self.assertDictEqual(self.file_names_1_2_3,
                             file_list.get())

    def _add_file_names(self):
        file_names_1_2_must = {}
        for each_key in self.file_names1.keys():
            file_names_1_2_must[each_key] = (self.file_names1[each_key]
                                             + self.file_names2[each_key])
        empty_list_1_2 = [""] * (self.number_of_files1 + self.number_of_files2)
        empty_list_3 = [""] * self.number_of_files3
        file_names_1_2_3_must = copy.deepcopy(file_names_1_2_must)
        all_keys_1_2_3 = set(list(file_names_1_2_must)
                             + list(self.file_names3))

        for each_key in all_keys_1_2_3:
            if each_key not in file_names_1_2_must:
                file_names_1_2_3_must[each_key] = empty_list_1_2
            if each_key in self.file_names3:
                file_names_1_2_3_must[each_key].extend(
                    self.file_names3[each_key])
            else:
                file_names_1_2_3_must[each_key].extend(empty_list_3)
        return file_names_1_2_3_must
