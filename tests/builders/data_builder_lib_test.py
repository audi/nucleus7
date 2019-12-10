# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from absl.testing import parameterized

from nucleus7.builders import data_builder_lib
from nucleus7.data.data_filter import DataFilter
from nucleus7.data.data_pipe import DataPipe
from nucleus7.data.dataset import Dataset
from nucleus7.data.dataset import DatasetFileList
from nucleus7.data.processor import DataProcessor
from nucleus7.data.reader import DataReader
from nucleus7.test_utils.model_dummies import FileListDummy
from nucleus7.test_utils.test_utils import register_new_class
from nucleus7.test_utils.test_utils import reset_register_and_logger


class _DummyDataset(Dataset):
    exclude_from_register = True

    def __init__(self, p1=10, p2=20, **kwargs):
        super().__init__(**kwargs)
        self.p1 = p1
        self.p2 = p2


class _DummyDatasetFileList(DatasetFileList):
    exclude_from_register = True

    def __init__(self, file_list, p1=10, p2=20, **kwargs):
        super().__init__(file_list=file_list, **kwargs)
        self.p1 = p1
        self.p2 = p2


class _DummyDataFilter(DataFilter):
    exclude_from_register = True

    def __init__(self, dp1=20):
        super().__init__()
        self.dp1 = dp1


class _DataReaderTF(DataReader):
    exclude_from_register = True
    is_tensorflow = True


class _DataProcessorTF(DataProcessor):
    exclude_from_register = True
    is_tensorflow = True


class TestDataBuilderLib(parameterized.TestCase):

    def setUp(self):
        reset_register_and_logger()
        register_new_class("data_filter1", _DummyDataFilter)
        register_new_class("file_list1", FileListDummy)
        register_new_class("reader1", _DataReaderTF)
        register_new_class("reader2", _DataReaderTF)
        register_new_class("processor1", _DataProcessorTF)
        register_new_class("processor2", _DataProcessorTF)

    @parameterized.parameters(
        {"with_file_list": True, "with_data_filter": True},
        {"with_file_list": False, "with_data_filter": False,
         "with_data_pipe": True},
        {"with_file_list": True, "with_data_filter": True,
         "with_data_pipe": True},
        {"with_file_list": True, "with_data_filter": True,
         "with_data_pipe": True, "with_file_list_mapping": True},
        {"with_file_list": False, "with_data_filter": True},
        {"with_file_list": True, "with_data_filter": False},
        {"with_file_list": True, "with_data_filter": False,
         "with_file_list_mapping": True},
        {"with_file_list": False, "with_data_filter": False})
    def test_build_data_object_from_config(
            self, with_file_list,
            with_data_filter,
            with_data_pipe=False,
            with_file_list_mapping=False):
        config_object = {"random_seed": 65477}
        if with_data_pipe:
            built_fn = lambda x: Dataset.from_data_pipe(**x).build()
            dataset_base_cls = Dataset
        else:
            if with_file_list:
                register_new_class("dummy_dataset", _DummyDatasetFileList)
                dataset_base_cls = _DummyDatasetFileList
                built_fn = None
            else:
                register_new_class("dummy_dataset", _DummyDataset)
                dataset_base_cls = _DummyDataset
                built_fn = None

            config_object.update({"class_name": "dummy_dataset",
                                  "p1": 100})
        file_list_keys_mapping = {"key1": "key1_r"}
        if with_file_list:
            config_object["file_list"] = {"class_name": "file_list1",
                                          "file_names": {"key1": ["value1"]},
                                          "name": "file_list_name"}
            _DummyDatasetFileList.file_list_keys = ["key1"]
            if with_file_list_mapping:
                config_object["file_list_keys_mapping"] = {"key1": "key1_r"}
                _DummyDatasetFileList.file_list_keys = ["key1_r"]
        if with_data_filter:
            config_object["data_filter"] = {"class_name": "data_filter1",
                                            "dp1": 1}

        if with_data_pipe:
            reader_config1 = {"class_name": "reader1", "name": "reader1_name"}
            reader_config2 = {"class_name": "reader2", "name": "reader2_name"}
            processor_config1 = {
                "class_name": "processor1", "name": "processor1_name"}
            processor_config2 = {
                "class_name": "processor2", "name": "processor2_name"}
            if with_file_list:
                config_object["data_pipe"] = [processor_config1, reader_config1,
                                              reader_config2, processor_config2]
            else:
                config_object["data_pipe"] = [processor_config1,
                                              processor_config2]
        if with_data_pipe and with_file_list_mapping:
            with self.assertRaises(ValueError):
                data_builder_lib.build_data_object_from_config(
                    config_object, base_cls=dataset_base_cls, built_fn=built_fn)
            return

        built_object = data_builder_lib.build_data_object_from_config(
            config_object, base_cls=dataset_base_cls, built_fn=built_fn)
        self.assertTrue(built_object.built)
        self.assertEqual(65477,
                         built_object.random_seed)

        self.assertIsInstance(built_object, dataset_base_cls)
        if with_data_filter:
            self.assertEqual(1,
                             len(built_object.data_filters))
            data_filter = built_object.data_filters[0]
            self.assertTrue(data_filter.built)
            self.assertIsInstance(data_filter, _DummyDataFilter)
            self.assertEqual(1,
                             data_filter.dp1)
        else:
            self.assertIsNone(built_object.data_filters)

        if with_file_list:
            file_list = built_object.file_list
            self.assertIsInstance(file_list, FileListDummy)
            self.assertTrue(file_list.built)
            self.assertEqual("file_list_name",
                             file_list.name)
            file_names_must = ({"key1_r": ["value1"]} if with_file_list_mapping
                               else {"key1": ["value1"]})
            self.assertDictEqual(file_names_must,
                                 file_list.get())
            if with_file_list_mapping:
                self.assertDictEqual(file_list_keys_mapping,
                                     built_object.file_list_keys_mapping)
            else:
                self.assertIsNone(built_object.file_list_keys_mapping)
        else:
            self.assertFalse(hasattr(built_object, "file_list"))
            self.assertFalse(hasattr(built_object, "file_list_keys_mapping"))

        if with_data_pipe:
            data_pipe = built_object.data_pipe
            self.assertIsInstance(data_pipe, DataPipe)
            self.assertTrue(data_pipe.built)
            if with_file_list:
                self.assertSetEqual({"reader1_name", "reader2_name"},
                                    {r.name for r in
                                     data_pipe.readers.values()})
            else:
                self.assertDictEqual({},
                                     data_pipe.readers)
            self.assertSetEqual({"processor1_name", "processor2_name"},
                                {r.name for r in
                                 data_pipe.processors.values()})
