# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import nucleus7 as nc7
from nucleus7.builders import data_pipe_builder
from nucleus7.data.data_pipe import DataPipe
from nucleus7.test_utils import test_utils


class TestDataPipeBuilder(test_utils.TestCaseWithReset):

    def setUp(self):
        super(TestDataPipeBuilder, self).setUp()
        test_utils.register_new_class("reader1", nc7.data.DataReader)
        test_utils.register_new_class("reader2", nc7.data.DataReader)
        test_utils.register_new_class("processor1", nc7.data.DataProcessor)
        test_utils.register_new_class("processor2", nc7.data.DataProcessor)
        test_utils.register_new_class("nucleotide1", nc7.core.Nucleotide)

    def test_build(self):
        reader1 = nc7.data.DataReader(name="reader1_name")
        reader2 = nc7.data.DataReader(name="reader2_name")
        processor1 = nc7.data.DataProcessor(name="processor1_name")
        processor2 = nc7.data.DataProcessor(name="processor2_name")
        data_pipe = data_pipe_builder.build(readers=[reader1, reader2],
                                            processors=[processor1, processor2])
        self.assertTrue(data_pipe.built)
        self.assertIsInstance(data_pipe, DataPipe)
        self.assertSetEqual({"reader1_name", "reader2_name"},
                            {r.name for r in data_pipe.readers.values()})
        self.assertSetEqual({"processor1_name", "processor2_name"},
                            {r.name for r in data_pipe.processors.values()})

    def test_build_data_pipe_from_configs(self):
        reader_config1 = {"class_name": "reader1", "name": "reader1_name"}
        reader_config2 = {"class_name": "reader2", "name": "reader2_name"}
        processor_config1 = {
            "class_name": "processor1", "name": "processor1_name"}
        processor_config2 = {
            "class_name": "processor2", "name": "processor2_name"}

        configs = [processor_config1, reader_config1, reader_config2,
                   processor_config2]
        data_pipe = data_pipe_builder.build_data_pipe_from_configs(configs)
        self.assertTrue(data_pipe.built)
        self.assertIsInstance(data_pipe, DataPipe)
        self.assertSetEqual({"reader1_name", "reader2_name"},
                            {r.name for r in data_pipe.readers.values()})
        self.assertSetEqual({"processor1_name", "processor2_name"},
                            {r.name for r in data_pipe.processors.values()})

        configs_wrong = configs + [{"class_name": "nucleotide1"}]
        with self.assertRaises(ValueError):
            data_pipe_builder.build_data_pipe_from_configs(configs_wrong)
