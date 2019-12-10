# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import copy
import unittest

import nucleus7 as nc7
from nucleus7.core.project_global_config import clear_project_global_config
from nucleus7.data import Dataset
from nucleus7.model import Model
from nucleus7.model import ModelPlugin


class TestProjectGlobalConfig(unittest.TestCase):

    def _get_config(self):
        global_config = {
            'param1': 10,
            'param2': 100,
            'param3': 200,
            'ModelPlugin': {
                'param1': 11,
                'param4': 100
            },
            'Dataset': {
                'param1': 200,
                'ds_param': 150,
                'ds_param2': 5
            },
            'DatasetChild': {
                'param1': 300,
                'ds_param': 200,
                'param3': 1,
            }
        }

        class DatasetChild(Dataset):
            def __init__(self, param1, param3, ds_param, ds_param2,
                         **kwargs):
                super().__init__(**kwargs)
                self.param1 = param1
                self.param3 = param3
                self.ds_param = ds_param
                self.ds_param2 = ds_param2

        class ModelPluginChild(ModelPlugin):
            def __init__(self, param1, param3, **kwargs):
                super().__init__(**kwargs)
                self.param1 = param1
                self.param3 = param3

        nucleotides = [DatasetChild, ModelPluginChild]
        dataset_default_params = {'ds_param': 200, 'ds_param2': 5,
                                  'param1': 300, 'param3': 1}
        plugin_default_params = {'param1': 11, 'param3': 200}
        default_params = [dataset_default_params, plugin_default_params]

        return global_config, nucleotides, default_params

    def test_query_nucleotide_parameters(self):
        global_config, nucleotides, default_params_must = self._get_config()
        nc7.project_global_config.add_global_project_config(global_config)
        dataset_params = nc7.query_nucleotide_global_config(
            nucleotides[0])
        plugin_params = nc7.query_nucleotide_global_config(
            nucleotides[1])
        model_params = nc7.query_nucleotide_global_config(Model)
        default_params = [dataset_params, plugin_params]

        self.assertIsNone(model_params)
        self.assertListEqual(default_params_must, default_params)

        clear_project_global_config()
        dataset_params = nc7.query_nucleotide_global_config(
            nucleotides[0])
        plugin_params = nc7.query_nucleotide_global_config(
            nucleotides[1])
        model_params = nc7.query_nucleotide_global_config(
            Model)
        self.assertIsNone(dataset_params)
        self.assertIsNone(plugin_params)
        self.assertIsNone(model_params)

    def test_add_defaults_to_nucleotide(self):
        global_config, nucleotides, default_params_must = self._get_config()
        nc7.project_global_config.add_global_project_config(global_config)
        new_params = [{'ds_param': 10, 'ds_param10': 54},
                      {'param1': 1, 'param8': 11}]
        params_model = {'param1': 15, 'param3': 4567, 'new_param': 10}

        for default_params, new_params, nucl_cls in zip(
                default_params_must, new_params, nucleotides):
            new_params_updated = nc7.project_global_config.add_defaults_to_nucleotide(
                nucl_cls, new_params)
            new_params_updated_must = copy.deepcopy(default_params)
            new_params_updated_must.update(new_params)
            self.assertDictEqual(new_params_updated_must, new_params_updated)

        model_params = nc7.project_global_config.add_defaults_to_nucleotide(
            Model, params_model)
        self.assertDictEqual(params_model, model_params)
