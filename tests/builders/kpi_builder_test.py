# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import itertools

from absl.testing import parameterized

import nucleus7 as nc7
from nucleus7.builders import kpi_builder
from nucleus7.core import project_global_config
from nucleus7.test_utils.test_utils import TestCaseWithTempDir
from nucleus7.test_utils.test_utils import register_new_class
from nucleus7.test_utils.test_utils import reset_register_and_logger


class _KPICacher(nc7.kpi.KPICacher):
    def __init__(self, p=0):
        super(_KPICacher, self).__init__()
        self.p = p


class _KPICacher2(nc7.kpi.KPICacher):
    def __init__(self, p2=0):
        super(_KPICacher2, self).__init__()
        self.p2 = p2


class _KPISaver(nc7.kpi.KPISaver):
    def __init__(self, p3=0):
        super(_KPISaver, self).__init__()
        self.p3 = p3


class _KPISaver2(nc7.kpi.KPISaver):
    def __init__(self, p4=0):
        super(_KPISaver2, self).__init__()
        self.p4 = p4


class _KPIPlugin(nc7.kpi.KPIPlugin):
    exclude_from_register = True

    def __init__(self, parameter1=1, **kwargs):
        self.parameter1 = parameter1
        super().__init__(**kwargs)


class _KPIAccumulator(nc7.kpi.KPIAccumulator):
    exclude_from_register = True

    def __init__(self, parameter2=2, **kwargs):
        self.parameter2 = parameter2
        super().__init__(**kwargs)


class TestKPIEvaluatorBuilder(TestCaseWithTempDir, parameterized.TestCase):

    def setUp(self):
        reset_register_and_logger()
        register_new_class("plugin1", _KPIPlugin)
        register_new_class('plugin2', _KPIPlugin)
        register_new_class("accumulator1", _KPIAccumulator)
        register_new_class("accumulator2", _KPIAccumulator)

        self.plugins_and_accumulators_configs = [
            {"class_name": "plugin1", "parameter1": 10},
            {"class_name": "plugin2", "parameter1": 20,
             "cachers": [{"class_name": "kpi_builder_test._KPICacher"}]},
            {"class_name": "accumulator1", "parameter2": 100,
             "cachers": [
                 {"class_name": "kpi_builder_test._KPICacher", "p": 10}],
             "savers": [{"class_name": "kpi_builder_test._KPISaver", "p3": 5}]},
            {"class_name": "accumulator2", "parameter2": 200,
             "savers": [
                 {"class_name": "kpi_builder_test._KPISaver2", "p4": 50}]},
        ]
        self.project_global_config = {
            "KPIPlugin": {
                "cachers": [
                    {"class_name": "kpi_builder_test._KPICacher2", "p2": 5}],
                "savers": [
                    {"class_name": "kpi_builder_test._KPISaver2", "p4": 100}]
            }
        }
        project_global_config.clear_project_global_config()
        super().setUp()

    def tearDown(self):
        super().tearDown()
        project_global_config.clear_project_global_config()

    @parameterized.parameters({"with_project_global_config": True},
                              {"with_project_global_config": False})
    def test_build_kpi_plugins(self, with_project_global_config):
        if with_project_global_config:
            project_global_config.add_global_project_config(
                self.project_global_config)
        plugins_and_accumulators = kpi_builder.build_kpi_plugins(
            self.plugins_and_accumulators_configs)
        self.assertLen(plugins_and_accumulators, 4)
        base_cls_must = [_KPIPlugin, _KPIPlugin,
                         _KPIAccumulator, _KPIAccumulator]
        attributes_must = [
            {"parameter1": 10},
            {"parameter1": 20},
            {"parameter2": 100},
            {"parameter2": 200},
        ]
        cachers_must = [
            [],
            [(_KPICacher, {"p": 0})],
            [(_KPICacher, {"p": 10})],
            [],
        ]
        savers_must = [
            [],
            [],
            [(_KPISaver, {"p3": 5})],
            [(_KPISaver2, {"p4": 50})],
        ]
        if with_project_global_config:
            cachers_must[0] = [(_KPICacher2, {"p2": 5})]
            cachers_must[-1] = [(_KPICacher2, {"p2": 5})]
            savers_must[0] = [(_KPISaver2, {"p4": 100})]
            savers_must[1] = [(_KPISaver2, {"p4": 100})]

        for i in range(4):
            plugin_i = plugins_and_accumulators[i]
            self.assertTrue(plugin_i.built)
            self.assertIsInstance(plugin_i, base_cls_must[i])
            for each_attr_name_must, each_attr_value_must in (
                    attributes_must[i].items()):
                self.assertEqual(each_attr_value_must,
                                 getattr(plugin_i, each_attr_name_must))
            for each_saver, each_saver_must in itertools.zip_longest(
                    plugin_i.savers, savers_must[i]):
                self.assertIsInstance(each_saver,
                                      each_saver_must[0])
                for each_attr_name_must, each_attr_value_must in (
                        each_saver_must[1].items()):
                    self.assertEqual(each_attr_value_must,
                                     getattr(each_saver, each_attr_name_must))

            for each_cacher, each_cacherr_must in itertools.zip_longest(
                    plugin_i.cachers, cachers_must[i]):
                self.assertIsInstance(each_cacher,
                                      each_cacherr_must[0])
                for each_attr_name_must, each_attr_value_must in (
                        each_cacherr_must[1].items()):
                    self.assertEqual(each_attr_value_must,
                                     getattr(each_cacher, each_attr_name_must))

    def test_build_kpi_evaluator(self):
        plugin1 = _KPIPlugin(name="plugin1")
        plugin2 = _KPIAccumulator(name="plugin2")
        plugin3 = _KPIAccumulator(name="plugin3")
        plugin4 = _KPIPlugin(name="plugin4")
        plugins_and_accumulators = [plugin1,
                                    plugin2,
                                    plugin3,
                                    plugin4]
        kpi_evaluator = kpi_builder.build_kpi_evaluator(
            plugins_and_accumulators)
        self.assertTrue(kpi_evaluator.built)
        self.assertEqual({"plugin1": plugin1,
                          "plugin4": plugin4},
                         kpi_evaluator.plugins)
        self.assertEqual({"plugin2": plugin2,
                          "plugin3": plugin3},
                         kpi_evaluator.accumulators)

    def test_build_kpi_evaluator_as_callback(self):
        pass

    # @parameterized.parameters({"with_filter": False},
    #                           {"with_filter": True})
    # def test_build(self, with_filter):
    #     save_dir = self.get_temp_dir()
    #     evaluator_config = {'class_name': 'evaluator',
    #                         'save_dir': save_dir,
    #                         "file_list": {'file_names': self.file_names,
    #                                       'class_name': "FileListDummy"},
    #                         "parameter1": 20}
    #     if with_filter:
    #         evaluator_config["data_filter"] = {
    #             "class_name": "data_filter1", "dp1": 1}
    #
    #     evaluator = kpi_builder.build(evaluator_config)
    #     self.assertIsInstance(evaluator, _KPIEvaluator)
    #     self.assertDictEqual(self.file_names,
    #                          evaluator.file_list.get())
    #     self.assertEqual(20,
    #                      evaluator.parameter1)
    #     if with_filter:
    #         self.assertEqual(1,
    #                          len(evaluator.data_filters))
    #         data_filter = evaluator.data_filters[0]
    #         self.assertTrue(data_filter.built)
    #         self.assertIsInstance(data_filter, _DummyDataFilter)
    #         self.assertEqual(1,
    #                          data_filter.dp1)
