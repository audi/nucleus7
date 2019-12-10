# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from collections import namedtuple
import copy
from unittest.mock import MagicMock

from absl.testing import parameterized

import nucleus7 as nc7
from nucleus7.builders import builder_lib
from nucleus7.test_utils.test_utils import TestCaseWithReset
from nucleus7.test_utils.test_utils import register_new_class


class TestBuilderLib(TestCaseWithReset, parameterized.TestCase):

    def test_build_registered_object(self):
        register_new_class('dummy_plugin', nc7.model.ModelPlugin)
        name = "plugin_name"
        inbound_nodes = ['node1', 'node2']
        plugin = builder_lib.build_registered_object(
            base_cls=nc7.model.ModelPlugin, class_name="dummy_plugin",
            name=name, inbound_nodes=inbound_nodes)
        self.assertIsInstance(plugin, nc7.model.ModelPlugin)
        self.assertTrue(plugin.build)
        self.assertEqual(plugin.name, name)
        self.assertListEqual(plugin.inbound_nodes, inbound_nodes)

        with self.assertRaises(ValueError):
            plugin2 = builder_lib.build_registered_object(
                base_cls=nc7.model.ModelMetric, class_name="dummy_plugin",
                name=name, inbound_nodes=inbound_nodes)

        plugin3 = builder_lib.build_registered_object(
            base_cls=nc7.model.ModelMetric, default_cls=nc7.model.ModelPlugin,
            inbound_nodes=inbound_nodes)
        self.assertIsInstance(plugin3, nc7.model.ModelPlugin)
        self.assertEqual(plugin3.name, plugin3.__class__.__name__)

    def test_build_chain_of_objects(self):
        class _Dummy(object):
            def __init__(self, config):
                for k, v in config.items():
                    setattr(self, k, v)

        configs = [{'config1': 'value1'},
                   {'config2': 'value2'}]
        build_fn = _Dummy
        dummy_chain = builder_lib.build_chain_of_objects(
            configs, build_fn=build_fn)
        self.assertEqual(dummy_chain[0].config1, 'value1')
        self.assertEqual(dummy_chain[1].config2, 'value2')

    @parameterized.parameters(
        {"inside_root": True},
        {"inside_root": True, "search_in_root": False},
        {"inside_root": True, "remove_built_config_from_main": False},
        {"inside_root": False},
        {"inside_root": False, "remove_built_config_from_main": False})
    def test_build_config_object(self, inside_root, search_in_root=True,
                                 remove_built_config_from_main=True):
        ConfigObject = namedtuple("ConfigObject", ["k1", "k2", "k3", "k4"])

        config1 = {"k1": "v1",
                   "k2": "v2",
                   "k3": "v3",
                   "other_param": "v10"}
        config2 = {"section": {"k1": "v1",
                               "k2": "v2",
                               "k3": "v3"},
                   "other_param": "v10"}
        config_fn = ConfigObject
        if inside_root:
            config_main = config1
        else:
            config_main = config2

        config_main_orig = copy.deepcopy(config_main)

        if inside_root and not search_in_root:
            with self.assertRaises(TypeError):
                builder_lib.build_config_object(
                    main_config=config_main, config_fn=config_fn,
                    search_in_root=search_in_root, k4="v4",
                    remove_built_config_from_main=remove_built_config_from_main)
            return

        config_built = builder_lib.build_config_object(
            main_config=config_main, config_fn=config_fn,
            search_in_root=search_in_root, k4="v4",
            additional_search_key="section",
            remove_built_config_from_main=remove_built_config_from_main)
        self.assertEqual(config_built.k1, "v1")
        self.assertEqual(config_built.k2, "v2")
        self.assertEqual(config_built.k3, "v3")
        self.assertEqual(config_built.k4, "v4")
        if remove_built_config_from_main:
            self.assertDictEqual(config_main, {"other_param": "v10"})
        else:
            self.assertDictEqual(config_main, config_main_orig)

    def test_eval_config(self):
        config_to_eval = {'c1': "[1, 2, 3]",
                          'c2': 3,
                          'c3': "{'a': 10, 'b': 20}"}
        config_to_eval_after_eval = {'c1': [1, 2, 3],
                                     'c2': 3,
                                     'c3': {'a': 10, 'b': 20}}
        self.assertDictEqual(builder_lib.eval_config(config_to_eval),
                             config_to_eval_after_eval)

    @parameterized.parameters(
        {"with_plugged_objects": True, "with_callbacks": False},
        {"with_plugged_objects": False, "with_callbacks": False},
        {"with_plugged_objects": True, "with_callbacks": True},
        {"with_plugged_objects": False, "with_callbacks": True})
    def test_build_with_plugged_objects(self, with_plugged_objects,
                                        with_callbacks):
        def _build_callback1(built_object, **inputs):
            return built_object + "_after_callback1"

        def _build_callback2(built_object, **inputs):
            return built_object + "_after_callback2"

        build_callback1 = MagicMock(side_effect=_build_callback1)
        build_callback2 = MagicMock(side_effect=_build_callback2)

        config = {"param1": "value1", "param2": "value2"}
        configs_plugged = {"plugged1": {"p11": "value11",
                                        "p12": "value12"},
                           "plugged2": {"p21": "value21"}}

        build_fn = MagicMock(return_value="final_object")
        plugged_build_fn1 = MagicMock(return_value="plugged_object1")
        plugged_build_fn2 = MagicMock(return_value="plugged_object2")
        plugged_build_fn3 = MagicMock(return_value="plugged_object3")

        plugged_config_keys_and_builds = [
            builder_lib.PluggedConfigKeyAndBuildFn(
                "plugged1", plugged_build_fn1, add_to_config=True),
            builder_lib.PluggedConfigKeyAndBuildFn(
                "plugged2", plugged_build_fn2, False),
            builder_lib.PluggedConfigKeyAndBuildFn(
                "plugged3", plugged_build_fn3, False),
        ]

        if with_plugged_objects:
            config.update(configs_plugged)

        if with_callbacks:
            build_callbacks = [build_callback1, build_callback2]
        else:
            build_callbacks = None

        built_object = builder_lib.build_with_plugged_objects(
            config, build_fn, plugged_config_keys_and_builds,
            build_callbacks)

        if with_callbacks:
            built_object_must = "final_object_after_callback1_after_callback2"
        else:
            built_object_must = "final_object"

        self.assertEqual(built_object_must,
                         built_object)
        if not with_plugged_objects:
            plugged_build_fn1.assert_not_called()
            plugged_build_fn2.assert_not_called()
        else:
            plugged_build_fn1.assert_called_once_with(
                {"p11": "value11", "p12": "value12"})
            plugged_build_fn2.assert_called_once_with(
                {"p21": "value21"})

        plugged_build_fn3.assert_not_called()

        if with_plugged_objects:
            build_fn.assert_called_once_with(
                {"param1": "value1", "param2": "value2",
                 "plugged1": "plugged_object1"})
        else:
            build_fn.assert_called_once_with(config)

        if with_callbacks and with_plugged_objects:
            build_callback1.assert_called_once_with(
                "final_object", plugged2="plugged_object2")
            build_callback2.assert_called_once_with(
                "final_object_after_callback1", plugged2="plugged_object2")
        elif with_callbacks:
            build_callback1.assert_called_once_with(
                "final_object")
            build_callback2.assert_called_once_with(
                "final_object_after_callback1")
