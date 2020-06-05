# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from absl.testing import parameterized

from nucleus7.optimization import configs as opt_configs


class TestOptimizationConfigs(parameterized.TestCase):

    def test_create_and_validate_optimization_config_global(self):
        optimization_parameters = {'optimizer_name': 'RMSprop',
                                   'learning_rate': 100,
                                   'optimizer_parameters': {
                                       'epsilon': 10,
                                       'decay': 5
                                   },
                                   'gradient_clip': 2,
                                   'gradient_noise_std': 0.1,
                                   'decouple_regularization': True}
        config = opt_configs.create_and_validate_optimization_config(
            **optimization_parameters, is_global=True)
        self.assertEqual(config.optimizer_name, "RMSprop")
        self.assertEqual(config.learning_rate, 100)
        self.assertEqual(config.decouple_regularization, True)
        self.assertEqual(config.gradient_clip, 2)
        self.assertEqual(config.gradient_noise_std, 0.1)
        self.assertIsNone(config.learning_rate_multiplier)
        self.assertDictEqual(config.optimizer_parameters,
                             {"epsilon": 10, 'decay': 5})
        self.assertNotIn('learning_rate', config.optimizer_parameters)

        with self.assertRaises(AssertionError):
            opt_configs.create_and_validate_optimization_config(
                is_global=True)
        with self.assertRaises(AssertionError):
            opt_configs.create_and_validate_optimization_config(
                optimizer_name='RMSprop',
                is_global=True)
        with self.assertRaises(AssertionError):
            opt_configs.create_and_validate_optimization_config(
                optimizer_name='RMSprop',
                learning_rate=10,
                learning_rate_multiplier=10,
                is_global=True)

    def test_create_optimizer_config_local(self):
        optimization_parameters = {'optimizer_name': 'RMSprop',
                                   'optimizer_parameters': {
                                       'epsilon': 10,
                                       'decay': 5
                                   },
                                   'gradient_clip': 2,
                                   'gradient_noise_std': 0.1,
                                   'decouple_regularization': True,
                                   'learning_rate_multiplier': 2}
        config = opt_configs.create_and_validate_optimization_config(
            **optimization_parameters, is_global=False)
        self.assertEqual(config.optimizer_name, "RMSprop")
        self.assertIsNone(config.learning_rate)
        self.assertEqual(config.decouple_regularization, True)
        self.assertEqual(config.gradient_clip, 2)
        self.assertEqual(config.gradient_noise_std, 0.1)
        self.assertDictEqual(config.optimizer_parameters,
                             {"epsilon": 10, 'decay': 5})
        self.assertEqual(config.learning_rate_multiplier, 2)
        self.assertNotIn('learning_rate', config.optimizer_parameters)

        with self.assertRaises(AssertionError):
            opt_configs.create_and_validate_optimization_config(
                learning_rate=10, is_global=False)
        with self.assertRaises(AssertionError):
            opt_configs.create_and_validate_optimization_config(
                learning_rate_manipulator=object(), is_global=False)

        optimization_parameters = {'epsilon': 10,
                                   'decay': 5,
                                   'gradient_clip': 2,
                                   'gradient_noise_std': 0.1,
                                   'decouple_regularization': True,
                                   'learning_rate_multiplier': None}
        config = opt_configs.create_and_validate_optimization_config(
            **optimization_parameters, is_global=False)
        self.assertIsNone(config.optimizer_name)
        self.assertEqual(config.learning_rate_multiplier, 1.0)
        self.assertNotIn('learning_rate', config.optimizer_parameters)

    @parameterized.parameters(
        {'global_decouple': True, 'local_decouple': False},
        {'global_decouple': None, 'local_decouple': False},
        {'global_decouple': True, 'local_decouple': None},
        {'global_decouple': False, 'local_decouple': True},
        {'local_learning_rate_multiplier': None},
        {'local_learning_rate_multiplier': 10}
    )
    def test_merge_optimizer_configs_different_optimizers(
            self, global_decouple=None, local_decouple=None,
            local_learning_rate_multiplier=2.0):
        global_optimization_parameters = {
            'optimizer_name': 'RMSprop',
            'learning_rate': 100,
            'optimizer_parameters': {
                'epsilon': 10,
                'decay': 5
            },
            'gradient_clip': 2,
            'gradient_noise_std': 0.1,
            'decouple_regularization': global_decouple}
        local_optimization_parameters = {
            'optimizer_name': 'Adadelta',
            'optimizer_parameters': {
                'rho': 5,
            },
            'gradient_clip': 20,
            'decouple_regularization': local_decouple,
            'learning_rate_multiplier': local_learning_rate_multiplier}
        global_config = opt_configs.create_and_validate_optimization_config(
            **global_optimization_parameters, is_global=True)
        local_config = opt_configs.create_and_validate_optimization_config(
            **local_optimization_parameters)
        local_config_merged = opt_configs.merge_optimization_configs(
            global_config, local_config)

        decouple_must = (local_decouple if local_decouple is not None
                         else global_decouple)
        self.assertEqual('Adadelta',
                         local_config_merged.optimizer_name)
        self.assertEqual(decouple_must,
                         local_config_merged.decouple_regularization)
        self.assertEqual(20,
                         local_config_merged.gradient_clip)
        self.assertEqual(0.1,
                         local_config_merged.gradient_noise_std)
        self.assertEqual({'rho': 5},
                         local_config_merged.optimizer_parameters)
        self.assertIsNone(local_config_merged.learning_rate)
        self.assertEqual(local_learning_rate_multiplier or 1.0,
                         local_config_merged.learning_rate_multiplier)

    @parameterized.parameters(
        {'global_decouple': True, 'local_decouple': False},
        {'global_decouple': True, 'local_decouple': False},
        {'global_decouple': None, 'local_decouple': False},
        {'global_decouple': True, 'local_decouple': None},
        {'global_decouple': False, 'local_decouple': True},
        {'local_learning_rate_multiplier': None},
        {'local_learning_rate_multiplier': 10},
        {'local_optimizer_name': 'RMSProp',
         'add_local_parameters': False},
        {'local_optimizer_name': 'RMSProp',
         'add_local_parameters': True},
        {'add_local_parameters': True},
        {'add_local_parameters': False},
    )
    def test_merge_optimizer_configs_same_optimizer(
            self, global_decouple=None, local_decouple=None,
            local_learning_rate_multiplier=2.0,
            add_local_parameters=True,
            local_optimizer_name=None):
        global_optimization_parameters = {
            'optimizer_name': 'RMSProp',
            'learning_rate': 100,
            'epsilon': 10,
            'decay': 5,
            'gradient_clip': 2,
            'gradient_noise_std': 0.1,
            'decouple_regularization': global_decouple}
        local_optimization_parameters = {
            'optimizer_name': local_optimizer_name,
            'decouple_regularization': local_decouple,
            'learning_rate_multiplier': local_learning_rate_multiplier}
        if add_local_parameters:
            local_optimization_parameters['decay'] = 2
        global_config = opt_configs.create_and_validate_optimization_config(
            **global_optimization_parameters, is_global=True)
        local_config = opt_configs.create_and_validate_optimization_config(
            **local_optimization_parameters)
        local_config_merged = opt_configs.merge_optimization_configs(
            global_config, local_config)
        if local_optimizer_name is not None:
            optimizer_parameters_must = (
                {'decay': 2} if add_local_parameters else None)
        else:
            optimizer_parameters_must = (
                {'decay': 2} if add_local_parameters else
                {'epsilon': 10, 'decay': 5})

        self.assertEqual(local_config_merged.optimizer_name,
                         'RMSProp')
        decouple_must = (local_decouple if local_decouple is not None
                         else global_decouple)
        self.assertEqual(local_config_merged.decouple_regularization,
                         decouple_must)
        self.assertEqual(2,
                         local_config_merged.gradient_clip)
        self.assertEqual(0.1,
                         local_config_merged.gradient_noise_std, 0.1)
        if optimizer_parameters_must is None:
            self.assertIsNone(local_config_merged.optimizer_parameters)
        else:
            self.assertDictEqual(optimizer_parameters_must,
                                 local_config_merged.optimizer_parameters)
        self.assertIsNone(local_config_merged.learning_rate)
