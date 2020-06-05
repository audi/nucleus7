# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from unittest.mock import MagicMock
from unittest.mock import call as mock_call

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from nucleus7.optimization.configs import (
    OptimizationConfig)
from nucleus7.optimization.learning_rate_manipulator import ConstantLearningRate
from nucleus7.optimization.optimization_handler import OptimizationHandler
from nucleus7.test_utils import test_opt_utils


class TestOptimization(tf.test.TestCase,
                       parameterized.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        np.random.seed(4564)
        self.optimization = OptimizationHandler()
        self.learning_rate_manipulator = ConstantLearningRate().build()
        self.global_optim_config = OptimizationConfig(
            optimizer=None,
            optimizer_name='rmsprop',
            learning_rate=0.01,
            learning_rate_multiplier=None,
            learning_rate_manipulator=self.learning_rate_manipulator,
            gradient_clip=0.1,
            gradient_noise_std=0.05,
            optimizer_parameters={'decay': 0.99},
            decouple_regularization=True
        )
        self.local_optim_config1 = OptimizationConfig(
            optimizer=None,
            optimizer_name='sgd',
            learning_rate=None,
            learning_rate_multiplier=0.25,
            learning_rate_manipulator=None,
            gradient_clip=0.05,
            gradient_noise_std=0.2,
            optimizer_parameters=None,
            decouple_regularization=True,
        )
        self.local_optim_config2 = OptimizationConfig(
            optimizer=None,
            optimizer_name=None,
            learning_rate=None,
            learning_rate_multiplier=0.01,
            learning_rate_manipulator=None,
            gradient_clip=0,
            gradient_noise_std=0,
            optimizer_parameters=None,
            decouple_regularization=False
        )

        self.vars_sizes = [[1], [2, 2], [2, 3, 3], [1, 1], [2, 2]]
        self.grads_and_vars_np = test_opt_utils.create_grads_and_vars_np(
            self.vars_sizes)
        self.reg_grads_np, _ = (
            zip(*test_opt_utils.create_grads_and_vars_np(self.vars_sizes)))

    def test_constructor(self):
        # no new tensorflow variables must be declared
        self.assertEmpty(tf.global_variables())

    def test_global_config_setter(self):
        wrong_config1 = self.global_optim_config._replace(
            optimizer=tf.train.GradientDescentOptimizer(0.1))
        wrong_config2 = self.global_optim_config._replace(
            optimizer_name=None)
        wrong_config3 = self.global_optim_config._replace(
            learning_rate=None)
        wrong_config4 = self.global_optim_config._replace(
            learning_rate_multiplier=0.1)
        with self.assertRaises(ValueError):
            self.optimization.global_config = wrong_config1
        with self.assertRaises(ValueError):
            self.optimization.global_config = wrong_config2
        with self.assertRaises(ValueError):
            self.optimization.global_config = wrong_config3
        with self.assertRaises(ValueError):
            self.optimization.global_config = wrong_config4

        self.optimization.global_config = self.global_optim_config

    def test_add_config_with_variables(self):
        (vars_tf, grads_and_vars_tf,
         reg_grads_and_vars_tf) = self._get_grads_and_vars_tf()
        vars_for_config1 = vars_tf[2:4]
        vars_for_config2 = vars_tf[4:]
        vars_for_config2_with_inters = vars_tf[3:]

        wrong_config1 = self.local_optim_config1._replace(
            optimizer=tf.keras.optimizers.SGD(0.1))
        wrong_config2 = self.local_optim_config1._replace(
            learning_rate=0.5)
        wrong_config3 = self.global_optim_config._replace(
            learning_rate_manipulator=ConstantLearningRate())
        with self.assertRaises(ValueError):
            self.optimization.add_config_with_variables(
                (wrong_config1, vars_for_config1))
        with self.assertRaises(ValueError):
            self.optimization.add_config_with_variables(
                (wrong_config2, vars_for_config1))
        with self.assertRaises(ValueError):
            self.optimization.add_config_with_variables(
                (wrong_config3, vars_for_config1))

        with self.assertRaises(ValueError):
            self.optimization.add_config_with_variables(
                (self.local_optim_config1, vars_for_config1))

        self.optimization.global_config = self.global_optim_config
        self.optimization.add_config_with_variables(
            (self.local_optim_config1, vars_for_config1))
        with self.assertRaises(ValueError):
            self.optimization.add_config_with_variables(
                (self.local_optim_config2, vars_for_config2_with_inters))

        self.optimization.add_config_with_variables(
            (self.local_optim_config2, vars_for_config2))

        self.assertEqual(
            self.local_optim_config1,
            self.optimization._local_configs_with_vars[0][0])
        config2_must = self.local_optim_config2._replace(
            optimizer_name=self.global_optim_config.optimizer_name,
            optimizer_parameters=self.global_optim_config.optimizer_parameters)
        self.assertEqual(
            config2_must,
            self.optimization._local_configs_with_vars[1][0])

        self.assertListEqual(
            list(vars_for_config1),
            list(self.optimization._local_configs_with_vars[0][1]))
        self.assertListEqual(
            list(vars_for_config2),
            list(self.optimization._local_configs_with_vars[1][1]))

    def test_initialize_for_session(self):
        def _get_current_learning_rate(initial_learning_rate, global_step):
            return tf.convert_to_tensor(initial_learning_rate)

        lr_manipulator = self.global_optim_config.learning_rate_manipulator
        lr_manipulator.get_current_learning_rate = MagicMock(
            side_effect=_get_current_learning_rate)

        with self.assertRaises(ValueError):
            self.optimization.initialize_for_session()
        self.optimization.global_config = self.global_optim_config
        (vars_tf, grads_and_vars_tf,
         reg_grads_and_vars_tf) = self._get_grads_and_vars_tf()
        vars_for_config1 = vars_tf[2:4]
        vars_for_config2 = vars_tf[4:]
        self.optimization.add_config_with_variables(
            (self.local_optim_config1, vars_for_config1))
        self.optimization.add_config_with_variables(
            (self.local_optim_config2, vars_for_config2))

        self.optimization.initialize_for_session()
        self.assertNotEmpty(tf.get_collection(tf.GraphKeys.GLOBAL_STEP))
        self.assertIs(tf.get_collection(tf.GraphKeys.GLOBAL_STEP)[0],
                      self.optimization.global_step)
        self.assertIsInstance(self.optimization.global_learning_rate,
                              tf.Tensor)

        global_optimizer = self.optimization.global_config.optimizer
        local_optimizer1 = (
            self.optimization._local_configs_with_vars[0][0].optimizer)
        local_optimizer2 = (
            self.optimization._local_configs_with_vars[1][0].optimizer)
        self.assertIsInstance(global_optimizer,
                              tf.keras.optimizers.RMSprop)
        self.assertIsInstance(local_optimizer1,
                              tf.keras.optimizers.SGD)
        self.assertIsInstance(local_optimizer2,
                              tf.keras.optimizers.RMSprop)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            global_step_eval = sess.run(self.optimization.global_step)
            global_lr_eval = sess.run(self.optimization.global_learning_rate)
            lr_global_optimizer_eval = sess.run(global_optimizer.learning_rate)
            lr_local_optimizer1_eval = sess.run(local_optimizer1.learning_rate)
            lr_local_optimizer2_eval = sess.run(local_optimizer2.learning_rate)

        self.assertEqual(0,
                         global_step_eval)
        self.assertAllClose(self.global_optim_config.learning_rate,
                            global_lr_eval)
        self.assertAllClose(self.global_optim_config.learning_rate,
                            lr_global_optimizer_eval)
        self.assertAllClose(
            self.global_optim_config.learning_rate
            * self.local_optim_config1.learning_rate_multiplier,
            lr_local_optimizer1_eval)
        self.assertAllClose(
            self.global_optim_config.learning_rate
            * self.local_optim_config2.learning_rate_multiplier,
            lr_local_optimizer2_eval)
        lr_manipulator.get_current_learning_rate.assert_called_once_with(
            self.global_optim_config.learning_rate,
            self.optimization.global_step)

    @parameterized.parameters(
        {'decouple_regularization': True, 'use_regularization': True},
        {'decouple_regularization': True, 'use_regularization': False},
        {'decouple_regularization': False, 'use_regularization': True},
        {'decouple_regularization': False, 'use_regularization': False})
    def test_filter_grads_and_vars_with_decouple_for_config(
            self, decouple_regularization=True,
            use_regularization=True):
        (vars_tf, grads_and_vars_tf,
         reg_grads_and_vars_tf) = self._get_grads_and_vars_tf()
        vars_for_config = vars_tf[2:4]
        config = self.local_optim_config1
        config = config._replace(
            decouple_regularization=decouple_regularization,
            learning_rate=0.5,
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.5))
        configs_with_filtered_grads_and_vars = (
            self.optimization.filter_grads_and_vars_with_decouple_for_config(
                config, vars_for_config, grads_and_vars_tf,
                reg_grads_and_vars_tf if use_regularization else None
            ))

        grads_and_vars_filtered_np = self.grads_and_vars_np[2:4]
        reg_grads_filtered_np = self.reg_grads_np[2:4]
        grads_filtered_np, _ = zip(*grads_and_vars_filtered_np)

        if decouple_regularization and use_regularization:
            self.assertEqual(2,
                             len(configs_with_filtered_grads_and_vars))
            config_res_norm, grads_and_vars = (
                configs_with_filtered_grads_and_vars[0])
            config_res_reg, grads_and_vars_reg = (
                configs_with_filtered_grads_and_vars[1])

            grads_filtered, vars_filtered = zip(*grads_and_vars)
            grads_filtered_eval = self.evaluate(grads_filtered)
            grads_filtered_reg, vars_filtered_reg = zip(*grads_and_vars_reg)
            grads_filtered_reg_eval = self.evaluate(grads_filtered_reg)
            vars_to_grads_eval = dict(zip(vars_filtered, grads_filtered_eval))
            vars_to_grads_reg_eval = dict(zip(vars_filtered_reg,
                                              grads_filtered_reg_eval))

            vars_to_grads_must = dict(
                zip(vars_for_config, grads_filtered_np))
            vars_to_grads_must_reg = dict(
                zip(vars_for_config, reg_grads_filtered_np))

            self.assertAllClose(vars_to_grads_must,
                                vars_to_grads_eval)
            self.assertAllClose(vars_to_grads_must_reg,
                                vars_to_grads_reg_eval)
            self.assertEqual(config,
                             config_res_norm)
            self.assertIsInstance(config_res_reg.optimizer,
                                  tf.train.GradientDescentOptimizer)
            self.assertEqual(config_res_norm.learning_rate,
                             config_res_reg.learning_rate)
            self.assertEqual(config_res_norm.optimizer._learning_rate,
                             config_res_reg.optimizer._learning_rate)
        else:
            self.assertEqual(1,
                             len(configs_with_filtered_grads_and_vars))
            config_res, grads_and_vars = (
                configs_with_filtered_grads_and_vars[0])
            grads_filtered, vars_filtered = zip(*grads_and_vars)
            grads_filtered_eval = self.evaluate(grads_filtered)
            vars_to_grads_eval = dict(zip(vars_filtered, grads_filtered_eval))
            if use_regularization:
                grads_filtered_np = [
                    grad + reg_grad for grad, reg_grad
                    in zip(grads_filtered_np, reg_grads_filtered_np)]
            vars_to_grads_must = dict(
                zip(vars_for_config, grads_filtered_np))
            self.assertAllClose(vars_to_grads_must,
                                vars_to_grads_eval)
            self.assertEqual(config,
                             config_res)

    @parameterized.parameters(
        {'use_regularization': True, 'use_only_global_config': True},
        {'use_regularization': True, 'use_only_global_config': False},
        {'use_regularization': False, 'use_only_global_config': True},
        {'use_regularization': False, 'use_only_global_config': False})
    def test_create_configs_with_grads_and_vars(self, use_only_global_config,
                                                use_regularization):
        res_must = []

        def _filter_grads_and_vars_with_decouple_for_config(
                optim_config, vars_for_config, grads_and_vars,
                regularization_grads_and_vars):
            config_with_vars = (
                OptimizationHandler.filter_grads_and_vars_with_decouple_for_config(
                    optim_config, vars_for_config, grads_and_vars,
                    regularization_grads_and_vars))
            res_must.extend(config_with_vars)
            return config_with_vars

        self.optimization.filter_grads_and_vars_with_decouple_for_config = (
            MagicMock(
                side_effect=_filter_grads_and_vars_with_decouple_for_config))

        (vars_tf, grads_and_vars_tf,
         reg_grads_and_vars_tf) = self._get_grads_and_vars_tf()
        vars_for_config1 = vars_tf[2:4]
        vars_for_config2 = vars_tf[4:]
        self.optimization.global_config = self.global_optim_config
        if not use_only_global_config:
            self.optimization.add_config_with_variables(
                (self.local_optim_config1, vars_for_config1))
            self.optimization.add_config_with_variables(
                (self.local_optim_config2, vars_for_config2))
        self.optimization.initialize_for_session()
        regularization_grads_and_vars = (
                use_regularization and reg_grads_and_vars_tf or None)
        optim_configs_with_variables = (
            self.optimization.create_configs_with_grads_and_vars(
                grads_and_vars=grads_and_vars_tf,
                regularization_grads_and_vars=regularization_grads_and_vars,
                all_trainable_variables=vars_tf
            ))
        if use_only_global_config:
            vars_for_global_config = set(vars_tf)
        else:
            vars_for_global_config = set(vars_tf[:2])
        filter_grads_and_vars_with_decouple_for_config_call_args_list = [
            mock_call(self.optimization.global_config, vars_for_global_config,
                      grads_and_vars_tf, regularization_grads_and_vars)]
        if not use_only_global_config:
            filter_grads_and_vars_with_decouple_for_config_call_args_list += [
                mock_call(self.optimization._local_configs_with_vars[0][0],
                          vars_for_config1,
                          grads_and_vars_tf, regularization_grads_and_vars),
                mock_call(self.optimization._local_configs_with_vars[1][0],
                          vars_for_config2,
                          grads_and_vars_tf, regularization_grads_and_vars)
            ]
        opt = self.optimization
        opt.filter_grads_and_vars_with_decouple_for_config.assert_has_calls(
            filter_grads_and_vars_with_decouple_for_config_call_args_list)

        if use_only_global_config:
            optim_configs_with_vars_len_must = 1
        else:
            optim_configs_with_vars_len_must = 3
        if use_regularization:
            optim_configs_with_vars_len_must += (
                    self.global_optim_config.decouple_regularization is True)
            if not use_only_global_config:
                optim_configs_with_vars_len_must += (
                        self.local_optim_config1.decouple_regularization
                        is True)
                optim_configs_with_vars_len_must += (
                        self.local_optim_config2.decouple_regularization
                        is True)
        self.assertEqual(optim_configs_with_vars_len_must,
                         len(optim_configs_with_variables))
        self.assertEqual(res_must,
                         optim_configs_with_variables)

    @parameterized.parameters(
        {'use_regularization': True, 'num_iterations': 1,
         'use_only_global_config': True},
        {'use_regularization': False, 'num_iterations': 5,
         'use_only_global_config': True},
        {'use_regularization': True, 'num_iterations': 1,
         'use_only_global_config': True},
        {'use_regularization': False, 'num_iterations': 5,
         'use_only_global_config': True},
        {'use_regularization': True, 'num_iterations': 1,
         'use_only_global_config': False},
        {'use_regularization': False, 'num_iterations': 5,
         'use_only_global_config': False},
        {'use_regularization': True, 'num_iterations': 1,
         'use_only_global_config': False},
        {'use_regularization': False, 'num_iterations': 5,
         'use_only_global_config': False}
    )
    def test_get_train_op(self, use_regularization, num_iterations,
                          use_only_global_config):
        global_optim_config = self.global_optim_config._replace(
            optimizer_name='SGD',
            optimizer_parameters=None,
            gradient_clip=None,
            gradient_noise_std=None)
        local_optim_config1 = self.local_optim_config1._replace(
            optimizer_name=None, optimizer_parameters=None,
            gradient_clip=None,
            gradient_noise_std=None)
        local_optim_config2 = self.local_optim_config2._replace(
            optimizer_name=None,
            optimizer_parameters=None,
            gradient_clip=None,
            gradient_noise_std=None)
        (vars_tf, grads_and_vars_tf,
         reg_grads_and_vars_tf) = self._get_grads_and_vars_tf()
        vars_for_config1 = vars_tf[2:4]
        vars_for_config2 = vars_tf[4:]
        grads_np, vars_np = zip(*self.grads_and_vars_np)

        if use_regularization:
            grads_np = [
                each_grad + each_grad_reg
                for each_grad, each_grad_reg in zip(
                    grads_np, self.reg_grads_np)]

        if use_only_global_config:
            updates_all = [
                each_grad * self.global_optim_config.learning_rate
                for each_grad in grads_np]
        else:
            update_global_config = [
                each_grad * self.global_optim_config.learning_rate
                for each_grad in grads_np[:2]]
            update_local_config1 = [
                each_grad
                * global_optim_config.learning_rate
                * local_optim_config1.learning_rate_multiplier
                for each_grad in grads_np[2:4]]
            update_local_config2 = [
                each_grad
                * global_optim_config.learning_rate
                * local_optim_config2.learning_rate_multiplier
                for each_grad in grads_np[4:]]
            updates_all = (update_global_config
                           + update_local_config1
                           + update_local_config2)

        vars_after_updates_np = [
            each_var - each_update * num_iterations
            for each_var, each_update in zip(vars_np, updates_all)]

        vars_name_to_value_after_updates_np = dict(
            zip(vars_tf, vars_after_updates_np))

        self.optimization.global_config = global_optim_config
        if not use_only_global_config:
            self.optimization.add_config_with_variables(
                (local_optim_config1, vars_for_config1))
            self.optimization.add_config_with_variables(
                (local_optim_config2, vars_for_config2))
        self.optimization.initialize_for_session()
        regularization_grads_and_vars = (
                use_regularization and reg_grads_and_vars_tf or None)

        train_op = self.optimization.get_train_op(
            grads_and_vars_tf, regularization_grads_and_vars,
            trainable_variables=vars_tf)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(num_iterations):
                sess.run(train_op)
            vars_eval = sess.run(vars_tf)
            global_step_eval = sess.run(self.optimization.global_step)
        vars_names_to_values_eval = dict(zip(vars_tf, vars_eval))
        self.assertEqual(num_iterations,
                         global_step_eval)
        if not use_only_global_config and use_regularization:
            self.assertEqual(5,
                             len(train_op.control_inputs))
        elif use_only_global_config and use_regularization:
            self.assertEqual(2,
                             len(train_op.control_inputs))
        elif use_only_global_config and not use_regularization:
            self.assertEqual(1,
                             len(train_op.control_inputs))
        else:
            self.assertEqual(3,
                             len(train_op.control_inputs))
        self.assertAllClose(vars_name_to_value_after_updates_np,
                            vars_names_to_values_eval, rtol=1e-4, atol=1e-4)

    def _get_grads_and_vars_tf(self):
        grads_and_vars_tf = test_opt_utils.convert_grads_and_vars_to_tf(
            self.grads_and_vars_np)
        grads_tf, vars_tf = zip(*grads_and_vars_tf)
        reg_grads_tf = test_opt_utils.convert_grads_to_tf(self.reg_grads_np)
        reg_grads_and_vars_tf = list(zip(reg_grads_tf, vars_tf))
        return vars_tf, grads_and_vars_tf, reg_grads_and_vars_tf
