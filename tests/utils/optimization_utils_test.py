# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from unittest.mock import patch

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from nucleus7.test_utils import test_opt_utils
from nucleus7.utils import optimization_utils


class TestOptimizationUtils(tf.test.TestCase,
                            parameterized.TestCase):
    def setUp(self):
        np.random.seed(4567)
        tf.reset_default_graph()
        self.vars_sizes = [[1], [2, 2], [3, 2, 2]]

    @parameterized.parameters({'num_towers': 1, 'use_mean': True},
                              {'num_towers': 1, 'use_mean': False},
                              {'num_towers': 5, 'use_mean': True},
                              {'num_towers': 5, 'use_mean': False})
    def test_sum_grads_with_vars(self, num_towers, use_mean):
        grads_and_vars_to_sum_np = (
            test_opt_utils.create_tower_grads_and_vars_np(
                num_towers, self.vars_sizes))
        grads_and_vars_to_sum_tf = (
            test_opt_utils.convert_tower_grads_and_vars_to_tf(
                grads_and_vars_to_sum_np))
        res = optimization_utils.sum_grads_with_vars(grads_and_vars_to_sum_tf,
                                                     use_mean=use_mean)

        res_grads = list(zip(*res))[0]
        res_vars = list(zip(*res))[1]
        vars_must = list(zip(*grads_and_vars_to_sum_tf[0]))[1]
        self.assertSetEqual(set(vars_must),
                            set(res_vars))
        grads_np = [list(zip(*each_grads_and_vars))[0]
                    for each_grads_and_vars in grads_and_vars_to_sum_np]
        if use_mean:
            grads_average_must = np.mean(grads_np, 0)
        else:
            grads_average_must = np.sum(grads_np, 0)
        grads_average_must = list(grads_average_must)

        res_grads_eval = self.evaluate(res_grads)
        vars_to_grad_must = dict(zip(vars_must, grads_average_must))
        vars_to_grad_res = dict(zip(res_vars, res_grads_eval))
        for each_var in vars_must:
            self.assertAllClose(vars_to_grad_must[each_var],
                                vars_to_grad_res[each_var])

    @parameterized.parameters({'num_towers': 2, 'use_mean': True},
                              {'num_towers': 2, 'use_mean': False},
                              {'num_towers': 5, 'use_mean': True},
                              {'num_towers': 5, 'use_mean': False})
    def test_sum_grads_with_vars_diff_length(self, num_towers, use_mean):
        grads_and_vars_to_sum_np = (
            test_opt_utils.create_tower_grads_and_vars_np(
                num_towers, self.vars_sizes))
        grads_and_vars_to_sum_tf = (
            test_opt_utils.convert_tower_grads_and_vars_to_tf(
                grads_and_vars_to_sum_np))
        vars_must = list(zip(*grads_and_vars_to_sum_tf[0]))[1]
        grads_and_vars_to_sum_np[0][0] = (
            np.zeros_like(grads_and_vars_to_sum_np[0][0][0]),
            grads_and_vars_to_sum_np[0][0][1])
        grads_and_vars_to_sum_np[0][1] = (
            np.zeros_like(grads_and_vars_to_sum_np[0][1][0]),
            grads_and_vars_to_sum_np[0][1][1])
        del grads_and_vars_to_sum_tf[0][1]
        del grads_and_vars_to_sum_tf[0][0]
        num_grads_to_sum = [num_towers] * len(self.vars_sizes)
        num_grads_to_sum[0] -= 1
        num_grads_to_sum[1] -= 1
        if num_towers > 2:
            grads_and_vars_to_sum_np[-2][0] = (
                np.zeros_like(grads_and_vars_to_sum_np[-2][0][0]),
                grads_and_vars_to_sum_np[-2][0][1])
            grads_and_vars_to_sum_np[-1][-1] = (
                np.zeros_like(grads_and_vars_to_sum_np[-1][-1][0]),
                grads_and_vars_to_sum_np[-1][-1][1])
            del grads_and_vars_to_sum_tf[-2][0]
            del grads_and_vars_to_sum_tf[-1][-1]
            num_grads_to_sum[0] -= 1
            num_grads_to_sum[-1] -= 1

        res = optimization_utils.sum_grads_with_vars(grads_and_vars_to_sum_tf,
                                                     use_mean=use_mean)

        res_grads = list(zip(*res))[0]
        res_vars = list(zip(*res))[1]
        self.assertSetEqual(set(vars_must),
                            set(res_vars))
        grads_np = [list(zip(*each_grads_and_vars))[0]
                    for each_grads_and_vars in grads_and_vars_to_sum_np]
        grads_average_must = np.sum(grads_np, 0)
        grads_average_must = list(grads_average_must)
        if use_mean:
            grads_average_must = [
                each_grad / each_num_grad
                for each_grad, each_num_grad in zip(grads_average_must,
                                                    num_grads_to_sum)
            ]

        res_grads_eval = self.evaluate(res_grads)
        vars_to_grad_must = dict(zip(vars_must, grads_average_must))
        vars_to_grad_res = dict(zip(res_vars, res_grads_eval))
        for each_var in vars_must:
            self.assertAllClose(vars_to_grad_must[each_var],
                                vars_to_grad_res[each_var])

    @parameterized.parameters({'num_towers': 1},
                              {'num_towers': 5})
    def test_average_grads_and_vars_from_multiple_devices(
            self, num_towers):
        consolidation_device = '/cpu:0'
        grads_and_vars_to_sum_np = (
            test_opt_utils.create_tower_grads_and_vars_np(
                num_towers, self.vars_sizes))
        grads_and_vars_to_sum_tf = (
            test_opt_utils.convert_tower_grads_and_vars_to_tf(
                grads_and_vars_to_sum_np))
        res = optimization_utils.average_grads_and_vars_from_multiple_devices(
            grads_and_vars_to_sum_tf, consolidation_device=consolidation_device)

        res_grads = list(zip(*res))[0]
        vars_res = list(zip(*res))[1]
        vars_must = list(zip(*grads_and_vars_to_sum_tf[0]))[1]
        self.assertSetEqual(set(vars_must),
                            set(vars_res))
        grads_np = [list(zip(*each_grads_and_vars))[0]
                    for each_grads_and_vars in grads_and_vars_to_sum_np]
        grads_average_must = np.mean(grads_np, 0)

        grads_average_must = list(grads_average_must)
        res_grads_eval = self.evaluate(res_grads)
        vars_to_grad_must = dict(zip(vars_must, grads_average_must))
        vars_to_grad_res = dict(zip(vars_res, res_grads_eval))
        for each_var in vars_must:
            self.assertAllClose(vars_to_grad_must[each_var],
                                vars_to_grad_res[each_var])

    def test_average_grads_and_vars_from_multiple_devices_wrong_len(self):
        consolidation_device = '/cpu:0'
        grads_and_vars_to_sum_np = (
            test_opt_utils.create_tower_grads_and_vars_np(
                5, self.vars_sizes))
        del grads_and_vars_to_sum_np[1][0]
        grads_and_vars_to_sum_tf = (
            test_opt_utils.convert_tower_grads_and_vars_to_tf(
                grads_and_vars_to_sum_np))
        with self.assertRaises(AssertionError):
            _ = optimization_utils.average_grads_and_vars_from_multiple_devices(
                grads_and_vars_to_sum_tf,
                consolidation_device=consolidation_device)

    @parameterized.parameters({'gradient_clip': 0.01},
                              {'gradient_clip': 100},
                              {'gradient_clip': 0.01, 'grad_l2_norm': 0.1},
                              {'gradient_clip': 100.0, 'grad_l2_norm': 10.0})
    def test_clip_grads_and_vars(self, gradient_clip, grad_l2_norm=None):
        grads_and_vars_np = test_opt_utils.create_grads_and_vars_np(
            self.vars_sizes)
        grads_and_vars_tf = test_opt_utils.convert_grads_and_vars_to_tf(
            grads_and_vars_np)
        grads_and_vars_tf_clipped = optimization_utils.clip_grads_and_vars(
            grads_and_vars_tf, gradient_clip, gradient_l2_norm=grad_l2_norm)
        grads_tf_clipped = list(zip(*grads_and_vars_tf_clipped))[0]
        grads_tf_clipped_eval = self.evaluate(grads_tf_clipped)
        grads_all = np.concatenate(
            [grad.ravel() for grad, _ in grads_and_vars_np], 0)
        if grad_l2_norm is None:
            grad_l2_norm = (grads_all ** 2).sum() ** 0.5
        grads_clipped_must = [
            each_grad * gradient_clip / np.maximum(grad_l2_norm, gradient_clip)
            for each_grad, _ in grads_and_vars_np]
        vars_must = list(zip(*grads_and_vars_tf))[1]
        vars_res = list(zip(*grads_and_vars_tf_clipped))[1]
        vars_to_grad_must = dict(zip(vars_must, grads_clipped_must))
        vars_to_grad_res = dict(zip(vars_res, grads_tf_clipped_eval))
        for each_var in vars_must:
            self.assertAllClose(vars_to_grad_must[each_var],
                                vars_to_grad_res[each_var])

    def test_filter_grads_for_vars(self):
        grads_and_vars_np = test_opt_utils.create_grads_and_vars_np(
            self.vars_sizes)
        grads_and_vars_tf = test_opt_utils.convert_grads_and_vars_to_tf(
            grads_and_vars_np)
        grads, variables = zip(*grads_and_vars_tf)
        vars_to_filter = [variables[2], variables[0]]
        grads_and_vars_filtered = optimization_utils.filter_grads_for_vars(
            grads_and_vars_tf, vars_to_filter)
        grads_must = [grads[2], grads[0]]
        grads_and_vars_filtered_must = list(zip(grads_must, vars_to_filter))
        self.assertSetEqual(set(grads_and_vars_filtered_must),
                            set(grads_and_vars_filtered))

    @patch('tensorflow.random_normal')
    def test_add_noise_to_grads_and_vars(self, random_normal):
        def _random_normal(shape, mean=None, stddev=None,
                           dtype=None, seed=None, name=None):
            return tf.ones(shape) * stddev

        random_normal.side_effect = _random_normal

        noise_std = 0.5
        grads_and_vars_np = test_opt_utils.create_grads_and_vars_np(
            self.vars_sizes)
        grads_and_vars_tf = test_opt_utils.convert_grads_and_vars_to_tf(
            grads_and_vars_np)
        grads_and_vars_tf_with_noise = (
            optimization_utils.add_noise_to_grads_and_vars(
                grads_and_vars_tf, gradient_noise_std=noise_std))

        grads_with_noise_must, _ = zip(*grads_and_vars_np)
        _, vars_must = zip(*grads_and_vars_tf)
        grads_with_noise_must = [each_grad + noise_std
                                 for each_grad in grads_with_noise_must]
        grads_with_noise, vars_with_noise = zip(*grads_and_vars_tf_with_noise)
        grads_with_noise_eval = self.evaluate(grads_with_noise)

        vars_to_grad_must = dict(zip(vars_must, grads_with_noise_must))
        vars_to_grad_res = dict(zip(vars_with_noise, grads_with_noise_eval))
        for each_var in vars_must:
            self.assertAllClose(vars_to_grad_must[each_var],
                                vars_to_grad_res[each_var])

    @parameterized.parameters(
        {},
        {"gradient_clip": 0.1, 'gradient_l2_norm': 0.5},
        {"gradient_noise_std": 0.25},
        {"gradient_clip": 0.1, 'gradient_l2_norm': 0.5,
         "gradient_noise_std": 0.25},
        {"gradient_clip": 0.1, 'gradient_l2_norm': 0.5,
         "gradient_noise_std": 0.25, "with_global_step": True})
    @patch('nucleus7.utils.optimization_utils.clip_grads_and_vars')
    @patch('nucleus7.utils.optimization_utils.add_noise_to_grads_and_vars')
    def test_train_op_with_clip_and_noise(
            self, add_noise_to_grads_and_vars, clip_grads_and_vars,
            gradient_clip=None, gradient_l2_norm=None,
            gradient_noise_std=None, with_global_step=False):

        def _clip_grads_and_vars(grads_and_vars, gradient_clip,
                                 gradient_l2_norm):
            grads, variables = zip(*grads_and_vars)
            grads = [each_grad + gradient_clip + gradient_l2_norm
                     for each_grad in grads]
            return list(zip(grads, variables))

        def _add_noise_to_grads_and_vars(grads_and_vars, gradient_noise_std):
            grads, variables = zip(*grads_and_vars)
            grads = [each_grad + gradient_noise_std
                     for each_grad in grads]
            return list(zip(grads, variables))

        global_step = tf.train.get_or_create_global_step()
        clip_grads_and_vars.side_effect = _clip_grads_and_vars
        add_noise_to_grads_and_vars.side_effect = _add_noise_to_grads_and_vars

        grads_and_vars_np = test_opt_utils.create_grads_and_vars_np(
            self.vars_sizes)
        grads_and_vars_tf = test_opt_utils.convert_grads_and_vars_to_tf(
            grads_and_vars_np)
        _, vars_tf = zip(*grads_and_vars_tf)
        learning_rate = 0.1
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate)
        train_op = optimization_utils.train_op_with_clip_and_noise(
            optimizer, grads_and_vars_tf,
            gradient_clip=gradient_clip,
            gradient_noise_std=gradient_noise_std,
            global_step=(global_step if with_global_step else None),
            gradient_l2_norm=(tf.convert_to_tensor(gradient_l2_norm)
                              if gradient_l2_norm else None)
        )
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(train_op)
            vars_res = sess.run(vars_tf)
            global_step_res = sess.run(global_step)

        grads_np_must, vars_np_must = zip(*grads_and_vars_np)
        if gradient_clip is not None:
            grads_np_must = [each_grad + gradient_clip + gradient_l2_norm
                             for each_grad in grads_np_must]
        if gradient_noise_std is not None:
            grads_np_must = [each_grad + gradient_noise_std
                             for each_grad in grads_np_must]

        update_must = [each_grad * learning_rate for each_grad in grads_np_must]
        variables_must = [
            each_var - each_update
            for each_var, each_update in zip(vars_np_must, update_must)]

        self.assertAllClose(variables_must,
                            vars_res)

        if gradient_clip is not None:
            self.assertEqual(1,
                             clip_grads_and_vars.call_count)
        if gradient_noise_std is not None:
            self.assertEqual(1,
                             add_noise_to_grads_and_vars.call_count)

        global_step_must = 1 if with_global_step else 0
        self.assertEqual(global_step_must,
                         global_step_res)

    def test_gradient_l2_norm(self):
        grads_and_vars_np = test_opt_utils.create_grads_and_vars_np(
            self.vars_sizes)
        grads_and_vars_tf = test_opt_utils.convert_grads_and_vars_to_tf(
            grads_and_vars_np)
        grads_tf, _ = zip(*grads_and_vars_tf)
        gradient_l2_norm = optimization_utils.get_gradient_l2_norm(grads_tf)
        gradient_l2_norm_eval = self.evaluate(gradient_l2_norm)
        grads_all_np = np.concatenate(
            [grad.ravel() for grad, _ in grads_and_vars_np], 0)
        grad_l2_norm_must = (grads_all_np ** 2).sum() ** 0.5
        self.assertAllClose(grad_l2_norm_must,
                            gradient_l2_norm_eval)
