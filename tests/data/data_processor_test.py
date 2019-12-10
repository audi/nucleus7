# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from unittest.mock import MagicMock

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from nucleus7.test_utils.model_dummies import DummyRandomAugmentationTf


class TestRandomAugmentationTf(tf.test.TestCase,
                               parameterized.TestCase):

    @parameterized.parameters(
        {"provide_augment": True, "provide_random_vars": True},
        {"provide_augment": False, "provide_random_vars": True},
        {"provide_augment": False, "probability": 0.0},
        {"provide_augment": False, "probability": 1.0},
        {"provide_augment": False, "probability": 1.0,
         "provide_random_vars": True},
    )
    def test_process(self, probability=0.0, provide_augment=False,
                     provide_random_vars=False):
        tf.reset_default_graph()
        data = np.random.uniform(size=[2, 3]).astype(np.float32)
        augmentation = DummyRandomAugmentationTf(
            augmentation_probability=probability).build()
        augmentation.initialize_augment_condition = MagicMock(
            wraps=augmentation.initialize_augment_condition)
        augmentation.augment = MagicMock(wraps=augmentation.augment)
        augmentation.not_augment = MagicMock(wraps=augmentation.not_augment)
        augmentation.create_random_variables = MagicMock(
            wraps=augmentation.create_random_variables)
        augment = tf.constant(True) if provide_augment else None
        noise_np = np.random.uniform(size=[]).astype(np.float32)
        noise = tf.constant(noise_np) if provide_random_vars else None

        result = augmentation.process(augment=augment,
                                      data=data,
                                      noise=noise)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            result_eval, random_vars_eval = sess.run(
                [result, augmentation.random_variables])

        self.assertSetEqual(set(result_eval),
                            set(augmentation.generated_keys_all))
        self.assertSetEqual(set(random_vars_eval),
                            set(augmentation.random_variables_keys))

        if provide_augment:
            augmentation.initialize_augment_condition.assert_not_called()
        else:
            augmentation.initialize_augment_condition.assert_called_once_with()

        random_vars_must = ({"noise": noise_np} if provide_random_vars
                            else random_vars_eval)
        if provide_random_vars:
            augmentation.create_random_variables.assert_not_called()
            self.assertAllClose(random_vars_must,
                                random_vars_eval)
        else:
            augmentation.create_random_variables.assert_called_once_with()

        augmentation.augment.assert_called_once_with(data=data)
        augmentation.not_augment.assert_called_once_with(data=data)

        augment_cond_must = True if provide_augment else probability == 1.0
        data_must = (data + random_vars_must["noise"] if augment_cond_must
                     else data)
        result_must = {"augment": augment_cond_must,
                       "data": data_must,
                       "noise": random_vars_must["noise"]}
        self.assertAllClose(result_must,
                            result_eval)
