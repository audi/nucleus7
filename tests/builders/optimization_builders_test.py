# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from absl.testing import parameterized

from nucleus7.builders.optimization_builders import (
    build_learning_rate_manipulator)
from nucleus7.optimization.learning_rate_manipulator import ConstantLearningRate
from nucleus7.optimization.learning_rate_manipulator import (
    LearningRateManipulator)


class _DummyLearningRate(LearningRateManipulator):
    def __init__(self, dummy_var):
        super().__init__()
        self.dummy_var = dummy_var


class _WrongInheritedLearningRate(object):
    pass


class TestBuildLearningRateManipulator(parameterized.TestCase):
    @parameterized.parameters(
        {'class_name': '', 'class_must': ConstantLearningRate},
        {'class_name': None, 'class_must': ConstantLearningRate},
        {'class_name': 'optimization_builders_test._DummyLearningRate',
         'class_must': _DummyLearningRate,
         'class_kwargs': {'dummy_var': 1.0}},
        {'class_name': "not_existing_class", 'class_must': None},
        {'class_name': 'optimization_builders_test._WrongInheritedLearningRate',
         'class_must': None})
    def test_factory(self, class_name, class_must, class_kwargs=None):
        class_kwargs = class_kwargs or {}
        if class_must is None:
            error_type = (ImportError if class_name == 'not_existing_class'
                          else ValueError)
            with self.assertRaises(error_type):
                _ = build_learning_rate_manipulator(class_name, **class_kwargs)
            return
        lr_manipulator = build_learning_rate_manipulator(class_name,
                                                         **class_kwargs)
        self.assertIsInstance(lr_manipulator,
                              class_must)
        if class_kwargs:
            for each_key, each_value in class_kwargs.items():
                self.assertEqual(each_value,
                                 getattr(lr_manipulator, each_key))
