# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Base class for learning rate manipulation
"""
from abc import abstractmethod

import tensorflow as tf

from nucleus7.core.base import BaseClass
from nucleus7.utils import tf_objects_factory


class LearningRateManipulator(BaseClass):
    """
    Base class for learning rate manipulation
    """

    @abstractmethod
    def get_current_learning_rate(
            self,
            initial_learning_rate: float,
            global_step: tf.Tensor) -> tf.Tensor:
        """
        Calculate and return the current learning rate

        Parameters
        ----------
        initial_learning_rate
            The specified learning rate
        global_step
            The global step of the optimization

        Returns
        -------
        current_learning_rate
            The currently wanted learning rate
        """


class ConstantLearningRate(LearningRateManipulator):
    """
    Class providing a constant learning rate
    """

    def get_current_learning_rate(
            self,
            initial_learning_rate: float,
            global_step: tf.Tensor):
        """
        See parent class for documentation
        """
        return tf.convert_to_tensor(initial_learning_rate, dtype=tf.float32)


class TFLearningRateDecay(LearningRateManipulator):
    """
    Wraps the learning rate decays from tensorflow itself

    Parameters
    ----------
    decay_type_name
        name of the decay from tf.nn namespace excluding '_decay' suffix, e.g.
        for exponential_decay it should be exponential
    decay_params
        parameters of the decay, which will be passed to decay function
    """

    def __init__(self, decay_type_name: str, **decay_params):
        super().__init__()
        self.decay_type_name = decay_type_name
        self.decay_params = decay_params

    def get_current_learning_rate(
            self,
            initial_learning_rate: float,
            global_step: tf.Tensor) -> tf.Tensor:
        """
        See parent class for documentation
        """
        return tf_objects_factory.learning_rate_decay_factory(
            initial_learning_rate, global_step,
            name=self.decay_type_name, **self.decay_params)
