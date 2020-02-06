# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Base interfaces for Data Processor
"""
import abc
import logging
from typing import Dict
from typing import List
from typing import Optional

import tensorflow as tf

from nucleus7.data.base import BaseDataPipeNucleotide
from nucleus7.data.data_filter import DataFilterMixin
from nucleus7.utils import object_utils
from nucleus7.utils import utils


class DataProcessor(BaseDataPipeNucleotide,
                    DataFilterMixin):
    """
    Data Processor interface, which can be used to process the data from
    readers or other processors
    """
    register_name_scope = "data_processor"
    exclude_from_register = True

    @abc.abstractmethod
    def process(self, **inputs):
        """
        Main method to process the inputs sample wise

        Parameters
        ----------
        inputs
            inputs to data processor

        Returns
        -------
        processed_data
            processed data
        """

    @object_utils.raise_exception_with_class_name
    def __call__(self, **inputs):
        # pylint: disable=arguments-differ
        # parent __call__ method has more generic signature
        result = self.process(**inputs)
        return result


class RandomAugmentationTf(DataProcessor):
    """
    Random augmentation processor.

    Augments the input with probability given by `augmentation_probability`.
    If the `augment` key is provided, then `augmentation_probability` is ignored
    and the tensorflow boolean values from augment specify whether
    the input is augmented.

    Can also be used as a random generator. In that case you can also override
    `not_augment(...)`. If doing so, take care to ensure that
    `not_augment(...)` and `augment(...)` result in dicts with the
    same structure.

    Parameters
    ----------
    augmentation_probability
        probability of augmentation being applied; ignored if the `augment` key
        is supplied
    random_seed
        random number generator seed

    Attributes
    ----------
    random_variables
        dict with random variables with keys from random_variables_keys;
        automatically generated if not passed.
    random_variables_keys
        If all of these random variables are passed to `process()` they will be
        used, otherwise `create_random_variables()` is called and passed
        through.
        These variables are added as optional incoming and required generated
        keys. These variables are not passed to the augment method,
        but are saved to `self.random_variables` dict
    incoming_keys
        * augment : optional key of tf.bool and size [];
          if used overrides ``augmentation_probability`` to specify when input
          is augmented
    generated_keys
        * augment : pass-through incoming `augment` key if provided,
          otherwise generates a bool flag for this augmentation
    """
    random_variables_keys = []
    is_tensorflow = True

    def __init__(self, *,
                 augmentation_probability: float = 0.5,
                 random_seed: Optional[int] = None,
                 **processor_kwargs):
        assert 0.0 <= augmentation_probability <= 1.0, (
            "Augmentation probability must be in [0.0, 1.0]"
        )
        super().__init__(**processor_kwargs)
        self.augmentation_probability = augmentation_probability
        self.random_seed = random_seed
        self.random_variables = {k: None for k in self.random_variables_keys}
        self._initialized = False
        self._augmentation_flag = None  # type: Optional[tf.Tensor]
        self._logger = logging.getLogger(__name__)
        self._random_seed_used_times = 0

    def get_random_seed(self):
        """
        Function that generates the random seed from self.random_seed and
        the name and increments it every time it was called.

        To have more random behavior but at the same time same reproducibility,
        it is better to use this method to get the seed compared to
        self.random_seed

        Returns
        -------
        random_seed
            new random seed
        """
        if self.random_seed is None:
            return None

        random_seed_ord = utils.string_to_int_code(self.name)
        self._random_seed_used_times += 1
        return self.random_seed + random_seed_ord + self._random_seed_used_times

    @property
    def random_cond_initialized(self) -> bool:
        """
        Returns
        -------
        True if augment_condition was initialized or provided, False otherwise
        """
        return self._initialized

    # pylint: disable=no-self-argument
    # classproperty is class property and so cls must be used
    @object_utils.classproperty
    def incoming_keys_optional(cls) -> List[str]:
        extra_keys = ["augment"] + cls.random_variables_keys
        return super().incoming_keys_optional + extra_keys

    # pylint: disable=no-self-argument
    # classproperty is class property and so cls must be used
    @object_utils.classproperty
    def generated_keys_required(cls) -> List[str]:
        extra_keys = ["augment"] + cls.random_variables_keys
        return super().generated_keys_required + extra_keys

    @abc.abstractmethod
    def augment(self, **inputs) -> Dict[str, tf.Tensor]:
        """
        Method for augmentation

        To access other random variables, use self.random_variables[key] where
        key belongs to self.random_variables_keys

        Parameters
        ----------
        inputs
            data for augmentation

        Returns
        -------
        augmented
            augmented data
        """

    def not_augment(self, **inputs) -> Dict[str, tf.Tensor]:
        """
        Method for augmentation

        Parameters
        ----------
        inputs
            data for augmentation

        Returns
        -------
        augmented
            augmented data
        """
        # pylint: disable=no-self-use
        # is an interface
        return inputs

    def create_random_variables(self) -> Dict[str, tf.Tensor]:
        """
        Create random variables. This method will be called if the random
        variables specified inside of self.random_variables_keys were not
        provided on the call

        Returns
        -------
        random_variables
            created random variables
        """
        if self.random_variables_keys:
            raise NotImplementedError(
                "Implement method create_random_variables!")
        return {}

    def initialize_augment_condition(self):
        """
        Initialize the augment_condition using provided probability
        """
        self._logger.info("Create new augment condition")
        random_uniform = tf.random_uniform(
            [], minval=0, maxval=1, seed=self.get_random_seed())
        self._augmentation_flag = tf.less_equal(
            random_uniform, self.augmentation_probability)
        self._initialized = True

    def process(self, augment: Optional[tf.Tensor] = None,
                **inputs):
        # pylint: disable=arguments-differ
        # base class has more generic signature
        if augment is not None:
            self._logger.info("Use provided augment condition and ignore "
                              "self.augmentation_probability")
            self._initialized = True
            self._augmentation_flag = augment
        if self._augmentation_flag is None:
            self.initialize_augment_condition()

        (inputs_only, passed_random_variables
         ) = self._split_inputs_with_random_variables(inputs)
        self._maybe_create_random_variables(passed_random_variables)

        result = tf.cond(self._augmentation_flag,
                         true_fn=lambda: self.augment(**inputs_only),
                         false_fn=lambda: self.not_augment(**inputs_only))
        result["augment"] = self._augmentation_flag
        result.update(self.random_variables)
        return result

    def _split_inputs_with_random_variables(self, inputs):
        inputs_only = {
            each_key: each_value for each_key, each_value in inputs.items()
            if each_key not in self.random_variables_keys}
        random_variables = {
            each_key: each_value for each_key, each_value in inputs.items()
            if (each_key in self.random_variables_keys
                and each_value is not None)}
        return inputs_only, random_variables

    def _maybe_create_random_variables(self, passed_random_variables):
        random_variables_keys_set = set(self.random_variables_keys)
        passed_variables_keys_set = set(passed_random_variables)
        missing_random_variables = set.difference(
            random_variables_keys_set, passed_variables_keys_set)
        if missing_random_variables:
            common_variables = set.intersection(
                random_variables_keys_set, passed_variables_keys_set)
            self._logger.info("Initialize random variables %s",
                              list(missing_random_variables))
            random_variables_to_use = self.create_random_variables()
            if random_variables_to_use:
                self._logger.info("Created random variables %s",
                                  list(common_variables))
            if common_variables:
                self._logger.info("Use passed random variables %s",
                                  list(common_variables))
                random_variables_to_use.update(passed_random_variables)
        else:
            self._logger.info("Use passed random variables %s",
                              self.random_variables_keys)
            random_variables_to_use = passed_random_variables

        self.random_variables = random_variables_to_use
