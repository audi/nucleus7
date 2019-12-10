# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Buffer specific interfaces
"""

from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import numpy as np

from nucleus7.utils import nest_utils
from nucleus7.utils import np_utils
from nucleus7.utils import object_utils
from nucleus7.utils import utils


class SamplesBuffer:
    """
    Buffer with samples
    """

    def __init__(self):
        self._buffer_flat = {}

    def get(self) -> dict:
        """
        Returns
        -------
        buffer_dict
            nested dict with buffer
        """
        return nest_utils.unflatten_dict_to_nested(self._buffer_flat)

    def get_flat(self):
        """
        Returns
        -------
        buffer_flat
            flat buffer
        """
        return self._buffer_flat

    def add(self, **sample_inputs):
        """
        Add sample to buffer

        If the key with sample data was not present inside of buffer, it will
        be padded with None

        Parameters
        ----------
        sample_inputs
            possibly nested samples of data
        """
        current_length = len(self)
        sample_inputs_flat = nest_utils.flatten_nested_struct(sample_inputs)
        all_keys = set.union(set(sample_inputs_flat), set(self.get_flat()))
        for each_key in all_keys:
            if each_key not in sample_inputs_flat:
                self._buffer_flat[each_key].append(None)
                continue

            self._buffer_flat.setdefault(each_key, [None] * current_length)
            self._buffer_flat[each_key].append(sample_inputs_flat[each_key])

    def keys(self) -> list:
        """
        Returns
        -------
        keys
            main keys in sorted order of the buffer items
        """
        keys_flat = self._buffer_flat.keys()
        keys_nested = {each_key.split("//")[0] for each_key in keys_flat}
        return sorted(keys_nested)

    def is_empty(self) -> bool:
        """
        Returns
        -------
        is_empty
            True if buffer is empty and False otherwise
        """
        return len(self) == 0

    def clear(self):
        """
        Clear buffer
        """
        self._buffer_flat = {}

    def __len__(self):
        if not self._buffer_flat:
            return 0

        return len(_get_first_key_and_subset(self._buffer_flat)[1])

    def __getitem__(self, item: Union[int, slice]) -> dict:
        buffer_slice_flat = {k: v[item] for k, v in self._buffer_flat.items()}
        return nest_utils.unflatten_dict_to_nested(buffer_slice_flat)

    def __delitem__(self, key: int):
        for each_key in self._buffer_flat:
            del self._buffer_flat[each_key][key]

    def __repr__(self):
        representation = "{} with keys {} and length {}".format(
            self.__class__.__name__, self.keys(), len(self))
        return representation


class AdditiveBuffer(SamplesBuffer):
    """
    Buffer which adds new values to current buffer values using numpy add
    """

    def __init__(self):
        super().__init__()
        self._number_of_samples = 0

    def add(self, **sample_inputs):
        _validate_buffer_keys_and_shapes(buffer=self, new_sample=sample_inputs)
        sample_inputs_flat = nest_utils.flatten_nested_struct(sample_inputs)
        for each_key, each_value in sample_inputs_flat.items():
            buffer_value = self._buffer_flat.setdefault(each_key, 0)
            new_value = self.add_value(buffer_value, each_value)
            self._buffer_flat[each_key] = new_value
        self._number_of_samples += 1

    def add_value(self, buffer_value, sample_value):
        """
        Function to add value to buffer

        Parameters
        ----------
        buffer_value
            current buffer value
        sample_value
            sample value to add to buffer

        Returns
        -------
        new_value
            new buffer value
        """
        # pylint: disable=no-self-use
        # is an interface
        buffer_value += sample_value
        return buffer_value

    def clear(self):
        super().clear()
        self._number_of_samples = 0

    def __len__(self):
        return self._number_of_samples


class AverageBuffer(AdditiveBuffer):
    """
    Buffer which averages all values added to it
    """

    def add_value(self, buffer_value, sample_value):
        new_average = (buffer_value * self._number_of_samples
                       + sample_value) / (self._number_of_samples + 1)
        return new_average


class BufferProcessor:
    """
    Processor which uses buffer to store the samples from batches of inputs
    and then processes the buffer when it was triggered using evaluate flag

    Parameters
    ----------
    process_buffer_fn
        callable which will be called on the buffer where it is triggered;
        should take kwargs as inputs and produce result as a dict
    buffer
        buffer to use; defaults to SamplesBuffer
    not_batch_keys
        keys that should be treated as not batched, e.g. they will be copied as
        is to all other sample inputs
    clear_buffer_after_evaluate
        flag if the buffer should be cleared after it was processed
    """

    def __init__(self,
                 process_buffer_fn: Optional[Callable[..., dict]] = None,
                 buffer=None,
                 not_batch_keys: Optional[list] = None,
                 clear_buffer_after_evaluate: bool = True):
        self.process_buffer_fn = process_buffer_fn
        self.clear_buffer_after_evaluate = clear_buffer_after_evaluate
        self.not_batch_keys = not_batch_keys or []
        self._buffer = buffer or SamplesBuffer()

    @property
    def extra_incoming_keys(self) -> list:
        """
        Extra keys that will be needed to use it inside of Nucleotide

        Returns
        -------
        extra_keys
            extra incoming keys to nucleotide to use this processor
        """
        return ["evaluate"]

    @property
    def buffer(self) -> SamplesBuffer:
        """
        Returns
        -------
        buffer
            current buffer
        """
        return self._buffer

    @object_utils.assert_property_is_defined("process_buffer_fn")
    def process_single_sample(self,
                              evaluate=None,
                              accumulate: bool = True,
                              **sample_inputs) -> Optional[dict]:
        """
        Process single sample by accumulating it and process the accumulator
        if evaluate=True

        Parameters
        ----------
        evaluate
            flag if the evaluation should be executed after accumulation
        accumulate
            if buffer accumulation should be executed
        sample_inputs
            sample inputs to kpi

        Returns
        -------
        result
            None if not evaluate and result of process_buffer_fn on the buffer
            otherwise
        """
        if accumulate:
            self._buffer.add(**sample_inputs)
        if not evaluate:
            return None

        if self.buffer.is_empty():
            return None

        result = self.process_buffer_fn(**self.buffer.get())
        if self.clear_buffer_after_evaluate:
            self.buffer.clear()
        return result

    def split_batch_to_samples(self,
                               **inputs) -> list:
        """
        Split batch inputs to sample inputs

        Parameters
        ----------
        inputs
            batch inputs

        Returns
        -------
        list_of_sample_inputs
            list of sample inputs
        """
        batch_inputs_as_list, not_batch_inputs = utils.split_batch_inputs(
            inputs, not_batch_keys=self.not_batch_keys)
        batch_inputs_with_not_batch_as_list = []
        for each_sample_input in batch_inputs_as_list:
            each_sample_input.update(not_batch_inputs)
            batch_inputs_with_not_batch_as_list.append(each_sample_input)
        return batch_inputs_with_not_batch_as_list

    def combine_samples_to_batch(self,
                                 list_of_sample_results: List[Optional[dict]]
                                 ) -> Optional[dict]:
        """
        Combine sample data to batch

        Parameters
        ----------
        list_of_sample_results
            list of sample results

        Returns
        -------
        batch_result
            batch result
        """
        # pylint: disable=no-self-use
        # is an interface
        list_of_sample_results_valid = [
            each_result for each_result in list_of_sample_results
            if each_result is not None]
        if not list_of_sample_results_valid:
            return None

        list_of_flat_sample_results = [
            nest_utils.flatten_nested_struct(each_result)
            for each_result in list_of_sample_results_valid]
        result_flat = {}
        for each_key in list_of_flat_sample_results[0]:
            result_flat[each_key] = np_utils.stack_with_pad(
                [each_result[each_key]
                 for each_result in list_of_flat_sample_results])
        result = nest_utils.unflatten_dict_to_nested(result_flat)
        return result

    def process_batch(self, **inputs) -> Optional[dict]:
        """
        Method to process the batch of inputs

        Parameters
        ----------
        inputs
            inputs

        Returns
        -------
        results
            processed results
        """
        samples_inputs = self.split_batch_to_samples(**inputs)
        results = []
        for each_sample_input in samples_inputs:
            sample_result = self.process_single_sample(
                **each_sample_input)
            results.append(sample_result)
        results_combined = self.combine_samples_to_batch(results)
        return results_combined


def _get_first_key_and_subset(file_names: dict):
    first_key = sorted(file_names.keys())[0]
    return first_key, file_names[first_key]


def _validate_buffer_keys_and_shapes(buffer: SamplesBuffer, new_sample: dict):
    sample_flat = nest_utils.flatten_nested_struct(new_sample)
    sample_keys_flat = set(sample_flat)
    buffer_keys_flat = set(buffer.get_flat())
    if buffer_keys_flat:
        if set.symmetric_difference(sample_keys_flat, buffer_keys_flat):
            msg = ("{} allows only samples with same keys! "
                   "(flat keys inside buffer: {}, "
                   "flat keys of new sample: {})"
                   ).format(buffer.__class__.__name__,
                            buffer_keys_flat, sample_keys_flat)
            raise ValueError(msg)
        for each_key, each_sample_value in sample_flat.items():
            buffer_value = buffer.get_flat()[each_key]
            if not np.allclose(np.shape(buffer_value),
                               np.shape(each_sample_value)):
                msg = ("Value of key {} has different shape compared to value "
                       "inside of buffer {}! (buffer shape: {}, "
                       "new value shape: {})"
                       ).format(each_key, buffer.__class__.__name__,
                                buffer_value.shape, np.shape(each_sample_value))
                raise ValueError(msg)
