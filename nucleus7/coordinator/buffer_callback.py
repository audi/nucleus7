# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Buffer Callback
"""

import abc
from typing import Optional

import numpy as np

from nucleus7.coordinator.callback import CoordinatorCallback
from nucleus7.core.buffer import BufferProcessor
from nucleus7.core.buffer import SamplesBuffer
from nucleus7.utils import nest_utils
from nucleus7.utils import object_utils
from nucleus7.utils import utils


class BufferCallback(CoordinatorCallback):
    """
    Callback that uses buffer to store the inputs and then when it needs to
    be retrieved, it processes the buffer

    Parameters
    ----------
    buffer_processor
        buffer processor to use
    clear_buffer_after_evaluate
        flag if the buffer should be cleared after it was processed
    not_batch_keys
        keys to exclude from input batch split; that keys will be added to all
        the sample inputs
    """
    exclude_from_register = True

    def __init__(self, buffer_processor: Optional[BufferProcessor] = None,
                 clear_buffer_after_evaluate: bool = True,
                 not_batch_keys: Optional[list] = None,
                 **kpi_plugin_kwargs):
        super().__init__(**kpi_plugin_kwargs)
        self.not_batch_keys = not_batch_keys or []
        self.buffer_processor = buffer_processor or BufferProcessor()
        self.buffer_processor.process_buffer_fn = self.process_buffer
        self.buffer_processor.not_batch_keys = not_batch_keys
        self.buffer_processor.clear_buffer_after_evaluate = (
            clear_buffer_after_evaluate)

    # pylint: disable=no-self-argument
    # classproperty is class property and so cls must be used
    @object_utils.classproperty
    def incoming_keys_optional(cls):
        extra_keys = ["evaluate"]
        return super().incoming_keys_optional + extra_keys

    @abc.abstractmethod
    def process_buffer(self, **buffer_data):
        """
        Main method to process the accumulated buffer

        Parameters
        ----------
        buffer_data
            data from buffer

        Returns
        -------
        summarized_data
            summarized data
        """

    @property
    def buffer(self) -> SamplesBuffer:
        """
        Returns
        -------
        buffer
            buffer
        """
        return self.buffer_processor.buffer

    def on_iteration_end(self, evaluate=None, **data):
        # pylint: disable=arguments-differ
        # parent on_iteration_end method has more generic signature
        evaluate = self._update_evaluate(evaluate, **data)
        return self.buffer_processor.process_batch(evaluate=evaluate, **data)

    def _update_evaluate(self, evaluate, **data):
        data_batch = {k: v for k, v in data.items()
                      if v not in self.not_batch_keys}
        if not data_batch:
            raise ValueError(
                "No batch data was provided to {}".format(self.name))
        data_flat = nest_utils.flatten_nested_struct(data_batch)
        first_data_key = list(data_flat.keys())[0]
        batch_size = np.shape(data_flat[first_data_key])[0]
        is_last_sample = utils.get_is_last_sample_batchwise(
            batch_size, is_last_iteration=self.iteration_info.is_last_iteration)
        if not evaluate:
            evaluate = np.asarray(is_last_sample)
        else:
            evaluate = np.logical_or(evaluate, is_last_sample)
        return evaluate

    def end(self):
        super(BufferCallback, self).end()
        self.buffer.clear()
