# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Accumulator for KPI plugin values
"""

from typing import Optional

from nucleus7.core.buffer import BufferProcessor
from nucleus7.core.buffer import SamplesBuffer
from nucleus7.kpi.kpi_plugin import KPIPlugin
from nucleus7.utils import object_utils


# pylint: disable=abstract-method
# this class is also an interface
class KPIAccumulator(KPIPlugin):
    """
    Special case of KPIPlugin, which is used to accumulate the intermediate
    KPI values and then to combine them

    Parameters
    ----------
    buffer_processor
        buffer processor to use
    clear_buffer_after_evaluate
        flag if the buffer should be cleared after it was processed
    """
    register_name_scope = "kpi_accumulator"
    exclude_from_register = True

    def __init__(self, buffer_processor: Optional[BufferProcessor] = None,
                 clear_buffer_after_evaluate: bool = True,
                 **kpi_plugin_kwargs):
        super().__init__(**kpi_plugin_kwargs)
        self.buffer_processor = buffer_processor or BufferProcessor()
        self.buffer_processor.process_buffer_fn = (
            super(KPIAccumulator, self).evaluate_on_sample)
        self.buffer_processor.clear_buffer_after_evaluate = (
            clear_buffer_after_evaluate)
        self._last_kpi = None
        self._last_prefix = None
        self._last_sample_evaluated = False

    # pylint: disable=no-self-argument
    # classproperty is class property and so cls must be used
    @object_utils.classproperty
    def incoming_keys_optional(cls):
        extra_keys = ["evaluate"]
        return super().incoming_keys_optional + extra_keys

    @property
    def last_kpi(self) -> Optional[dict]:
        """
        Returns
        -------
        last_kpi
            last evaluated kpi
        """
        return self._last_kpi

    @property
    def buffer(self) -> SamplesBuffer:
        """
        Returns
        -------
        buffer
            buffer
        """
        return self.buffer_processor.buffer

    def clear_state(self):
        super(KPIAccumulator, self).clear_state()
        self.buffer_processor.buffer.clear()
        self._last_kpi = None
        self._last_prefix = None
        self._last_sample_evaluated = False

    def evaluate_on_sample(self, *, evaluate=None,
                           prefix: Optional[str] = None,
                           **sample_inputs) -> dict:
        """
        Main method of accumulator

        Parameters
        ----------
        evaluate
            flag if the evaluation should be executed after accumulation
        prefix
            prefix to use inside of cachers and savers
        sample_inputs
            sample inputs to kpi

        Returns
        -------
        kpi
            evaluated kpi; if evaluate == False, then it will return None
        """
        # pylint: disable=arguments-differ
        # super evaluate_sample has more generic signature
        if self.is_last_sample:
            evaluate = True

        result = self.buffer_processor.process_single_sample(
            evaluate=evaluate, prefix=prefix, **sample_inputs)
        if result:
            self._last_kpi = {k: v for k, v in result.items()}
        if self.is_last_sample:
            self._last_sample_evaluated = True
        self._last_prefix = prefix
        return result

    def finalize(self):
        """
        Final evaluation. It will evaluate the kpi on the accumulated buffer
        in case if self.is_last_sample was set to True and last sample was
        not evaluated
        """
        if not self.is_last_iteration:
            return

        if self._last_sample_evaluated:
            return

        result = self.buffer_processor.process_single_sample(
            evaluate=True, prefix=self._last_prefix, accumulate=False)
        if result:
            self._last_kpi = {k: v for k, v in result.items()}

    def _get_full_prefix(self, cache_prefix: Optional[str] = None):
        if (isinstance(cache_prefix, (tuple, list)) and
                all([each_prefix is None for each_prefix in cache_prefix])):
            cache_prefix = None

        if isinstance(cache_prefix, (tuple, list)) and len(cache_prefix) > 2:
            cache_prefix = [cache_prefix[0], cache_prefix[-1]]
            if len(set(cache_prefix)) == 1:
                cache_prefix = cache_prefix
        return super()._get_full_prefix(cache_prefix)
