# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Session Run Hooks used inside of tensorflow sessions
"""
import logging
import os
import time
from typing import Callable
from typing import Dict
from typing import List
from typing import Union

import tensorflow as tf
try:
    from tensorflow.contrib.estimator.python.estimator import early_stopping
except ImportError:
    from tensorflow.python.estimator import early_stopping

from nucleus7.coordinator.callback import CoordinatorCallback
from nucleus7.coordinator.callbacks_handler import CallbacksHandler
from nucleus7.coordinator.configs import RunIterationInfo
from nucleus7.core.nucleotide import Nucleotide
from nucleus7.model.fields import CollectionNames
from nucleus7.utils import io_utils
from nucleus7.utils import model_utils
from nucleus7.utils import tf_collections_utils


class SummarySaverHook(tf.train.SummarySaverHook):
    """
    Save summaries in slightly different format as original one and to
    different folder. x axis of summaries represents global step
    and allows to shift evaluation so that last step of evaluation in one
    epoch is aligned with last step from training.

    Parameters
    ----------
    save_steps
        number of steps between summary saves
    save_secs
        number of seconds between summary saves
    output_dir
        summary directories
    get_graph_fn
        function to get the graph, which is stored to summaries
    step_offset
        offset of the step on x axis of summaries
    flush_after_n_summaries
        number of iteration till summary writer will be flushed to the disk;
        setting it too high may cause increase in memory load
    """

    # pylint: disable=too-many-instance-attributes
    # not possible to have less arguments without more complexity
    def __init__(self,
                 save_steps: int = None,
                 save_secs: int = None,
                 output_dir: str = None,
                 get_graph_fn: Callable = None,
                 step_offset: int = 0,
                 flush_after_n_summaries: int = 100):
        # pylint: disable=super-init-not-called
        # SummaryWriterHook is used more as an interface
        # pylint: disable=too-many-arguments
        # not possible to have less arguments without more complexity
        self._summary_writer = None
        self._get_graph_fn = get_graph_fn or tf.get_default_graph
        self._output_dir = output_dir
        self._scaffold = None
        self._summary_op = None
        self._timer = tf.train.SecondOrStepTimer(every_secs=save_secs,
                                                 every_steps=save_steps)
        self._step_offset_init = step_offset
        self._step_offset = step_offset
        self._flush_after_n_summaries = flush_after_n_summaries
        self._num_saved_summaries = 0

    def begin(self):
        """
        Add graph to summary writer and create directory for summaries it it
        does not exist

        overridden from :obj:`tf.train.SummarySaverHook`. See its documentation
        for more information
        """
        self._summary_op = tf.get_collection(tf.GraphKeys.SUMMARIES)
        super().begin()
        io_utils.maybe_mkdir(self._output_dir)
        graph = self._get_graph_fn()
        self._summary_writer.add_graph(graph)

    def after_run(self, run_context, run_values):
        """
        Save summaries to event files

        overridden from :obj:`tf.train.SummarySaverHook`. See its documentation
        for more information
        """
        _ = run_context
        if not self._summary_writer:
            return
        logger = logging.getLogger(__name__)

        global_step = run_values.results["global_step"]
        if self._next_step is None or self._request_summary:
            global_step = run_context.session.run(self._global_step_tensor)

        save_step = (global_step - self._step_offset)

        if self._request_summary:
            self._timer.update_last_triggered_step(save_step)
            if "summary" in run_values.results:
                for summary in run_values.results["summary"]:
                    self._summary_writer.add_summary(summary, save_step)

        # pylint: disable=attribute-defined-outside-init
        # is from tensorflow SaverHook
        self._next_step = save_step + 1
        if self._step_offset_init > 0:
            self._step_offset -= 1
        if self._step_offset == 0:
            self._step_offset = self._step_offset_init

        if self._num_saved_summaries % self._flush_after_n_summaries == 0:
            logger.info("Flush summary writer")
            self._summary_writer.flush()
            self._num_saved_summaries = 0
        self._num_saved_summaries += 1


class MetricUpdateHook(tf.train.SessionRunHook):
    """
    Fetch all metrics together with their update ops, so they
    are evaluated inside of Evaluator.

    Normally, Evaluator does not use metrics for training, so this hook is
    needed for it
    """

    def before_run(self, run_context: tf.train.SessionRunContext):
        """
        Add metrics to fetches, so they are returned at the end of training
        iteration

        overridden from :obj:`tf.train.SessionRunHook`. See its documentation
        for more information
        """
        try:
            metrics = tf_collections_utils.collection2nested(
                CollectionNames.METRIC)
            return tf.train.SessionRunArgs(fetches=metrics)
        except ValueError:
            return tf.train.SessionRunArgs(fetches=None)


class CustomNucleotideInitializerHook(tf.train.SessionRunHook):
    """
    This session run hook will call initialization of nucleotides which have
    `CustomSessionHandlerMixin` interface or have `initialize_session`
    implemented
    """

    def __init__(self, nucleotides: Union[List[Nucleotide],
                                          Dict[str, Nucleotide]]):
        if isinstance(nucleotides, list):
            nucleotides = {each_nucleotide.name: each_nucleotide
                           for each_nucleotide in nucleotides}
        self._nucleotides = nucleotides

    def after_create_session(self, session, coord):
        for each_nucleotide in self._nucleotides.values():
            if hasattr(each_nucleotide, "initialize_session"):
                with session.as_default():
                    each_nucleotide.initialize_session()
        super().after_create_session(session, coord)


# pylint: disable=no-member,protected-access,too-few-public-methods
# This class exists and can be accessed only as a private
class EarlyStoppingInitHook(early_stopping._CheckForStoppingHook):
    """
    Wrapper across private _CheckForStoppingHook
    """
# pylint: enable=no-member,protected-access,too-few-public-methods


class _CallBackSessionRunHook(tf.train.SessionRunHook):
    """
    Execute :obj:`CoordinatorCallback` or :obj:`CallbacksHandler` as
    :obj:`tf.train.SessionRunHook` for use inside of trainer session

    Parameters
    ----------
    callback_or_handler
        callback or callback handler call inside of the session hook
    summary_dir
        directory of summary writer
    max_number_of_iterations_per_epoch
        max number of iterations for all modes; needed to recalculate iteration
        number from global_step.
    """

    def __init__(self,
                 callback_or_handler: Union[CoordinatorCallback,
                                            CallbacksHandler],
                 summary_dir: str,
                 max_number_of_iterations_per_epoch: int = 0):
        self._callback_or_handler = callback_or_handler
        self._summary_dir = summary_dir
        self._global_step_tensor = None
        self._time_run_start = None
        self._max_number_of_iterations_per_epoch = (
            max_number_of_iterations_per_epoch)

    def begin(self):
        """
        overridden from :obj:`tf.train.SessionRunHook`. See its documentation
        for more information
        """
        self._callback_or_handler.begin()
        summary_writer = tf.summary.FileWriterCache.get(self._summary_dir)
        self._callback_or_handler.summary_writer = summary_writer
        self._global_step_tensor = tf.train.get_or_create_global_step()

    def before_run(self, run_context):
        """
        overridden from :obj:`tf.train.SessionRunHook`. See its documentation
        for more information
        """
        logger = logging.getLogger(__name__)
        self._callback_or_handler.on_iteration_start()
        self._time_run_start = time.time()
        graph = tf.get_default_graph()
        inputs_preprocessed = tf_collections_utils.collection2nested(
            CollectionNames.INPUTS_PREPROCESSED, graph=graph,
            raise_error=False)
        predictions_raw = tf_collections_utils.collection2nested(
            CollectionNames.PREDICTIONS_RAW, graph=graph,
            raise_error=False)
        predictions = tf_collections_utils.collection2nested(
            CollectionNames.PREDICTIONS, graph=graph,
            raise_error=False)
        losses = tf_collections_utils.collection2nested(
            CollectionNames.LOSSES, graph=graph,
            raise_error=False)
        metrics = tf_collections_utils.collection2nested(
            CollectionNames.METRIC, graph=graph,
            raise_error=False)
        data_all = dict()
        data_all['dataset'] = inputs_preprocessed
        data_all.update(predictions)
        data_all.update(predictions_raw)
        data_all.update(losses)
        data_all.update(metrics)
        data2callback = self._callback_or_handler.filter_inputs(inputs=data_all)
        logger.debug('Callback %s has following inputs: %s',
                     self._callback_or_handler.name, data2callback.keys())
        requests = {'data': data2callback,
                    "global_step": self._global_step_tensor}
        return tf.train.SessionRunArgs(requests)

    def after_run(self, run_context, run_values):
        """
        overridden from :obj:`tf.train.SessionRunHook`. See its documentation
        for more information
        """
        execution_time = time.time() - self._time_run_start
        data = run_values.results['data']
        global_step = run_context.session.run(self._global_step_tensor)
        mode = self._callback_or_handler.mode
        number_iterations_per_epoch = (
            self._callback_or_handler.number_iterations_per_epoch)
        iteration_number = (
            self._callback_or_handler.iteration_info.iteration_number)

        epoch_number, iteration_number, summary_step = (
            model_utils.get_iteration_stat_from_global_step(
                mode=mode,
                global_step=global_step,
                previous_iteration_number=iteration_number,
                number_iterations_per_epoch=number_iterations_per_epoch,
                max_number_of_iterations_per_epoch=
                self._max_number_of_iterations_per_epoch
            ))
        is_last_iteration = not iteration_number % number_iterations_per_epoch
        iteration_info = RunIterationInfo(
            epoch_number=epoch_number,
            iteration_number=iteration_number,
            execution_time=execution_time,
            is_last_iteration=is_last_iteration,
            session_run_context=run_context)
        self._callback_or_handler.iteration_info = iteration_info
        self._callback_or_handler.summary_step = summary_step
        self._callback_or_handler(**data)

    def end(self, session):
        """
        overridden from :obj:`tf.train.SessionRunHook`. See its documentation
        for more information
        """
        self._callback_or_handler.end()
        super().end(session)


def convert_callback_to_session_hook(
        callback: CoordinatorCallback,
        summary_dir: str,
        max_number_of_iterations_per_epoch: int
) -> tf.train.SessionRunHook:
    """
    Convert callback to :obj:`tf.train.SessionRunHook`

    Parameters
    ----------
    callback
        callback to transform
    summary_dir
        directory of summary writer
    max_number_of_iterations_per_epoch
        max number of iterations for all modes; needed to recalculate iteration
        number from global_step.

    Returns
    -------
    callback_as_hook
        callback as session_run_hook
    """
    assert callback.mode is not None, (
        "Set the mode for callback {}!".format(callback.name)
    )
    summary_dir = _get_and_maybe_create_summary_dir_for_mode(
        summary_dir, callback.mode)
    callback_as_hook = _CallBackSessionRunHook(
        callback_or_handler=callback, summary_dir=summary_dir,
        max_number_of_iterations_per_epoch=max_number_of_iterations_per_epoch)
    return callback_as_hook


def convert_callbacks_handler_to_session_hook(
        callbacks_handler: CallbacksHandler,
        summary_dir: str,
        max_number_of_iterations_per_epoch: int
) -> tf.train.SessionRunHook:
    """
    Convert callbacks handler to session run hook

    Parameters
    ----------
    callbacks_handler
        callbacks handler to convert
    summary_dir
        directory to save the summaries
    max_number_of_iterations_per_epoch
        max number of iterations per epoch, use to calculate the step from
        global_step

    Returns
    -------
    callback_as_hook
        session run hook
    """
    assert callbacks_handler.mode is not None, (
        "Set the mode for callbacks handler!"
    )
    summary_dir = _get_and_maybe_create_summary_dir_for_mode(
        summary_dir, callbacks_handler.mode)
    callback_as_hook = _CallBackSessionRunHook(
        callback_or_handler=callbacks_handler, summary_dir=summary_dir,
        max_number_of_iterations_per_epoch=max_number_of_iterations_per_epoch)
    return callback_as_hook


def _get_and_maybe_create_summary_dir_for_mode(
        summary_dir: str, mode: str) -> str:
    summary_dir = os.path.join(summary_dir, mode)
    io_utils.maybe_mkdir(summary_dir)
    return summary_dir
