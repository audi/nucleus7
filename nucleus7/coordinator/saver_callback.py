# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Callbacks to save data
"""
import abc
import logging
import os
from typing import Optional

import tensorflow as tf

# pylint: disable=cyclic-import
from nucleus7.coordinator import CoordinatorCallback
# pylint: enable=cyclic-import
from nucleus7.utils import file_utils
from nucleus7.utils import io_utils
from nucleus7.utils import object_utils
from nucleus7.utils import project_utils
from nucleus7.utils import tf_data_utils
from nucleus7.utils import utils


# pylint: disable=too-many-instance-attributes
# attributes cannot be combined or extracted further
class SaverCallback(CoordinatorCallback):
    """
    Callbacks to save data sample-wise

    It will take inputs as batches, unstack them to be samples and then all
    the samples will be saved.

    As save_name, it can use a sample value from the key "save_names" or will
    construct it from iteration_info if it was provided

    Parameters
    ----------
    not_batch_keys
        keys to exclude from input batch split; that keys will be added to all
        the sample inputs
    save_prefix
        prefix to use before additional save name; it will be joined with "-"
    save_suffix
        suffix to use after additional save name; it will be joined with "-"
    target_separator
        separator to use to join save_target together with the full save name
    remove_save_ext
        if the save extension should be removed from save name
    save_name_depth
        depth of save name to preserve, if save_name was provided; e.g. if
        save_name was a/b/c/d and save_name_depth=0, then only d will be used
        and with save_name_depth=2 b/c/d will be used
    """
    exclude_from_register = True

    def __init__(self,
                 save_prefix: Optional[str] = None,
                 save_suffix: Optional[str] = None,
                 target_separator: str = "/",
                 not_batch_keys: Optional[list] = None,
                 remove_save_ext: bool = True,
                 save_name_depth: int = 0,
                 **callback_kwargs):
        super().__init__(**callback_kwargs)
        self.not_batch_keys = not_batch_keys
        self.save_prefix = save_prefix
        self.save_suffix = save_suffix
        self.target_separator = target_separator
        self.remove_save_ext = remove_save_ext
        self.save_name_depth = save_name_depth
        self._save_name = None
        self._last_unsaved_sample = None
        self._sample_index = 0

    # pylint: disable=no-self-argument
    # classproperty is class property and so cls must be used
    @object_utils.classproperty
    def incoming_keys_optional(cls):
        extra_keys = ["save_names", "save"]
        return super().incoming_keys_optional + extra_keys

    @property
    def save_name(self) -> str:
        """
        Returns
        -------
        save_name
            full save target including save name
        """
        return self._save_name

    @property
    def sample_index(self) -> int:
        """
        Returns
        -------
        sample_index
            sample index inside of the batch
        """
        return self._sample_index

    @object_utils.assert_property_is_defined("log_dir")
    def set_save_name(self, save_name: Optional[str] = None):
        """
        Set full save target by joining prefix, save name, suffix together
        with save_target

        Parameters
        ----------
        save_name
            save name
        """
        logger = logging.getLogger(__name__)
        if isinstance(save_name, bytes):
            save_name = save_name.decode()

        if self.remove_save_ext and save_name:
            save_name = os.path.splitext(save_name)[0]
        if save_name:
            save_name = os.path.join(*file_utils.get_basename_with_depth(
                save_name, self.save_name_depth))

        if not save_name:
            save_name = self.get_save_name_from_iteration_info()

        additional_names = [self.save_prefix, save_name, self.save_suffix]
        additional_names_concat = "-".join(
            [each_name for each_name in additional_names if
             each_name is not None])
        self._save_name = self.target_separator.join(
            [self.log_dir, additional_names_concat])

        if save_name and os.path.sep in save_name:
            root_directory = os.path.split(self._save_name)[0]
            try:
                os.makedirs(root_directory)
                logger.info("Directory %s was created by %s",
                            root_directory, self.name)
            except FileExistsError:
                logger.debug("Directory %s needed by %s already exists",
                             root_directory, self.name)
                io_utils.maybe_mkdir(root_directory)

    def get_save_name_from_iteration_info(self):
        """
        Compose save name from iteration info like epoch number, iteration
        number and sample index in that batch

        Returns
        -------
        save_name_from_iteration
            save name from iteration if iteration_info was provided and None
            otherwise
        """
        if self.iteration_info.epoch_number > 0:
            save_name = "epoch_{:03d}_iter_{:05d}_sample_{:03d}".format(
                self.iteration_info.epoch_number,
                self.iteration_info.iteration_number, self.sample_index)
            return save_name
        return None

    @abc.abstractmethod
    def save_sample(self, **sample_data):
        """
        Main method how to save the sample

        Parameters
        ----------
        sample_data
            sample data to save
        """

    def split_batch_inputs(self, **data):
        """
        Split possibly nested batch inputs to sample wise

        Parameters
        ----------
        data
            batch data

        Returns
        -------
        sample_data_as_list
            list of sample data
        """
        (batch_data_as_list, not_batch_inputs
         ) = utils.split_batch_inputs(data,
                                      not_batch_keys=self.not_batch_keys)
        batch_data_as_list_with_not_batch = []
        for each_sample_data in batch_data_as_list:
            each_sample_data.update(not_batch_inputs)
            batch_data_as_list_with_not_batch.append(each_sample_data)
        return batch_data_as_list_with_not_batch

    def on_iteration_end(self, **data):
        logger = logging.getLogger(__name__)
        batch_data_as_list = self.split_batch_inputs(**data)
        for i_sample, each_sample_data in enumerate(batch_data_as_list):
            should_save = each_sample_data.pop("save", True)
            save_name = each_sample_data.pop("save_names", None)
            self._sample_index = i_sample
            self.set_save_name(save_name)
            if should_save:
                self.save_sample(**each_sample_data)
            else:
                logger.info("Current inputs of sample %s to %s will be skipped "
                            "due to save=False", i_sample, self.name)
                self._last_unsaved_sample = each_sample_data

    def end(self):
        super(SaverCallback, self).end()
        logger = logging.getLogger(__name__)
        if self._last_unsaved_sample:
            logger.info("Save last unsaved inputs with %s", self.name)
            self.save_sample(**self._last_unsaved_sample)
            self._last_unsaved_sample = None
        logger.info("Saver %s closed", self.name)


# pylint: enable=too-many-instance-attributes

# pylint: disable=too-many-instance-attributes
# is needed to make the TfRecordsSaverCallback more generic
class TfRecordsSaverCallback(SaverCallback):
    """
    Saver callback which stores sample data to tfrecords files

    Parameters
    ----------
    number_of_samples_per_file
        max number of samples in one file
    compression_type
        type of compression to be used, if not specified then its not applied
        there are two types of compression possible {'GZIP', 'ZLIB'}
    ignore_empty_arrays
        if the array is empty, e.g. it has 0 in the shape, it will not store
        it; tfrecords parser raises
        'Invalid argument: Key: {key}. Can't parse serialized Example' on
        empty arrays
    """
    dynamic_incoming_keys = True

    def __init__(self,
                 number_of_samples_per_file: int = 50,
                 save_prefix: Optional[str] = "data",
                 file_ext: str = "tfrecords",
                 compression_type: Optional[str] = None,
                 ignore_empty_arrays: bool = True,
                 **saver_kwargs):
        super().__init__(save_prefix=save_prefix, **saver_kwargs)
        self.file_ext = file_ext
        self.number_of_samples_per_file = number_of_samples_per_file
        if compression_type:
            assert compression_type in ['GZIP', 'ZLIB'], (
                "Compression type must be in {}, received {}!!!".format(
                    ['GZIP', 'ZLIB'], compression_type))
        self.compression_type = compression_type
        if self.compression_type:
            self.save_suffix = "-".join(
                [self.save_suffix, self.compression_type])
        self.ignore_empty_arrays = ignore_empty_arrays
        self._tfrecords_writer = None  # type: tf.python_io.TFRecordWriter
        self._num_samples_written = 0
        self._last_additional_name = None
        self._current_additional_name = None
        self._file_locker = None  # type: project_utils.ProjectLock

    def set_save_name(self, save_name: Optional[str] = None):
        super().set_save_name(save_name)
        self._save_name = ".".join([self._save_name, self.file_ext])
        self._current_additional_name = save_name

    def get_save_name_from_iteration_info(self):
        return None

    def create_tfrecords_features(self, **data):
        """
        Create tfrecords features from the data

        Parameters
        ----------
        data
            sample data

        Returns
        -------
        features
            tfrecords features
        """
        # pylint: disable=no-self-use
        # is an interface
        feature = tf_data_utils.nested_to_tfrecords_feature(
            data, ignore_empty_arrays=self.ignore_empty_arrays)
        return feature

    def save_sample(self, **data):
        logger = logging.getLogger(__name__)
        self._maybe_create_new_tfrecords_writer()
        feature = self.create_tfrecords_features(**data)
        logger.info("%s: save sample with %s feature keys",
                    self.name, list(feature))
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        self._tfrecords_writer.write(example.SerializeToString())
        self._num_samples_written += 1

    def create_new_tfrecords_writer(self):
        """
        Creates new file_writer and closes old one

        Also if needed, increments the file name
        """
        logger = logging.getLogger(__name__)
        if self._file_locker is None:
            self._create_file_locker()
        self._file_locker.lock_or_wait()
        self._maybe_close_tfrecords_writer()
        writer_file_name = file_utils.get_incremented_path(
            self.save_name, True)
        if self.compression_type:
            options = tf.python_io.TFRecordOptions(
                getattr(tf.python_io.TFRecordCompressionType,
                        self.compression_type))
        else:
            options = []
        logger.info("Create new tfrecords writer for file %s",
                    writer_file_name)
        self._tfrecords_writer = tf.python_io.TFRecordWriter(
            writer_file_name, options=options)
        self._file_locker.release()

    def end(self):
        super(TfRecordsSaverCallback, self).end()
        self._maybe_close_tfrecords_writer()

    def _maybe_create_new_tfrecords_writer(self):
        create_based_on_samples_written = not (self._num_samples_written %
                                               self.number_of_samples_per_file)
        create_based_on_additional_name = (self._last_additional_name
                                           != self._current_additional_name)
        if create_based_on_samples_written or create_based_on_additional_name:
            self.create_new_tfrecords_writer()
            self._num_samples_written = 0
            self._last_additional_name = self._current_additional_name

    def _maybe_close_tfrecords_writer(self):
        logger = logging.getLogger(__name__)
        if self._tfrecords_writer is not None:
            logger.info("Close last tfrecords writer")
            self._tfrecords_writer.close()

    def _create_file_locker(self):
        lock_name = ".".join([self.name, 'lockname'])
        self._file_locker = project_utils.ProjectLock(self.log_dir, lock_name)
