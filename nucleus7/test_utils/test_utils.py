# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
General test utils
"""

import shutil
import tempfile
import unittest

import tensorflow as tf

from nucleus7.core.config_logger import reset_logged_configs
from nucleus7.core.project import reset_project
from nucleus7.core.project_artifacts import reset_artifacts
from nucleus7.core.register import reset_register
from nucleus7.utils import tf_data_utils


class TestCaseWithReset(unittest.TestCase):
    """
    TestCase which will reset register and logger for every test
    """

    def setUp(self):
        reset_register_and_logger()


class TestCaseWithTempDir(unittest.TestCase):
    """
    TestCase which will create the temp directory on every test and then
    remove it when test is over
    """

    def setUp(self):
        self._temp_dir = None

    def get_temp_dir(self) -> str:
        """
        Get temp directory

        Returns
        -------
        temp_dir_path
            path to directory
        """
        if self._temp_dir is None:
            self._temp_dir = tempfile.mkdtemp()
        return self._temp_dir

    def tearDown(self):
        if self._temp_dir is not None:
            shutil.rmtree(self._temp_dir)


def register_new_class(name, super_cls):
    """
    Create new class and register it class with the register name and inherit
    the class from super_cls

    Parameters
    ----------
    name
        register name
    super_cls
        super class for new class

    """

    # pylint: disable=unused-variable
    # Dummy class name is used for convention and is registered automatically
    # pylint: disable=too-few-public-methods
    # it inherits from the super_cls
    # pylint: disable=missing-docstring
    # is dummy, so docstring is not needed
    class Dummy(super_cls):
        register_name = name

        def __init__(self, *args, **kwargs):
            try:
                super().__init__(inbound_nodes=None, *args, **kwargs)
            except TypeError:
                super().__init__(*args, **kwargs)


def reset_register_and_logger():
    """
    Reset register and logger
    """
    reset_register()
    reset_logged_configs()
    reset_project()
    reset_artifacts()


def write_tf_records(data, file_name, writer=None, close_writer=True):
    """
    write tfrecords file with data

    Parameters
    ----------
    data
        sample data to write
    file_name
        file name for tfrecords
    writer
        tfrecords writer to use
    close_writer
        if the writer should be closed
    """
    writer = writer or tf.python_io.TFRecordWriter(file_name)
    feature = tf_data_utils.nested_to_tfrecords_feature(data)

    example = tf.train.Example(
        features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
    if close_writer:
        writer.close()
