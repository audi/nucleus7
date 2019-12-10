# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import os

from absl.testing import parameterized
import matplotlib

matplotlib.use("agg")
import pytest
import tensorflow as tf

from nucleus7.test_utils.test_utils import reset_register_and_logger
from nucleus7.utils.jupyter_utils import convert_jupyter_to_python
from nucleus7.utils.jupyter_utils import read_file_and_execute


class TestTutorials(parameterized.TestCase, tf.test.TestCase):
    def setUp(self):
        reset_register_and_logger()

    @parameterized.parameters(
        {"file_test": "build_and_train.ipynb",
         "prefix": "number_epochs = 2; number_of_iterations = 3"},
        {"file_test": "inference.ipynb", "prefix": "number_of_samples = 10"})
    @pytest.mark.slow
    @pytest.mark.jupyter
    def test_file_not_crashing(self, file_test, prefix):
        tests_dir = os.path.dirname(__file__)
        tutorials_dir = os.path.join(
            os.path.split(tests_dir)[0], "tutorials", "notebooks")
        in_file = os.path.join(tutorials_dir, file_test)
        out_file = convert_jupyter_to_python(in_file)
        tutorials_dir = os.path.split(in_file)[0]
        read_file_and_execute(out_file, delete=True, prefix=prefix,
                              exec_dir=tutorials_dir)
