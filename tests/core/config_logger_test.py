# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from absl.testing import parameterized

from nucleus7.core import config_logger
from nucleus7.test_utils.test_utils import TestCaseWithReset


class _Dummy(object):

    def __init__(self, par1, par2, par3=100, par4=None,
                 par5="temp"):
        self.par1 = par1
        self.par2 = par2
        self.par3 = par3
        self.par4 = par4
        self.par5 = par5
        self.name = "dummy_name"


class TestConfigLogger(TestCaseWithReset, parameterized.TestCase):

    @parameterized.parameters(
        {"use_args": True},
        {"use_args": False},
        {"use_args": True, "exclude_args": ["par5", "par1"]},
        {"use_args": False, "exclude_args": ["par5", "par1"]})
    def test_add_constructor_parameters_to_log(
            self, use_args, exclude_args=None):
        args = ()
        kwargs = {"par3": 30}
        if use_args:
            args = (10, 20)
        else:
            kwargs.update({"par1": 10, "par2": 20})

        obj = _Dummy(*args, **kwargs)
        log_parameters_must = {
            "par1": 10, "par2": 20, "par3": 30, "par4": None, "par5": "temp",
            "class_name": "config_logger_test._Dummy"}
        if exclude_args:
            del log_parameters_must["par1"]
            del log_parameters_must["par5"]

        config_logger.add_constructor_parameters_to_log(
            instance=obj, name_scope="log_ns", parameters_args=args,
            parameters_kwargs=kwargs, exclude_args=exclude_args)
        logged_configs = config_logger.get_logged_configs()
        self.assertDictEqual({"log_ns": {"dummy_name": [log_parameters_must]}},
                             logged_configs)

    def test_reset_logged_configs(self):
        obj = _Dummy(10, 20)
        config_logger.add_constructor_parameters_to_log(
            instance=obj, name_scope="log_ns", parameters_args=(10, 20),
            parameters_kwargs={})
        self.assertNotEmpty(config_logger.get_logged_configs())
        config_logger.reset_logged_configs()
        self.assertEmpty(config_logger.get_logged_configs())
