# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from unittest.mock import call as mock_call
from unittest.mock import patch

from absl.testing import parameterized

from nucleus7.core import base
from nucleus7.test_utils.test_utils import reset_register_and_logger


class _DummyBaseClass(base.BaseClass):

    def __init__(self, par1=None):
        self.par1 = par1
        super(_DummyBaseClass, self).__init__()

    @property
    def defaults(self):
        return {"par1": "default"}


class TestMetaLogAndRegister(parameterized.TestCase):
    def setUp(self):
        reset_register_and_logger()

    @parameterized.parameters(
        {"use_child": False, "register_name": "reg_name",
         "register_name_scope": "reg_ns", "log_name_scope": "log_ns",
         "exclude_args_from_log": ["par1", "par2"],
         "exclude_from_register": False, "exclude_from_log": False},
        {"use_child": False,
         "register_name_scope": "reg_ns", "log_name_scope": "log_ns",
         "exclude_args_from_log": ["par1", "par2"],
         "exclude_from_register": True, "exclude_from_log": False},
        {"use_child": False},
        {"use_child": True, "register_name": "reg_name",
         "register_name_scope": "reg_ns", "log_name_scope": "log_ns",
         "exclude_args_from_log": ["par1", "par2"],
         "exclude_from_register": False, "exclude_from_log": False},
        {"use_child": True,
         "register_name_scope": "reg_ns", "log_name_scope": "log_ns",
         "exclude_args_from_log": ["par1", "par2"],
         "exclude_from_register": True, "exclude_from_log": False},
        {"use_child": True},
    )
    @patch("nucleus7.core.register.register_to_name_scope")
    @patch("nucleus7.core.config_logger.add_constructor_parameters_to_log")
    def test_init(self, add_constructor_parameters_to_log_fn,
                  register_to_name_scope_fn,
                  use_child, register_name=None, register_name_scope=None,
                  log_name_scope=None, exclude_args_from_log=None,
                  exclude_from_register=False, exclude_from_log=False):

        ClsWithMeta = self._declare_cls(
            register_name_=register_name,
            register_name_scope_=register_name_scope,
            log_name_scope_=log_name_scope,
            exclude_from_register_=exclude_from_register,
            exclude_from_log_=exclude_from_log,
            exclude_args_from_log_=exclude_args_from_log
        )
        cls = ClsWithMeta
        cls_parent = None
        if use_child:
            cls_parent = cls

            class ChildClsWithMeta(ClsWithMeta):
                pass

            cls = ChildClsWithMeta

        if use_child:
            self.assertEqual(cls.__name__,
                             cls._register_name)
        else:
            self.assertEqual(register_name,
                             cls._register_name)
        if use_child:
            self.assertFalse(cls._exclude_from_register)
        else:
            self.assertEqual(exclude_from_register or False,
                             cls._exclude_from_register)

        if use_child:
            self.assertFalse(cls._exclude_from_log)
        else:
            if exclude_from_log is None:
                self.assertEqual(exclude_from_register or False,
                                 cls._exclude_from_log)
            else:
                self.assertEqual(exclude_from_log,
                                 cls._exclude_from_log)

        if register_name_scope is not None:
            self.assertEqual(register_name_scope,
                             cls._register_name_scope)
        else:
            self.assertEqual(cls.__name__,
                             cls._register_name_scope)

        if log_name_scope is not None:
            self.assertEqual(log_name_scope,
                             cls._log_name_scope)
        else:
            if register_name_scope is not None:
                self.assertEqual(register_name_scope,
                                 cls._log_name_scope)
            else:
                self.assertEqual(cls.__name__,
                                 cls._log_name_scope)
        if exclude_args_from_log is None:
            self.assertEmpty(cls._exclude_args_from_log)
        else:
            self.assertListEqual(exclude_args_from_log,
                                 cls._exclude_args_from_log)

        register_to_name_scope_fn_num_calls = 0
        if use_child or not exclude_from_register:
            register_to_name_scope_fn.assert_has_calls(
                [mock_call(cls._register_name_scope, cls,
                           name=cls._register_name)])
            register_to_name_scope_fn_num_calls += 1

        if use_child and not exclude_from_register:
            if use_child:
                cls_ = cls_parent
            else:
                cls_ = cls
            register_to_name_scope_fn.assert_has_calls(
                [mock_call(cls_._register_name_scope, cls_,
                           name=cls_._register_name)])
            register_to_name_scope_fn_num_calls += 1

        self.assertEqual(register_to_name_scope_fn_num_calls,
                         register_to_name_scope_fn.call_count)
        add_constructor_parameters_to_log_fn.assert_not_called()

    @patch("nucleus7.core.register.register_to_name_scope")
    @patch("nucleus7.core.config_logger.add_constructor_parameters_to_log")
    def test_call(self,
                  add_constructor_parameters_to_log_fn,
                  register_to_name_scope_fn,
                  exclude_from_log=False,
                  exclude_args_from_log=None):
        ClsWithMeta = self._declare_cls(
            register_name_="register_name",
            register_name_scope_="register_ns",
            log_name_scope_="log_ns",
            exclude_from_register_=False,
            exclude_from_log_=exclude_from_log,
            exclude_args_from_log_=exclude_args_from_log
        )
        obj = ClsWithMeta(10, 20, par3=30)

        self.assertEqual(1,
                         register_to_name_scope_fn.call_count)
        if exclude_from_log:
            add_constructor_parameters_to_log_fn.assert_not_called()
        else:
            add_constructor_parameters_to_log_fn.assert_called_once_with(
                obj, "log_ns", parameters_args=(10, 20),
                parameters_kwargs={"par3": 30},
                exclude_args=exclude_args_from_log or [])

        self.assertEqual(10,
                         obj.par1)
        self.assertEqual(20,
                         obj.par2)
        self.assertEqual(30,
                         obj.par3)

    def _declare_cls(self, register_name_=None, register_name_scope_=None,
                     log_name_scope_=None, exclude_args_from_log_=None,
                     exclude_from_register_=False, exclude_from_log_=False):
        class WithMeta(metaclass=base.MetaLogAndRegister):
            dummy_parameter = "dummy_parameter"
            register_name = register_name_
            register_name_scope = register_name_scope_
            log_name_scope = log_name_scope_
            exclude_from_register = exclude_from_register_
            exclude_from_log = exclude_from_log_
            exclude_args_from_log = exclude_args_from_log_

            def __init__(self, par1=100, par2=None, par3='par3'):
                self.par1 = par1
                self.par2 = par2
                self.par3 = par3

        return WithMeta


class TestBaseClass(parameterized.TestCase):
    def setUp(self):
        reset_register_and_logger()

    @parameterized.named_parameters(
        {"testcase_name": "with_parameter", "par1": 100},
        {"testcase_name": "without_parameter", "par1": None})
    def test_build(self, par1):
        dummy = _DummyBaseClass(par1=par1)
        self.assertFalse(dummy.built)
        dummy_after_build = dummy.build()
        self.assertTrue(dummy.built)
        self.assertIs(dummy,
                      dummy_after_build)

        if par1 is None:
            self.assertEqual("default",
                             dummy.par1)
        else:
            self.assertEqual(par1,
                             dummy.par1)
