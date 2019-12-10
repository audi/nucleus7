# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from absl.testing import parameterized

from nucleus7 import model
from nucleus7.core import register
from nucleus7.test_utils.test_utils import reset_register_and_logger
from nucleus7.utils.warnings import RegisterWarning


class TestRegister(parameterized.TestCase):
    def setUp(self):
        reset_register_and_logger()

    def test_register_and_retrieve(self):
        class DummyPlugin(model.ModelPlugin):
            register_name = "dummy"
            pass

        class DummyLoss(model.ModelLoss):
            pass

        with self.assertWarns(RegisterWarning):
            class DummyPlugin(model.ModelPlugin):
                register_name = "dummy"
                pass

        self.assertEqual(DummyPlugin,
                         register.retrieve_from_register('dummy'))
        self.assertEqual(DummyLoss,
                         register.retrieve_from_register('DummyLoss'))
        with self.assertRaises(ValueError):
            register.retrieve_from_register('dummy2')

        with self.assertRaises(ValueError):
            register.retrieve_from_register('DummyLoss', DummyPlugin)

    @parameterized.named_parameters(
        {"testcase_name": "with_name", "with_name": True},
        {"testcase_name": "without_name", "with_name": False})
    def test_register_to_name_scope_and_remove(self, with_name):
        class _Dummy(object):
            pass

        register.reset_register()
        name = with_name and "dummy" or None
        register.register_to_name_scope("dummy_name_scope", cls=_Dummy,
                                        name=name)
        retrieve_name = with_name and "dummy" or _Dummy.__name__
        self.assertIs(_Dummy,
                      register.retrieve_from_register(retrieve_name))
        remove_cls_or_name = with_name and "dummy" or _Dummy
        register.remove_from_name_scope("dummy_name_scope", remove_cls_or_name)

        self.assertEmpty(register.get_register())

    def test_reset_register(self):
        class _Dummy(object):
            pass

        register.register_to_name_scope("dummy_name_scope", cls=_Dummy,
                                        name="dummy")
        self.assertNotEmpty(register.get_register())
        register.reset_register()
        self.assertEmpty(register.get_register())
