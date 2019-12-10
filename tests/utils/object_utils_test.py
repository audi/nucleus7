# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import unittest

import numpy as np

from nucleus7.utils import object_utils


class _Foo(object):
    def __init__(self, a=20):
        self.a = 1

    def method(self):
        self.a += 1


class _Bar(_Foo):
    def __init__(self, a=20):
        super(_Bar, self).__init__(a)


class TestObjectUtils(unittest.TestCase):

    def test_get_object_name(self):
        obj = np.float32
        output = object_utils.get_object_name(obj)

        self.assertTrue(isinstance(output, str))
        self.assertTrue('numpy.float32' in output)

    def test_classproperty(self):
        class Dummy(object):
            class_attribute = 10

            @object_utils.classproperty
            def get_class_attribute(self):
                return self.class_attribute

        self.assertEqual(10, Dummy.get_class_attribute)
        dummy1 = Dummy()
        dummy1.class_attribute = 100
        dummy2 = Dummy()
        self.assertEqual(100, dummy1.get_class_attribute)
        self.assertEqual(10, dummy2.get_class_attribute)

    def test_get_full_method_signature(self):
        def _f1(e, k, a, d=100, c=200, *args, b, l=10):
            pass

        def _f2(e, k, *, b):
            pass

        def _f3(e, k, b):
            pass

        def _f4(*, a, b=20):
            pass

        self.assertTupleEqual(
            (['e', 'k', 'a', 'd', 'c'], ['b', 'l'],
             {'d': 100, 'c': 200, 'l': 10}),
            object_utils.get_full_method_signature(method=_f1)
        )
        self.assertTupleEqual(
            (['e', 'k'], ['b'], {}),
            object_utils.get_full_method_signature(method=_f2)
        )
        self.assertTupleEqual(
            (['e', 'k', 'b'], [], {}),
            object_utils.get_full_method_signature(method=_f3)
        )
        self.assertTupleEqual(
            ([], ['a', 'b'], {'b': 20}),
            object_utils.get_full_method_signature(method=_f4)
        )

    def test_get_full_method_signature_with_inheritance(self):
        class _Bar(object):
            def __init__(self, a, b, c=200, d=None):
                pass

        class _Foo(_Bar):
            def __init__(self, a, e, *, b=20):
                pass

        class _Baz(_Foo):
            def __init__(self, o, *args, k=100, **kwargs):
                pass

        self.assertTupleEqual(
            (["a", "b", "c", "d"], [], {"d": None, "c": 200}),
            object_utils.get_full_method_signature(_Bar)
        )
        self.assertTupleEqual(
            (["a", "e"], ["b"], {"b": 20}),
            object_utils.get_full_method_signature(_Foo)
        )
        self.assertTupleEqual(
            (["o", "a", "e"], ["k", "b"], {"b": 20, "k": 100}),
            object_utils.get_full_method_signature(_Baz)
        )

    def test_get_parent_method(self):
        self.assertListEqual([object],
                             object_utils.get_parent_method(_Foo))
        self.assertListEqual(
            [], object_utils.get_parent_method(_Foo, _Foo.__init__))
        self.assertListEqual(
            [], object_utils.get_parent_method(_Foo, _Foo.method))

        self.assertListEqual([_Foo, object],
                             object_utils.get_parent_method(_Bar))
        self.assertListEqual(
            [_Foo.__init__],
            object_utils.get_parent_method(_Bar, _Bar.__init__))
        self.assertListEqual(
            [_Foo.method],
            object_utils.get_parent_method(_Bar, _Bar.method))
