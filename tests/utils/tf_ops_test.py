# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import tensorflow as tf

from nucleus7.utils import tf_ops


class TestTFOpsUtils(tf.test.TestCase):

    def test_concat_or_stack(self):
        self.assertAllClose(self.evaluate(
            tf_ops.concat_or_stack([1, 2], 0)), [1, 2])
        self.assertAllClose(self.evaluate(
            tf_ops.concat_or_stack([[1], [2]], 0)), [1, 2])
        self.assertAllClose(self.evaluate(
            tf_ops.concat_or_stack([[[1]], [[2]]], 0)), [[1], [2]])
