# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import numpy as np
import tensorflow as tf

from nucleus7.utils import np_utils


class TestNpUtils(tf.test.TestCase):

    def test_stack_with_pad(self):
        arrays = []
        with self.assertRaises(ValueError):
            np_utils.stack_with_pad(arrays)

        arrays = [np.random.random([5, 5]),
                  np.random.random([5, 5]),
                  np.random.random([5, 5])]
        self.assertAllClose(np.stack(arrays, 0),
                            np_utils.stack_with_pad(arrays, 0))

        arrays = [np.random.random([5, 5]),
                  np.random.random([5]),
                  np.random.random([5, 5])]
        with self.assertRaises(ValueError):
            np_utils.stack_with_pad(arrays)

        arrays = [np.random.random([2, 3, 2]),
                  np.random.random([5, 1, 1]),
                  np.random.random([3, 4, 1])]
        arrays_padded_and_stacked = np_utils.stack_with_pad(arrays, 0)

        self.assertAllEqual(np.array([3, 5, 4, 2]),
                            arrays_padded_and_stacked.shape)

        arrays_padded = [
            np.pad(arrays[0], [[0, 3], [0, 1], [0, 0]], 'constant'),
            np.pad(arrays[1], [[0, 0], [0, 3], [0, 1]], 'constant'),
            np.pad(arrays[2], [[0, 2], [0, 0], [0, 1]], 'constant')]
        arrays_padded_and_stacked_must = np.stack(arrays_padded, 0)
        self.assertAllClose(arrays_padded_and_stacked_must,
                            arrays_padded_and_stacked)
