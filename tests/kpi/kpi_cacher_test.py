# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import numpy as np
import tensorflow as tf

from nucleus7.kpi.cacher import KPIMD5Cacher


class TestKPIMD5Cacher(tf.test.TestCase):

    def setUp(self):
        self._kpi_values = {
            "a": 100,
            "b": np.random.rand(2),
            "c": {"c1": 5,
                  "c2": np.random.rand(10, 5)},
            "d": [1, 2, 3]
        }
        self._kpi_values2 = {
            "a": 200,
            "b": 5
        }

    def test_cache_and_restore(self):
        cacher = KPIMD5Cacher().build()
        cacher.cache_target = self.get_temp_dir()
        cacher.calculate_hash(self._kpi_values)
        self.assertIsNone(cacher.restore())
        cacher.cache(self._kpi_values)

        self.assertAllClose(cacher.restore(),
                            self._kpi_values)

        cacher.calculate_hash(self._kpi_values)
        self.assertAllClose(cacher.restore(),
                            self._kpi_values)

        cacher.calculate_hash(self._kpi_values2)
        self.assertIsNone(cacher.restore())

        cacher.calculate_hash(self._kpi_values2, ["other", "prefix"])
        self.assertIsNone(cacher.restore())
        cacher.cache(self._kpi_values)
        self.assertAllClose(cacher.restore(),
                            self._kpi_values)
