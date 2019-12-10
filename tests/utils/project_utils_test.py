# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import copy

from absl.testing import parameterized

from nucleus7.utils import project_utils


class TestProjectUtils(parameterized.TestCase):

    @parameterized.parameters({"base_index": -1},
                              {"base_index": 0},
                              {"base_index": 1},
                              {"base_index": 2})
    def test_update_config_with_other_config(self, base_index):
        list_base = "__BASE__"
        update_config = "__UPDATE_CONFIG__"
        config1 = {
            "config1": {"a": 10,
                        "b": {"b1": [1, 2], "b2": [1]},
                        "c": [1]},
            "config3": [{'a': 1, "b": 2}, {'d': 5}],
            "config4": {"c": 10},
            "config5": {"bar": 5},
            "config6": {"a": 5, "c": 1},
        }
        config2 = {
            "config1": {"a": 20, "b": None},
            "config2": [{"bar": 1, "foo": 20}],
            "config5": [list_base, {"foo": 5}],
            "config6": {update_config: True, "c": 20},
        }
        config3 = [{"d": 10}, {"e": -10}]
        if base_index < 0:
            config2["config3"] = config3
        elif base_index == 0:
            config2["config3"] = [list_base, *config3]
        elif base_index == 1:
            config2["config3"] = [config3[0], list_base, config3[1]]
        else:
            config2["config3"] = [config3[0], config3[1], list_base]

        config1_orig_copy = copy.deepcopy(config1)
        config2_orig_copy = copy.deepcopy(config2)
        config_updated_must = {
            "config1": {"a": 20, "b": None},
            "config2": [{"bar": 1, "foo": 20}],
            "config4": {"c": 10},
            "config5": [{"bar": 5}, {"foo": 5}],
            "config6": {"a": 5, "c": 20},
        }
        config13 = config1["config3"]
        if base_index < 0:
            config_updated_must["config3"] = config3
        elif base_index == 0:
            config_updated_must["config3"] = [*config13, *config3]
        elif base_index == 1:
            config_updated_must["config3"] = [config3[0], *config13, config3[1]]
        else:
            config_updated_must["config3"] = [config3[0], config3[1], *config13]

        config_updated = project_utils.update_config_with_other_config(
            config1, config2)

        self.assertDictEqual(config1_orig_copy,
                             config1)
        self.assertDictEqual(config2_orig_copy,
                             config2)
        self.assertDictEqual(config_updated_must,
                             config_updated)
