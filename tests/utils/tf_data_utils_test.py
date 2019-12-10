# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from unittest.mock import MagicMock
from unittest.mock import call as mock_call
from unittest.mock import patch

import tensorflow as tf

from nucleus7.utils import nest_utils
from nucleus7.utils import tf_data_utils


class TestTfRecordsMixin(tf.test.TestCase):

    def setUp(self):
        self.data = {"data1": 10,
                     "data2": 20,
                     "data3": [1, 5],
                     "data4": {"sub1": 100, "sub2": 200}}

    @patch.object(tf, "parse_single_example")
    @patch.object(tf, "decode_raw")
    def test_parse_tfrecord_example(self, tf_decode_raw,
                                    tf_parse_single_example):
        def _get_tfrecords_features():
            return {"data1": "feature_1",
                    "data2": "feature_data2",
                    "data3": ["feature_data3_0", "feature_data3_1"],
                    "data4": {"sub1": "feature_data4_sub1",
                              "sub2": "feature_data4_sub2"}}

        def _get_tfrecords_output_types():
            return {"data1": "string_value",
                    "data4/sub1": "float_value"}

        def _parse_single_example(example, features):
            example_flat = nest_utils.flatten_nested_struct(example, "/")
            result = {k: "-".join([str(example_flat[k]), features[k]])
                      for k in example_flat}
            return result

        def _postprocess_tfrecords(**data):
            data_flat = nest_utils.flatten_nested_struct(data)
            return nest_utils.unflatten_dict_to_nested(
                {k: v + "_pp" for k, v in data_flat.items()})

        tf_decode_raw.side_effect = lambda x, y: x + "_raw"
        tf_parse_single_example.side_effect = _parse_single_example

        mixin = tf_data_utils.TfRecordsMixin()
        mixin.get_tfrecords_features = MagicMock(wraps=_get_tfrecords_features)
        mixin.get_tfrecords_output_types = MagicMock(
            wraps=_get_tfrecords_output_types)
        mixin.postprocess_tfrecords = MagicMock(wraps=_postprocess_tfrecords)
        mixin.decode_field = MagicMock(wraps=mixin.decode_field)
        result = mixin.parse_tfrecord_example(self.data)

        features = _get_tfrecords_features()
        features_flat = nest_utils.flatten_nested_struct(features, "/")
        data_flat = nest_utils.flatten_nested_struct(self.data, "/")
        output_types_flat = nest_utils.flatten_nested_struct(
            _get_tfrecords_output_types(), "/")
        result_must = nest_utils.unflatten_dict_to_nested(
            {k: "-".join([str(data_flat[k]), features_flat[k]])
                + ("_raw_pp" if k in output_types_flat
                   else "_pp")
             for k in features_flat},
            "/")
        self.assertAllEqual(result_must,
                            result)

        mixin.get_tfrecords_features.assert_called_once_with()
        mixin.get_tfrecords_output_types.assert_called_once_with()

        combine_fn_before_decode = lambda x: "-".join([str(x[0]), x[1]])
        decode_values = nest_utils.flatten_nested_struct(
            nest_utils.combine_nested(
                [self.data, features], combine_fun=combine_fn_before_decode),
            "/")
        decode_field_calls = [mock_call(each_key, decode_values[each_key],
                                        output_types_flat.get(each_key))
                              for each_key in decode_values]
        mixin.decode_field.assert_has_calls(decode_field_calls, any_order=True)

        data_to_postprocess_must = nest_utils.unflatten_dict_to_nested(
            {k: "-".join([str(data_flat[k]), features_flat[k]])
                + ("_raw" if k in output_types_flat else "")
             for k in features_flat},
            "/")
        mixin.postprocess_tfrecords.assert_called_once_with(
            **data_to_postprocess_must)
