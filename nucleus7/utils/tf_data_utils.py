# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Utils to work with tensorflow data
"""
import abc
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import tensorflow as tf

from nucleus7.utils import nest_utils
from nucleus7.utils import np_utils


class TfRecordsMixin:
    """
    Mixin for tfrecords processing
    """

    @abc.abstractmethod
    def get_tfrecords_features(self) -> dict:
        """
        Return dict, possibly nested, with feature names as keys and its
        serialized type as values of type :obj:`FixedLenFeature`.
        Keys should not have any '/', use nested dict instead.

        Returns
        -------
        features
            features inside of tfrecords file

        See Also
        --------
        output_types
            :func:`tf.decode_raw`
        """

    def get_tfrecords_output_types(self) -> Optional[dict]:
        """
        Return dict, possibly nested, with tensor names and tensor types.
        Keys should not have any '/', use nested dict instead.

        Should be same types which were used to store the tfrecord file

        If not implemented, assumes that the types are defined already inside
        of features, e.g. no decoding is needed

        Returns
        -------
        output_types
            output types of tfrecords features

        See Also
        --------
        output_types
            :func:`tf.parse_single_example`
        """
        # pylint: disable=no-self-use
        # is an interface
        return None

    def postprocess_tfrecords(self, **data) -> dict:
        """
        Postprocess tfrecords sample if needed.

        This function is applied after sample is decoded from tfrecords file

        Parameters
        ----------
        **data
            (nested) dict with raw tfrecords data

        Returns
        -------
        postprocessed_data
            postprocessed data
        """
        # pylint: disable=no-self-use
        # is an interface
        return data

    def decode_field(self,
                     field_name: str,
                     field_value: Union[tf.Tensor, tf.SparseTensor],
                     field_type: Optional[tf.DType] = None) -> tf.Tensor:
        """
        Decode a field from a tfrecord example

        Parameters
        ----------
        field_name
            name of the field, if nested - will be separated using "/"
        field_value
            value of the field from tfrecords example
        field_type
            type of the decoded field from self.get_tfrecords_output_types
            or None, if it was not provided
        """
        # pylint: disable=no-self-use
        # is designed to be overridden
        # pylint: disable=unused-argument
        # this method is really an interface, but has a default implementation.
        if field_type is None:
            return field_value
        return tf.decode_raw(field_value, field_type)

    def parse_tfrecord_example(self, example) -> dict:
        """Parse tfrecord example"""
        features_flat = nest_utils.flatten_nested_struct(
            self.get_tfrecords_features(), '/')
        output_types = self.get_tfrecords_output_types() or {}
        output_types_flat = nest_utils.flatten_nested_struct(
            output_types, '/')
        parsed_example = tf.parse_single_example(example, features_flat)
        data_decoded = {}
        for field_name, field_value in parsed_example.items():
            output_type = output_types_flat.get(field_name)
            data_decoded[field_name] = self.decode_field(
                field_name, field_value, output_type)
        data = nest_utils.unflatten_dict_to_nested(data_decoded, '/')
        data = self.postprocess_tfrecords(**data)
        return data


def _int32_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _ints32_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def nested_to_tfrecords_feature(nested_values: dict,
                                ignore_empty_arrays: bool = True) -> dict:
    """
    Create tf records features from nested dictionary structure

    Structure will be first flatten with separator '/' and then encoded
    using corresponding feature type.

    Parameters
    ----------
    nested_values
        nested dict holding data
    ignore_empty_arrays
        if the array is empty, e.g. it has 0 in the shape, it will not store
        it; tfrecords parser raises
        'Invalid argument: Key: {key}. Can't parse serialized Example' on
        empty arrays

    Returns
    -------
    features
        dict with features from flatten elements of nested_values
    """

    def _get_feature(value): # pylint: disable=too-many-return-statements
        if (ignore_empty_arrays
                and isinstance(value, np.ndarray)
                and value.size == 0):
            return None
        if (isinstance(value, np.ndarray)
                and all(isinstance(i, str) for i in value)):
            value = [val.encode('utf-8') for val in value]
            return _bytes_list_feature(value)
        if isinstance(value, list):
            value = np.array(value)
        if isinstance(value, np.ndarray):
            return _bytes_feature(value.tostring())
        if isinstance(value, int):
            return _int32_feature(value)
        if isinstance(value, str):
            return _bytes_feature(value.encode())
        if isinstance(value, float):
            return _float_feature(value)
        if isinstance(value, bytes):
            return _bytes_feature(value)
        raise ValueError('Value of type {} cannot be encoded!'.format(
            type(value)))

    flatten_values = nest_utils.flatten_nested_struct(
        nested_values, separator='/')
    features = {}
    for name, value in flatten_values.items():
        feature = _get_feature(value)
        if feature is None:
            continue
        features[name] = feature

    return features


def dense_to_sparse_feature(dense_feature: np.ndarray,
                            feature_name: str,
                            values_dtype=np.float32,
                            sparse_value_as_zeroes=0) -> dict:
    """
    Convert dense feature to sparse one

    Parameters
    ----------
    dense_feature
        dense feature
    feature_name
        name of the feature
    values_dtype
        dtype of values inside of sparse tensor
    sparse_value_as_zeroes
        value to use as a sparse

    Returns
    -------
    sparse_feature_dict
        dict holding indices, values and shape of resulted sparse tensor

    """
    spa = np_utils.dense_to_sparse(
        dense_feature, sparse_value_as_zeroes=sparse_value_as_zeroes)
    spa_inds = spa[0].astype(int)
    spa_values = np.array(spa[1]).astype(values_dtype)
    spa_shape = np.array(spa[2]).astype(int)
    res = {'_'.join([feature_name, 'indices']):
               _bytes_feature(spa_inds.tostring()),
           '_'.join([feature_name, 'values']):
               _bytes_feature(spa_values.tostring()),
           '_'.join([feature_name, 'shape']):
               _bytes_feature(spa_shape.tostring())}
    return res


def combine_features_from_list_of_dict_datasets(
        list_of_datasets: List[tf.data.Dataset]) -> tf.data.Dataset:
    """
    Combine the features from list_of_datasets to one dataset. Datasets will
    be resolved in the order they are in the list, so if the ds1 has feature
    with name 'key1' and ds2 has it, ds1['key1'] will be used

    Parameters
    ----------
    list_of_datasets
        list of tf.data.Dataset instances to be combined

    Returns
    -------
    dataset_with_all_features
        dataset with all features

    Raises
    ------
    ValueError
        if features with same key from different datasets have different dtypes
        or shapes
    """

    def combine_fn(*list_of_features) -> tf.data.Dataset:
        """
        Method to combine the features

        Parameters
        ----------
        list_of_features
            list of features to combine

        Returns
        -------
        data_with_combined_features
            data with combined features
        """
        features_combined_flatten = {}

        for each_features in list_of_features:
            each_features_flatten = nest_utils.flatten_nested_struct(
                each_features)
            for (each_feature_name,
                 each_feature) in each_features_flatten.items():
                if each_feature_name in features_combined_flatten:
                    _assert_tensors_have_same_shape(
                        features_combined_flatten[each_feature_name],
                        each_feature)
                    _assert_tensors_have_same_type(
                        features_combined_flatten[each_feature_name],
                        each_feature)
                else:
                    features_combined_flatten[each_feature_name] = each_feature

        features_combined = nest_utils.unflatten_dict_to_nested(
            features_combined_flatten)
        data_with_featured = tf.data.Dataset.from_tensors(features_combined)
        return data_with_featured

    datasets_zip = tf.data.Dataset.zip(tuple(list_of_datasets))
    dataset_combined = datasets_zip.flat_map(combine_fn)
    return dataset_combined


def _assert_tensors_have_same_shape(tensor1: tf.Tensor, tensor2: tf.Tensor):
    if tensor1.get_shape().as_list() != tensor2.get_shape().as_list():
        msg = "Shape of tensor {} is not compatible with tensor {}".format(
            tensor1, tensor2)
        raise ValueError(msg)


def _assert_tensors_have_same_type(tensor1: tf.Tensor, tensor2: tf.Tensor):
    if tensor1.dtype != tensor2.dtype:
        msg = "Type of tensor {} is not compatible with tensor {}".format(
            tensor1, tensor2)
        raise ValueError(msg)
