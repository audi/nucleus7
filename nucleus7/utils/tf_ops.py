# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Tensorflow operations
"""

from typing import Dict
from typing import List

import numpy as np
import tensorflow as tf

from nucleus7.utils import nest_utils


def squash_dims(tensor: tf.Tensor, dims: list) -> tf.Tensor:
    """
    Squash, e.g. combine, multiple dimensions from tensor

    Parameters
    ----------
    tensor
        tensor to squash
    dims
        dimensions to squash

    Returns
    -------
    squashed_tensor
        tensor with squashed dimensions

    """
    tensor_shape = tf.shape(tensor)
    squash_dimension_size = tf.reduce_prod(
        tf.gather(tensor_shape,
                  np.arange(dims[0], dims[-1] + 1)), keep_dims=True)
    tensor_new_shape_list = [
        s for s in [tensor_shape[:dims[0]], squash_dimension_size,
                    tensor_shape[dims[1] + 1:]]
        if len(s.shape.dims) > 0]
    tensor_new_shape = tf.concat(tensor_new_shape_list, 0)

    tensor_reshaped = tf.reshape(tensor, tensor_new_shape)
    return tensor_reshaped


def dense_to_sparse(dense_tensor: tf.Tensor, shape: list) -> tf.SparseTensor:
    """
    Convert dense tensor to sparse one

    Parameters
    ----------
    dense_tensor
        dense tensor
    shape
        shape of tensor

    Returns
    -------
    sparse_tensor
        sparse tensor

    """
    sparse_indices = tf.where(tf.not_equal(dense_tensor, 0))
    sparse_tensor = tf.SparseTensor(
        sparse_indices, tf.gather_nd(dense_tensor, sparse_indices), shape)
    return sparse_tensor


def split_with_pad(tensor: tf.Tensor, num_splits: int, axis=0):
    """
    Split the value to list of num_splits
    and pad it with last value to make the value splittable by num_splits

    Parameters
    ----------
    tensor
        tensor to split
    num_splits
        number of splits
    axis
        axis to split

    Returns
    -------
    split_tensor
        tensor split with possible repeated last values

    """
    split_shape = tf.shape(tensor)[axis]

    def _pad_with_last(value):
        number_of_dimensions = len(value.get_shape().as_list())
        selection_idns = ([slice(None)] * axis + [slice(-1, None)] +
                          [slice(None)] * (number_of_dimensions - axis - 1))
        last_elem = value[selection_idns]
        multiples_axis = num_splits - tf.mod(split_shape, num_splits)
        multiples = ([1] * axis + [multiples_axis]
                     + [1] * (number_of_dimensions - axis - 1))
        v_pad = tf.tile(last_elem, multiples)
        v_new = tf.concat([tensor, v_pad], axis=axis)
        return v_new

    cond = tf.equal(tf.mod(split_shape, num_splits), 0)
    tensor = tf.cond(cond, lambda: tensor, lambda: _pad_with_last(tensor))
    return tf.split(tensor, num_splits, axis)


def replace_nan_with_zeros(tensor: tf.Tensor):
    """
    Replace NaN inside of tensor with zeroes

    Parameters
    ----------
    tensor
        tensor

    Returns
    -------
    tensor_without_nans
        tensor with replaced NaNs
    """
    tensor = tf.where(tf.is_nan(tensor), tf.zeros_like(tensor), tensor)
    return tf.where(tf.is_inf(tensor), tf.zeros_like(tensor), tensor)


def concat_padded(list_of_tensors: List[tf.Tensor], axis: int = 0,
                  expand_nonconcat_dim: bool = True) -> tf.Tensor:
    """
    Concatenate tensors and pad tensors with smaller dimension.
    Uses sparse concatenation inside, so can be slow

    Parameters
    ----------
    list_of_tensors
        list of tensors
    axis
        axis to concatenate
    expand_nonconcat_dim
        whether to allow the expansion in the non-concat dimensions.

    Returns
    -------
    concatenated_tensor
        concatenated tensor

    """
    t_sparse = [dense_to_sparse(t, tf.shape(t, out_type=tf.int64))
                for t in list_of_tensors]
    t_concatenated_sparse = tf.sparse_concat(
        axis, t_sparse, expand_nonconcat_dim=expand_nonconcat_dim)
    return tf.sparse_tensor_to_dense(t_concatenated_sparse)


def stack_padded(list_of_tensors: List[tf.Tensor], axis: int = 0,
                 expand_nonconcat_dim: bool = True) -> tf.Tensor:
    """
    Stack tensors and pad tensors with smaller dimension.
    Uses sparse concatenation inside, so can be slow

    Parameters
    ----------
    list_of_tensors
        list of tensors
    axis
        axis to concatenate
    expand_nonconcat_dim
        whether to allow the expansion in the non-concat dimensions.

    Returns
    -------
    stacked_tensor
        stacked tensor

    """
    list_of_tensors_expanded = [tf.expand_dims(t, axis=axis)
                                for t in list_of_tensors]
    return concat_padded(list_of_tensors_expanded, axis=axis,
                         expand_nonconcat_dim=expand_nonconcat_dim)


def concat_or_stack(list_of_tensors: List[tf.Tensor],
                    axis: int = 0) -> tf.Tensor:
    """
    If rank of tensor is 0, then stacks it otherwise concatenates them

    Parameters
    ----------
    list_of_tensors
        list of tensors
    axis
        axis to concatenate

    Returns
    -------
    new_tensor
        stacked or concatenated tensor
    """
    # TODO(oleksandr.vorobiov@audi.de): rewrite without exception catching
    #  - tf.cond is not working
    try:
        return tf.concat(list_of_tensors, axis)
    except ValueError:
        return tf.stack(list_of_tensors, axis)


def maybe_cast_dtype(inputs: Dict[str, tf.Tensor],
                     cast_dtypes: Dict[tf.DType, tf.DType] = None
                     ) -> Dict[str, tf.Tensor]:
    """
    Cast values from nested inputs structure according to cast_dtypes mapping.
    If dtype of value inside of inputs is not inside of cast_dtypes keys, it
    will not be casted at all.

    Parameters
    ----------
    inputs
        possibly nested dict, with values as tensors
    cast_dtypes
        dict with mapping of which dtype should be casted to which, e.g.
        {float32: float16} means that all of float32 tensors will be casted
        to float16 before passing to nucleotide

    Returns
    -------
    inputs_casted : dict
        same structure as inputs, but with inputs casted according to
        cast_dtypes
    """
    if cast_dtypes is None:
        return inputs
    inputs_flatten = nest_utils.flatten_nested_struct(inputs)
    for k, each_input in inputs_flatten.items():
        if not isinstance(each_input, tf.Tensor):
            continue
        dtype_input = each_input.dtype
        if dtype_input in cast_dtypes:
            input_casted = tf.cast(each_input, cast_dtypes[dtype_input])
            inputs_flatten[k] = input_casted
    inputs_casted = nest_utils.unflatten_dict_to_nested(inputs_flatten)
    return inputs_casted


def add_gradient_noise(gradient, noise_stddev: float = 1e-3,
                       name: str = "add_gradient_noise") -> tf.Tensor:
    """
    Add random normal noise to gradient

    Parameters
    ----------
    gradient
        initial gradient to add the noise
    noise_stddev
        standard deviation of the noise
    name
        variable scope name to use

    Returns
    -------
    gradient_with_noise
        gradient with noise

    References
    ----------
    http://arxiv.org/abs/1511.06807
    """
    # pylint: disable=not-context-manager
    # tf.op_scope is a context manager
    with tf.name_scope(name):
        noise = tf.random_normal(tf.shape(gradient), stddev=noise_stddev)
        gradient_with_noise = tf.add(gradient, noise, name=name)
    return gradient_with_noise
