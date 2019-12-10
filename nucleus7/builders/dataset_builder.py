# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Builders for dataset
"""
from typing import List
from typing import Union

from nucleus7.builders import data_builder_lib
from nucleus7.data.dataset import Dataset
from nucleus7.data.dataset import DatasetMix
from nucleus7.utils import deprecated


def build(
        dataset_config: Union[dict, List[dict]]) -> Union[Dataset, DatasetMix]:
    """
    Build dataset based on its config and file list if provided

    Parameters
    ----------
    dataset_config
        configuration of dataset or a list of dataset configurations;
        in last case, the DatasetMix will be generated out of all of datasets
        in the list; if you want to specify the
        sampling_weights for the mix, it should be set with
        `sampling_weight` inside of the each dataset config

    Returns
    -------
    dataset
        dataset
    """
    if isinstance(dataset_config, list):
        return _build_dataset_mix(dataset_config)
    return _build_single_dataset(dataset_config)


def _build_dataset_mix(datasets_configs: List[dict]) -> DatasetMix:
    """
    Build the dataset mix from

    Parameters
    ----------
    datasets_configs
        list of single dataset configs; if you want to specify the
        sampling_weights, it should be set with sampling_weight inside of the
        each dataset config

    Returns
    -------
    dataset_mix
        mix of datasets
    """
    datasets = []
    sampling_weights = []
    merge_on_same_file_list = []
    for each_dataset_config in datasets_configs:
        sampling_weights.append(each_dataset_config.pop("sampling_weight", 1.0))
        merge_on_same_file_list.append(each_dataset_config.pop(
            "merge_on_same_file_list", True))
        dataset = _build_single_dataset(each_dataset_config)
        datasets.append(dataset)
    dataset_mix = DatasetMix(datasets=datasets,
                             sampling_weights=sampling_weights,
                             merge_on_same_file_list=merge_on_same_file_list
                             ).build()
    return dataset_mix


def _build_single_dataset(dataset_config) -> Dataset:
    """
    Build single dataset based on its config

    Parameters
    ----------
    dataset_config
        dataset config

    Returns
    -------
    dataset
        dataset
    """
    deprecated.replace_deprecated_parameter_in_config(
        "name", "subtype", dataset_config, required=False)

    if ("data_pipe" in dataset_config
            and "class_name" not in dataset_config):
        build_fn = lambda x: Dataset.from_data_pipe(**x).build()
    else:
        build_fn = None
    dataset = data_builder_lib.build_data_object_from_config(
        config=dataset_config, base_cls=Dataset,
        built_fn=build_fn)
    return dataset
