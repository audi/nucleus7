# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Interfaces for datasets
"""
import abc
import glob
import logging
import os
import random
import tempfile
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import tensorflow as tf

from nucleus7.core import project_artifacts
from nucleus7.core.dna_helix import DNAHelix
from nucleus7.core.nucleotide import Nucleotide
from nucleus7.data.data_filter import DataFilterMixin
from nucleus7.data.data_pipe import DataPipe
from nucleus7.data.data_pipe import DataPipeMixin
from nucleus7.data.file_list import FileList
from nucleus7.data.file_list import FileListMixin
from nucleus7.utils import io_utils
from nucleus7.utils import model_utils
from nucleus7.utils import nest_utils
from nucleus7.utils import object_utils
from nucleus7.utils import tf_data_utils
from nucleus7.utils import tf_varscopes_utils

_TFRECORDS_DATA_KEY = 'data'


# pylint: disable=too-many-instance-attributes
# is needed to make the Dataset more generic
class Dataset(Nucleotide,
              DataFilterMixin,
              model_utils.CustomSessionHandlerMixin):
    """
    General Dataset class to be used for the training

    Parameters
    ----------
    cache_dir
        directory name to cache the data; file name with name of dataset and
        some random suffix will be generated there; this cache can be removed
        by calling datasets.clear_cache()
    num_parallel_calls
        used in dataset map function; see tf.data.Dataset.map
    random_seed
        sets the random seed, e.g. for shuffling
    shuffle_buffer_size
        number of samples to shuffle in the buffer; shuffling performed after
        feature extraction, so do not set it to very high number; if you want to
        disable shuffle, set shuffle_buffer_size = 0
    prefetch_buffer_size
        number of batches to prefetch; if you want to disable prefetching, set
        prefetch_buffer_size = 0
    fix_batch_dimension
        if the batch dimension should be hard coded, e.g. could be possible to
        use tf.unstack(batch) directly
    number_of_shards
        defines the number of shards of dataset; is good choice to use for
        distributed learning
    shard_index
        defines the shard index of dataset; is good choice to use for
        distributed learning
    subtype
        additional name and type to add to the dataset, e.g. train / eval;
        will be added to cache file name

    Attributes
    ----------
    register_name_scope
        name scope for register and for the config logger
    _cache_file : str
        name of cache file

    Raises
    ------
    ValueError
        if invalid combination for shard_index and number_of_shards is provided
    ValueError
        if shuffle_buffer_size < 0
    ValueError
        if prefetch_buffer_size < 0
    FileNotFoundError
        if directory before cache_dir does not exist

    """
    register_name_scope = "dataset"
    exclude_from_register = True

    def __init__(self, *,
                 num_parallel_calls: int = 16,
                 random_seed: int = 7567,
                 shuffle_buffer_size: int = 100,
                 prefetch_buffer_size: int = 10,
                 cache_dir: Optional[str] = None,
                 fix_batch_dimension: bool = False,
                 subtype: Optional[str] = None,
                 number_of_shards: int = 1,
                 shard_index: int = 0):
        if number_of_shards < 1 or shard_index >= number_of_shards:
            msg = ("Impossible combination of shard_index ({}) "
                   "and number_of_shards({})".format(shard_index,
                                                     number_of_shards))
            raise ValueError(msg)
        if shuffle_buffer_size < 0:
            raise ValueError("shuffle_buffer_size must be >= 0")
        if prefetch_buffer_size < 0:
            raise ValueError("prefetch_buffer_size must be >= 0")
        if cache_dir:
            cache_root_dir = os.path.split(cache_dir)[0]
            if not os.path.isdir(cache_root_dir):
                raise FileNotFoundError(
                    "root directory of cache directory {} does not "
                    "exist!".format(cache_dir))

        super(Dataset, self).__init__([], name="dataset")
        self.num_parallel_calls = num_parallel_calls
        self.random_seed = random_seed
        self.shuffle_buffer_size = shuffle_buffer_size
        self.prefetch_buffer_size = prefetch_buffer_size
        self.cache_dir = cache_dir
        self.fix_batch_dimension = fix_batch_dimension
        self.number_of_shards = number_of_shards
        self.shard_index = shard_index
        self._subtype = subtype or ""
        self._cache_file = None  # type: str

    @property
    def subtype(self) -> str:
        """
        Subtype of the dataset

        Returns
        -------
        subtype
            subtype
        """
        return self._subtype

    @abc.abstractmethod
    def create_initial_data(self) -> tf.data.Dataset:
        """
        Create the tf.data.Dataset

        Returns
        -------
        data
            element data
        """

    def extract_features_from_initial_data(
            self, data: tf.data.Dataset) -> tf.data.Dataset:
        """
        Extract features from data, e.g. read from file names

        Parameters
        ----------
        data
            data

        Returns
        -------
        data
            initial data
        """
        # pylint: disable=no-self-use
        # is an interface
        return data

    @tf_varscopes_utils.with_name_scope("Dataset")
    def __call__(self, batch_size: int) -> tf.data.Dataset:
        # pylint: disable=arguments-differ
        # base class has more generic signature
        return self.create_batch(batch_size=batch_size)

    @object_utils.assert_is_built
    @object_utils.assert_property_is_defined('mode')
    def create_batch(self, batch_size: int) -> tf.data.Dataset:
        """
        Generate the batch of data samples

        Parameters
        ----------
        batch_size
            batch size

        Returns
        -------
        data_batch
            data
        """
        data_sample = self.create_features_for_single_sample()
        data_batch = self.combine_samples_to_batch(data_sample, batch_size)
        if self.prefetch_buffer_size > 0:
            data_batch = self.prefetch(data_batch)
        return data_batch

    @object_utils.assert_is_built
    @object_utils.assert_property_is_defined('mode')
    def create_features_for_single_sample(self) -> tf.data.Dataset:
        """
        Create the sample tensorflow dataset and extract features from it

        Returns
        -------
        data_with_features
            data with features
        """
        data = self.create_initial_data()
        if self.number_of_shards > 1:
            data = self.shard(data)
        data = self.extract_features_from_initial_data(data)
        if self.cache_dir is not None:
            data = self.cache(data)
        if self.is_training:
            data = self.shuffle(data)
        data = self.repeat(data)
        if self.data_filters:
            data = self.filter_sample(data)
        return data

    @object_utils.assert_is_built
    def filter_sample(self, data: tf.data.Dataset) -> tf.data.Dataset:
        """
        Filter according to data filters

        Parameters
        ----------
        data
            input dataset

        Returns
        -------
        data
            filtered dataset
        """

        def _predicate_fn(inputs):
            return self.data_filter_true(**inputs)

        data = data.filter(_predicate_fn)
        return data

    def shard(self, data: tf.data.Dataset) -> tf.data.Dataset:
        """
        Shard the data

        Parameters
        ----------
        data
            data

        Returns
        -------
        data_sharded
            shard of the data
        """
        data = data.shard(self.number_of_shards, self.shard_index)
        return data

    def shuffle(self, data: tf.data.Dataset) -> tf.data.Dataset:
        """
        Shuffle the dataset

        Parameters
        ----------
        data
            data

        Returns
        -------
        data_shuffled
            shuffled data
        """
        data = data.shuffle(self.shuffle_buffer_size,
                            seed=self.random_seed)
        return data

    @object_utils.assert_is_built
    @object_utils.assert_property_is_defined('mode')
    def cache(self, data: tf.data.Dataset) -> tf.data.Dataset:
        """
        Cache the dataset

        Parameters
        ----------
        data
            tensorflow dataset to cache

        Returns
        -------
        data_cached
            cached data
        """
        self._cache_file = _get_cache_fname(
            self.cache_dir, "-".join(
                [self.__class__.__name__, self._subtype, str(self.mode)]))
        data = data.cache(self._cache_file)
        return data

    @staticmethod
    def repeat(data: tf.data.Dataset) -> tf.data.Dataset:
        """
        Repeat dataset

        Parameters
        ----------
        data
            tensorflow dataset to cache

        Returns
        -------
        data_repeated
            repeated data
        """
        data = data.repeat()
        return data

    def prefetch(self, data: tf.data.Dataset) -> tf.data.Dataset:
        """
        Prefetch the data to memory

        Parameters
        ----------
        data
            data

        Returns
        -------
        data
            data after prefetching
        """
        data = data.prefetch(self.prefetch_buffer_size)
        return data

    def clear_cache(self):
        """
        Clear the cache of dataset if it exists
        """
        if self._cache_file is not None:
            try:
                logger = logging.getLogger(__name__)
                logger.info("Remove cache file %s", self._cache_file)
                cache_files = glob.glob(self._cache_file + '*')
                for each_cache_file in cache_files:
                    os.remove(each_cache_file)
            except FileNotFoundError:
                pass

    def combine_samples_to_batch(
            self, data: tf.data.Dataset, batch_size: int) -> tf.data.Dataset:
        """
        Apply batching on data

        Parameters
        ----------
        data
            data
        batch_size
            batch size

        Returns
        -------
        data_batch
            batched data
        """
        data = data.padded_batch(batch_size, data.output_shapes,
                                 drop_remainder=self.fix_batch_dimension)
        return data

    @staticmethod
    def from_data_pipe(
            data_pipe: DataPipe, file_list: Optional[FileList] = None,
            **dataset_kwargs
    ) -> Union['DatasetFromPipe',
               'DatasetFileListFromPipe',
               'DatasetTfRecordsFromPipe']:
        """
        Construct Dataset out of the pipe

        Parameters
        ----------
        data_pipe
            data pipe to use
        file_list
            file list to use; if provided and data_pipe is a tfrecords
            data pipe, it will result to DatasetTfRecords and if it is not a
            tfrecords pipe, it will be DatasetFileList;
            otherwise it will be Dataset
        dataset_kwargs
            additional kwargs to pass to corresponding Dataset constructor

        Returns
        -------
        dataset
            dataset with data pipe inside
        """
        if file_list is None:
            return DatasetFromPipe(data_pipe=data_pipe, **dataset_kwargs)
        if data_pipe.read_from_tfrecords:
            return DatasetTfRecordsFromPipe(
                data_pipe=data_pipe, file_list=file_list, **dataset_kwargs)
        return DatasetFileListFromPipe(
            data_pipe=data_pipe, file_list=file_list, **dataset_kwargs)

    def __del__(self):
        self.clear_cache()

    def __exit__(self, exc_type, exc_value, traceback):
        self.clear_cache()


class DatasetFileList(Dataset, FileListMixin):
    """
    Dataset to read the files from file_list

    Parameters
    ----------
    file_list
        built FileList instance
    initial_shuffle
        if the file names should be shuffled in the beginning
    file_list_keys_mapping
        mapping for file lists keys in the form
        {file_list_key: new_file_list_key}

    Attributes
    ----------
    exclude_args_from_log
        fields that will not be included to the config logger
    """
    exclude_args_from_log = ["file_list"]
    exclude_from_register = True

    def __init__(self, file_list: FileList,
                 initial_shuffle: bool = False,
                 file_list_keys_mapping: Optional[Dict[str, str]] = None,
                 **dataset_kwargs):
        super().__init__(**dataset_kwargs)
        self.file_list = file_list
        self.file_list_keys_mapping = file_list_keys_mapping
        self.initial_shuffle = initial_shuffle

    @abc.abstractmethod
    def read_raw_data_from_file(self, **sample_fnames) -> dict:
        """
        Abstract method to reed the raw data using :obj:`tf.data.Dataset`

        Parameters
        ----------
        sample_fnames
            sample with file names to read the data from

        Returns
        -------
        data
            data read from sample_fnames
        """

    @project_artifacts.add_project_artifact(
        "file_list", "file_list.file_names_pairs")
    def build(self) -> "DatasetFileList":
        """
        shuffle the file list if needed and check if file list keys are
        supported by the dataset

        Returns
        -------
        self
            dataset for chaining

        Raises
        ------
        ValueError
            if the dataset file list keys are not inside of file_list keys
        """
        self.file_list = self.file_list.filter_by_keys(
            self.file_list_keys_required, self.file_list_keys_optional,
            self.file_list_keys_mapping)
        if self.initial_shuffle:
            random.seed(self.random_seed)
            random.shuffle(self.file_list)
        return super().build()

    def create_initial_data(self):
        data = tf.data.Dataset.from_tensor_slices(self.file_list.get())
        return data

    def extract_features_from_initial_data(self, data: tf.data.Dataset):
        data = data.map(self._read_raw_data_from_file,
                        num_parallel_calls=self.num_parallel_calls)
        return data

    def _read_raw_data_from_file(self, sample_fnames: dict):
        return self.read_raw_data_from_file(**sample_fnames)


# pylint: disable=abstract-method
# this uses abstract mixin, and is interface
class DatasetTfRecords(DatasetFileList, tf_data_utils.TfRecordsMixin):
    """
    Dataset to read the samples from tfrecord files

    Parameters
    ----------
    file_list
        FileList to use, must contain only 1 single key - 'data'
    compression_type
        specify only if it is used while saving the tfrecord files
        Possible options {'GZIP', 'ZLIB', None}
    num_parallel_reads
        num_parallel_reads for tf.data.TFRecordDataset
    """
    file_list_keys = [_TFRECORDS_DATA_KEY]
    exclude_from_register = True

    def __init__(self,
                 file_list: FileList,
                 num_parallel_reads=None,
                 compression_type=None,
                 **dataset_kwargs):
        if compression_type:
            assert compression_type in ['GZIP', 'ZLIB'], (
                "Compression type must be in {}, received {}!!!".format(
                    ['GZIP', 'ZLIB'], compression_type))
        self.compression_type = compression_type
        self.num_parallel_reads = num_parallel_reads
        super().__init__(file_list=file_list, **dataset_kwargs)
        assert len(self.file_list_keys) == 1, (
            "Length of file_list_keys inside of {} class must be 1!".format(
                self.name)
        )

    def create_initial_data(self) -> tf.data.Dataset:
        data = tf.data.TFRecordDataset(
            self.file_list.get()[self.file_list_keys[0]],
            compression_type=self.compression_type,
            num_parallel_reads=self.num_parallel_reads)
        return data

    def extract_features_from_initial_data(self, data) -> tf.data.Dataset:
        """
        Parse the tfrecord example and shuffle data if needed
        """
        data = data.map(self.parse_tfrecord_example,
                        num_parallel_calls=self.num_parallel_calls)
        return data

    def read_raw_data_from_file(self, **sample_fnames) -> dict:
        # this abstract method is not used in this interface
        pass


# pylint: enable=abstract-method

class DatasetMix(Dataset):
    """
    Dataset that combines multiple datasets and samples one of them inside of
    the batch.

    To be able to combine datasets inside of a batch, it will create zero
    features for all features inside of all datasets and add them to each subset
    together with the key sample_mask_{dataset.subtype}, which indicates which
    subset is sampled.

    Parameters
    ----------
    datasets
        datasets for the mix
    sampling_weights
        fractions of datasets for sampling during training; should have same
        length as datasets; if not provided, subsets will be sampled uniformly;
        in case of multiple datasets, that needs to be combined according to
        file_list, sampling_weight will be the mean of all combined sampling
        weights
    merge_on_same_file_list
        flags if the dataset should be merged with other datasets if they have
        the same file lists; if not set, it will merge datasets with the same
        file lists and use sampling_weight as a sum of sampling weights and
        results of the samples will be combined in following manner:
        [{a: 1, b: 2, c: 3}, {a: 4, b: 5, e: 6, f: 7}] will result in
        {a: 1, b: 2, c: 3, e: 6, f: 7} and then this dataset will be mixed
        as usual with the rest of datasets; subtype of the combined dataset
        will be then subtype of first dataset

    Attributes
    ----------
    exclude_args_from_log
        fields that will not be included to the config logger

    See Also
    --------
    :func:`tf.contrib.data.sample_from_datasets`
        for the sampling strategy during training
    :func:`tf.contrib.data.choose_from_datasets`
        for the sampling strategy during evaluation

    """
    exclude_from_register = True
    exclude_args_from_log = ["datasets"]

    def __init__(self,
                 datasets: List[Dataset],
                 sampling_weights: Optional[list] = None,
                 merge_on_same_file_list: Optional[list] = None):
        self.datasets = datasets
        self.sampling_weights = sampling_weights
        self.merge_on_same_file_list = (merge_on_same_file_list
                                        or [True] * len(datasets))
        random_seed = self.datasets[0].random_seed
        fix_batch_dimension = self.datasets[0].fix_batch_dimension
        super().__init__(random_seed=random_seed,
                         fix_batch_dimension=fix_batch_dimension)
        self.dynamic_generated_keys = any(each_dataset.dynamic_generated_keys
                                          for each_dataset in self.datasets)

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, mode: str):
        self._mode = mode
        for each_dataset in self.datasets:
            each_dataset.mode = mode

    # pylint: disable=no-self-argument
    # classproperty is class property and so cls must be used
    @object_utils.classproperty
    def generated_keys_required(cls):
        if not hasattr(cls, "datasets"):
            return []

        required_keys_from_datasets = _combine_keys(
            [each_dataset.generated_keys_required
             for each_dataset in cls.datasets])
        return required_keys_from_datasets

    # pylint: disable=no-self-argument
    # classproperty is class property and so cls must be used
    @object_utils.classproperty
    def generated_keys_optional(cls):
        if not hasattr(cls, "datasets"):
            return []

        optional_keys_from_datasets = _combine_keys(
            [each_dataset.generated_keys_optional
             for each_dataset in cls.datasets])
        sample_mask_keys = ['sample_mask_{}'.format(each_dataset.subtype)
                            for each_dataset in cls.datasets]
        return optional_keys_from_datasets + sample_mask_keys

    @property
    def dna_helix(self) -> Optional[DNAHelix]:
        """
        Returns
        -------
        dna_helix as a sum from components which have it defined
        """
        dna_helix = None
        for each_dataset in self.datasets:
            if getattr(each_dataset, "dna_helix", None) is not None:
                if dna_helix is None:
                    dna_helix = each_dataset.dna_helix
                else:
                    dna_helix = dna_helix + each_dataset.dna_helix
        return dna_helix

    def build(self) -> "DatasetMix":
        """
        Perform the checks on datasets, e.g. if the subtype is set or if the
        datasets have compatible padded_shapes

        Returns
        -------
        self
            self for chaining

        Raises
        ------
        ValueError
            if some checks not passed
        """
        self._check_datasets_for_compatibility()
        super().build()
        return self

    def create_features_for_single_sample(self) -> tf.data.Dataset:
        """
        Extract features for each subset separately, add the zero features
        together with sample_masks and then sample the subset

        Returns
        -------
        features_combined
            features combined from all datasets

        """
        features_datasets = [each_dataset.create_features_for_single_sample()
                             for each_dataset in self.datasets]

        if not any(self.merge_on_same_file_list):
            dataset_names = [
                each_dataset.subtype for each_dataset in self.datasets]
            sampling_weights = self.sampling_weights
        else:
            features_datasets, dataset_names, sampling_weights = (
                _combine_features_from_datasets_with_same_filelist(
                    self.datasets, features_datasets,
                    self.merge_on_same_file_list, self.sampling_weights))

        features_combined = _add_sample_mask_and_zero_features(
            features_datasets, dataset_names)
        features_sampled = _sample_dataset_from_list_of_datasets(
            features_combined, sampling_weights, self.is_training,
            self.random_seed)
        return features_sampled

    def create_initial_data(self):
        raise NotImplementedError(
            "Do not use create_initial_data inside of Mix")

    def _check_datasets_for_compatibility(self):
        """
        Check the compatibility of self.datasets

        Raises
        ------
        ValueError
            if only 1 dataset was provided
        ValueError
            it not all datasets are built and fix_batch_dimension has same value
            for all datasets
        ValueError
            if datasets have not unique subtype names
        ValueError
            if padded shapes for same keys are not equal
        """
        self._check_number_of_datasets()
        self._check_if_all_datasets_are_built()
        self._check_unique_subtype_names()

    def _check_number_of_datasets(self):
        """
        Check number of datasets is greater than 1

        Raises
        ------
        ValueError
            if only 1 dataset was provided
        """
        if len(self.datasets) <= 1:
            msg = "Only 1 dataset provided. There is nothing to mix with!"
            raise ValueError(msg)

    def _check_if_all_datasets_are_built(self):
        """
        Check if all datasets are built and fix_batch_dimension has same value

        Raises
        ------
        ValueError
            it not all datasets are built
        """
        fix_batch_values = []
        for each_dataset in self.datasets:
            object_utils.assert_object_is_built(each_dataset)
            fix_batch_values.append(each_dataset.fix_batch_dimension)
        if len(set(fix_batch_values)) > 1:
            msg = "Use same value for fix_batch_dimension for all datasets!"
            raise ValueError(msg)

    def _check_unique_subtype_names(self):
        """
        Check if all datasets have unique subtypes

        Raises
        ------
        ValueError
            if datasets have not unique subtype names
        """
        subtypes = set()
        for each_dataset in self.datasets:
            dataset_subtype = each_dataset.subtype
            if dataset_subtype in subtypes:
                msg = ("Subtype {} repeats! Please use unique subtypes (names) "
                       "for each subset!").format(dataset_subtype)
                raise ValueError(msg)
            subtypes.add(dataset_subtype)


class DatasetFromPipe(DataPipeMixin, Dataset):
    """
    Dataset that uses data_pipe to generate data

    Parameters
    ----------
    data_pipe
        data pipe object to use
    output_keys_mapping
        keys mapping to map the data_pipe nucleotide results to dataset
        generated keys
    """
    exclude_from_register = True
    exclude_args_from_log = ["data_pipe"]

    def __init__(self, data_pipe: DataPipe,
                 output_keys_mapping: Optional[dict] = None,
                 **dataset_kwargs):
        DataPipeMixin.__init__(self, data_pipe=data_pipe,
                               output_keys_mapping=output_keys_mapping)
        Dataset.__init__(self, **dataset_kwargs)

    def build(self):
        super().build()
        _check_data_pipe_compatibility(self, self.data_pipe)
        self.data_pipe.build_dna()
        return self

    def create_initial_data(self) -> tf.data.Dataset:
        return tf.data.Dataset.from_tensors(self.process_sample_with_pipe())


# pylint: disable=abstract-method
# this class uses more high level API instead of low level abstract methods
class DatasetFileListFromPipe(DataPipeMixin, DatasetFileList):
    """
    Dataset that uses data_pipe to generate data and file list

    Parameters
    ----------
    data_pipe
        data pipe object to use
    output_keys_mapping
        keys mapping to map the data_pipe nucleotide results to dataset
        generated keys
    """
    exclude_from_register = True
    exclude_args_from_log = ["data_pipe", "file_list"]

    def __init__(self, data_pipe: DataPipe,
                 output_keys_mapping: Optional[dict] = None,
                 **dataset_kwargs):
        if "file_list_keys_mapping" in dataset_kwargs:
            raise ValueError("file_list_keys_mapping should be provided inside "
                             "of DataReaders for DataPipe datasets!")
        DataPipeMixin.__init__(self, data_pipe=data_pipe,
                               output_keys_mapping=output_keys_mapping)
        DatasetFileList.__init__(self, **dataset_kwargs)
        self.file_list_keys = self.file_list.keys()

    def build(self):
        super().build()
        _check_data_pipe_compatibility(self, self.data_pipe)
        self.data_pipe.build_dna()
        return self

    def create_initial_data(self) -> tf.data.Dataset:
        data = tf.data.Dataset.from_tensor_slices(self.file_list.get())
        return data

    def extract_features_from_initial_data(self, data: tf.data.Dataset):
        data = data.map(lambda x: self.process_sample_with_pipe(**x),
                        num_parallel_calls=self.num_parallel_calls)
        return data


# pylint: enable=abstract-method

# pylint: disable=abstract-method
# this class uses more high level API instead of low level abstract methods
class DatasetTfRecordsFromPipe(DataPipeMixin, DatasetTfRecords):
    """
    Dataset that uses data_pipe to generate data from tf record files

    Parameters
    ----------
    data_pipe
        data pipe object to use
    output_keys_mapping
        keys mapping to map the data_pipe nucleotide results to dataset
        generated keys
    compression_type
        specify only if it is used while saving the tfrecord files
        Possible options {'GZIP', 'ZLIB', None}
    num_parallel_reads
        num_parallel_reads for tf.data.TFRecordDataset
    """
    exclude_from_register = True
    exclude_args_from_log = ["data_pipe", "file_list"]
    file_list_keys = ["tfrecords"]

    def __init__(self, data_pipe: DataPipe,
                 output_keys_mapping: Optional[dict] = None,
                 compression_type: Optional[str] = None,
                 num_parallel_reads: int = 16,
                 **dataset_kwargs):
        if "file_list_keys_mapping" in dataset_kwargs:
            raise ValueError("file_list_keys_mapping should be provided inside "
                             "of DataReaders for DataPipe datasets!")
        DataPipeMixin.__init__(self, data_pipe=data_pipe,
                               output_keys_mapping=output_keys_mapping)
        DatasetTfRecords.__init__(
            self,
            compression_type=compression_type,
            num_parallel_reads=num_parallel_reads,
            **dataset_kwargs)
        self.file_list_keys = self.file_list.keys()

    def build(self):
        super().build()
        _check_data_pipe_compatibility(self, self.data_pipe)
        self.data_pipe.build_dna()
        return self

    def create_initial_data(self) -> tf.data.Dataset:
        data = tf.data.TFRecordDataset(
            self.file_list.get()[self.file_list_keys[0]],
            compression_type=self.compression_type,
            num_parallel_reads=self.num_parallel_reads)
        return data

    def extract_features_from_initial_data(self, data: tf.data.Dataset):
        data = data.map(lambda x: self.process_sample_with_pipe(tfrecords=x),
                        num_parallel_calls=self.num_parallel_calls)
        return data


# pylint: enable=abstract-method

def _check_data_pipe_compatibility(dataset: Dataset, data_pipe: DataPipe):
    if not data_pipe.is_tensorflow_pipe:
        raise ValueError("Currently data pipes for use as dataset must "
                         "consist of tensorflow data elements only!")

    if isinstance(dataset, DatasetFileList) and not data_pipe.readers:
        raise ValueError("Provide DataReaders to read from file list!")
    if not isinstance(dataset, DatasetFileList) and data_pipe.readers:
        raise ValueError("Provide FileList to use DataReaders")

    if (isinstance(dataset, DatasetTfRecords)
            and not data_pipe.read_from_tfrecords):
        raise ValueError(
            "Provide TfRecords readers to read from tf records files!")


def _add_sample_mask_and_zero_features(
        features_datasets: List[tf.data.Dataset],
        dataset_names: List[str]
) -> List[tf.data.Dataset]:
    """
    Add zero features and loss masks to each dataset

    Parameters
    ----------
    features_datasets
        lists of tensorflow datasets with features inside
    dataset_names
        list of dataset names

    Returns
    -------
    data_with_loss_mask_and_zero_features
        same datasets as inputs but with sample_mask_{subtypes} keys and
        with all keys from other datasets with zeroes of same type

    """
    zero_features = _create_zero_features_for_each_subset(
        features_datasets)
    sample_masks = _create_sample_masks_for_each_subset(dataset_names)
    data_with_sample_mask_and_zero_features = []
    for each_subset, each_sample_mask in zip(features_datasets,
                                             sample_masks):
        subset_with_sample_mask_and_zero_features = (
            tf_data_utils.combine_features_from_list_of_dict_datasets(
                [each_subset, zero_features, each_sample_mask]))
        data_with_sample_mask_and_zero_features.append(
            subset_with_sample_mask_and_zero_features)
    return data_with_sample_mask_and_zero_features


def _sample_dataset_from_list_of_datasets(
        list_of_data: List[tf.data.Dataset],
        sampling_weights: List[float],
        is_training: bool,
        random_seed: Optional[int] = None
) -> tf.data.Dataset:
    """
    Sample (select) one dataset to use as a features

    Parameters
    ----------
    list_of_data
        list of tensorflow datasets with zero and sample_mask values
    sampling_weights
        list of sampling weights
    is_training
        flag indicating if it is a training mode
    random_seed
        random seed to use to sample datasets

    Returns
    -------
    data
        sampled dataset
    """
    number_of_subsets = len(list_of_data)
    if is_training:
        data = tf.contrib.data.sample_from_datasets(
            list_of_data, sampling_weights, random_seed)
    else:
        choice_dataset = tf.data.Dataset.range(number_of_subsets).repeat()
        data = tf.contrib.data.choose_from_datasets(
            list_of_data, choice_dataset)
    return data


def _create_sample_masks_for_each_subset(dataset_names
                                         ) -> List[tf.data.Dataset]:
    """
    Create sample_masks_{subtype} datasets for each subtype

    Parameters
    ----------
    dataset_names
        names (subtypes) of datasets

    Returns
    -------
    sample_mask_datasets
        datasets with sample_mask_{subtype} keys
    """
    sample_mask_subsets = []
    for each_subset_name in dataset_names:
        sample_mask = _get_sample_mask(each_subset_name, dataset_names)
        sample_mask_data_subset = tf.data.Dataset.from_tensors(
            sample_mask).repeat()
        sample_mask_subsets.append(sample_mask_data_subset)
    return sample_mask_subsets


def _create_zero_features_for_each_subset(
        features_datasets: List[tf.data.Dataset]
) -> tf.data.Dataset:
    """
    Create zero features for each dataset

    Parameters
    ----------
    features_datasets
        lists of tensorflow datasets with features inside

    Returns
    -------
    dataset_with_zero_features
        datasets with features of tf.zeros_like(each_feature) for each
        dataset

    """

    def _get_default_features(nested_features):
        features_flatten = nest_utils.flatten_nested_struct(nested_features)
        zero_values_flatten = {
            each_key: tf.zeros_like(each_value)
            for each_key, each_value in features_flatten.items()}
        zero_values = nest_utils.unflatten_dict_to_nested(
            zero_values_flatten)
        return zero_values

    zero_features = []
    for each_subset in features_datasets:
        zero_features.append(each_subset.map(_get_default_features))

    zero_features_combined = (
        tf_data_utils.combine_features_from_list_of_dict_datasets(
            zero_features))
    return zero_features_combined


def _combine_features_from_datasets_with_same_filelist(
        datasets: List[Union[Dataset, DatasetFileList]],
        features_datasets: List[tf.data.Dataset],
        merge_on_same_file_list: List[bool],
        sampling_weights: Optional[List[float]]
) -> Tuple[List[tf.data.Dataset], List[str], List[float]]:
    """
    Combine features of datasets to single dataset if they have the same
    file lists

    Parameters
    ----------
    datasets
        list of datasets
    features_datasets
        list of data features
    merge_on_same_file_list
        list of flags if features of the dataset should be combined with other
        if other have same file list
    sampling_weights
        sampling weights to combine

    Returns
    -------
    new_features
        list of features after combination
    dataset_names
        list of new dataset names
    sampling_weights
        list of new sampling weights
    """

    def _combine_subsets_data(data, _combined_inds_lists: List[list],
                              _inds_to_leave: list,
                              combine_fn: Optional[Callable] = None):
        if combine_fn is None:
            combine_fn = lambda x: x

        data_to_leave = [data[each_ind] for each_ind in _inds_to_leave]
        data_combined = [
            combine_fn([data[each_ind] for each_ind in inds_to_combine])
            for inds_to_combine in _combined_inds_lists]
        new_data = data_combined + data_to_leave
        return new_data

    dataset_orig_names = [each_dataset.subtype for each_dataset in datasets]
    (combined_inds_lists, all_combined_inds, inds_to_leave
     ) = _get_dataset_indices_to_combine(datasets, merge_on_same_file_list)

    if not all_combined_inds:
        return features_datasets, dataset_orig_names, sampling_weights

    logger = logging.getLogger(__name__)
    logger.info("Datasets %s ")

    new_datasets = _combine_subsets_data(
        datasets, combined_inds_lists, inds_to_leave)
    new_features_datasets = _combine_subsets_data(
        features_datasets, combined_inds_lists, inds_to_leave,
        tf_data_utils.combine_features_from_list_of_dict_datasets)
    sampling_weights_new = _combine_subsets_data(
        sampling_weights, combined_inds_lists, inds_to_leave, np.mean)
    dataset_names_new = [
        (each_dataset.subtype if not isinstance(each_dataset, list)
         else each_dataset[0].subtype)
        for each_dataset in new_datasets]

    return new_features_datasets, dataset_names_new, sampling_weights_new


def _get_dataset_indices_to_combine(datasets, merge_on_same_file_list):
    possible_dataset_to_combine_inds = [
        i for i, (merge_flag, each_dataset) in enumerate(
            zip(merge_on_same_file_list, datasets))
        if isinstance(each_dataset, DatasetFileList) and merge_flag]
    if len(possible_dataset_to_combine_inds) <= 1:
        return None, None, None

    combined_inds_lists = []
    all_combined_inds = set()
    for i, each_ind in enumerate(possible_dataset_to_combine_inds):
        if each_ind in all_combined_inds:
            continue

        datasets_with_same_filelist_at_ind = []
        dataset_at_ind = datasets[each_ind]
        file_list_at_ind = dataset_at_ind.file_list
        for _, other_ind in enumerate(
                possible_dataset_to_combine_inds[i + 1:], i + 1):
            dataset_at_other_ind = datasets[other_ind]
            if dataset_at_other_ind.file_list == file_list_at_ind:
                datasets_with_same_filelist_at_ind.append(other_ind)
        if datasets_with_same_filelist_at_ind:
            (datasets_with_same_filelist_at_ind
             ) = [each_ind] + datasets_with_same_filelist_at_ind
            combined_inds_lists.append(
                datasets_with_same_filelist_at_ind)
            all_combined_inds.update(
                set(datasets_with_same_filelist_at_ind))

    inds_to_leave = [each_ind for each_ind, _ in enumerate(datasets)
                     if each_ind not in all_combined_inds]
    return combined_inds_lists, all_combined_inds, inds_to_leave


def _assert_same_tensor_shapes(tensor_shape1: Union[tf.TensorShape, list],
                               tensor_shape2: Union[tf.TensorShape, list]):
    if isinstance(tensor_shape1, tf.TensorShape):
        tensor_shape1 = tensor_shape1.as_list()
    if isinstance(tensor_shape2, tf.TensorShape):
        tensor_shape2 = tensor_shape2.as_list()
    if tensor_shape1 != tensor_shape2:
        msg = "Shape {} is not compatible with shape {}".format(
            tensor_shape1, tensor_shape2)
        raise ValueError(msg)


def _get_cache_fname(cache_dir: str, file_name: str):
    """
    Create name for cache file with temp stamp

    Parameters
    ----------
    cache_dir
        directory for cache
    file_name
        file name; to this file name the temp unique suffix will be added

    Returns
    -------
    cache_fname
        file name for cache
    """
    if cache_dir is None:
        return None
    io_utils.maybe_mkdir(cache_dir)
    _, cache_fname = tempfile.mkstemp(prefix=file_name + "-", dir=cache_dir)
    os.remove(cache_fname)
    logger = logging.getLogger(__name__)
    logger.info("Use cache with file file_name %s", cache_fname)
    return cache_fname


def _get_sample_mask(subset_name, all_subset_names):
    sample_mask = {'sample_mask_' + each_name: 0.0
                   for each_name in all_subset_names}
    sample_mask['sample_mask_' + subset_name] = 1.0
    return sample_mask


def _combine_keys(list_of_keys_list: list) -> list:
    keys_all = [each_key for each_list in list_of_keys_list
                for each_key in each_list]
    return list(set(keys_all))
