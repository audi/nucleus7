# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Interfaces for data feeders
"""
import abc
import logging
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generator
from typing import Iterator
from typing import Optional
from typing import Union

from nucleus7.core import project_artifacts
from nucleus7.core.nucleotide import Nucleotide
from nucleus7.data.data_filter import DataFilterMixin
from nucleus7.data.data_pipe import DataPipe
from nucleus7.data.data_pipe import DataPipeMixin
from nucleus7.data.file_list import FileList
from nucleus7.data.file_list import FileListMixin
from nucleus7.utils import mlflow_utils
from nucleus7.utils import np_utils
from nucleus7.utils import object_utils


class DataFeeder(Nucleotide, DataFilterMixin):
    """
    Class for the feeding of data to the tensorflow graph

    Parameters
    ----------
    allow_smaller_final_batch
        if the last batch may be smaller than batch_size

    Attributes
    ----------
    register_name_scope
        name scope for register and for the config logger
    _generator
        generator for data
    samples_n
        number of samples; should be modified inside of build_generator method
    """
    register_name_scope = "data_feeder"
    exclude_from_register = True

    def __init__(self, *,
                 allow_smaller_final_batch: bool = True):
        super().__init__([], name="dataset")
        self.allow_smaller_final_batch = allow_smaller_final_batch
        self.samples_n = -1
        self._generator = None

    @abc.abstractmethod
    def build_generator(self) -> Union[Generator, Iterator]:
        """
        Build generator for data
        """

    def create_data_from_generator(self, data: dict) -> dict:
        """
        Create data from generator

        Parameters
        ----------
        data
            output of generator
        """
        # pylint: disable=no-self-use
        # is an interface
        return data

    def get_combine_sample_to_batch_fn(self, combine_key: str
                                       ) -> Callable[[list], Any]:
        """
        Parameters
        ----------
        combine_key
            key that used for combination of samles to batch

        Returns
        -------
        combine_fn
            this function will be used to combine the samples for t
        """
        # pylint: disable=unused-argument,no-self-use
        # this is an interface
        return np_utils.stack_with_pad

    def combine_samples_to_batch(self, samples_data: list) -> dict:
        """
        Combine list of sample data to batch

        Parameters
        ----------
        samples_data
            list of dicts with sample data

        Returns
        -------
        batch_data
            batch data
        """
        all_data_keys = {each_key for each_sample in samples_data
                         for each_key in each_sample}
        batch = {
            each_data_key: self.get_combine_sample_to_batch_fn(each_data_key)(
                [each_sample[each_data_key] for each_sample in samples_data])
            for each_data_key in all_data_keys}
        return batch

    def get_batch(self, batch_size: int) -> dict:
        """
        Read the data and generates the feed_dict to inputs

        Parameters
        ----------
        batch_size
            batch size to use

        Returns
        -------
        batch
            mapping generated_keys to feeding value
        """
        logger = logging.getLogger(__name__)

        samples_data = []
        if self._generator is None:
            self._generator = self.build_generator()

        number_of_samples = 0
        while number_of_samples < batch_size:
            try:
                next_sample_from_generator = next(self._generator)
                sample_data = self.create_data_from_generator(
                    next_sample_from_generator)

                if not self.data_filter_true(**sample_data):
                    continue

                samples_data.append(sample_data)
                number_of_samples += 1
            except StopIteration:
                if not self.allow_smaller_final_batch or number_of_samples == 0:
                    logger.info('generator of %s is exhausted',
                                self.__class__.__name__)
                    raise StopIteration()
                break

        batch = self.combine_samples_to_batch(samples_data)
        return batch

    @object_utils.assert_is_built
    @mlflow_utils.log_nucleotide_exec_time_to_mlflow(method_name="get_batch")
    def __call__(self, batch_size):
        """
        Create the batch

        Parameters
        ----------
        batch_size
            batch size to use

        Returns
        -------
        batch
            mapping generated_keys to feeding value
        """
        # pylint: disable=arguments-differ
        # parent __call__ method has more generic signature
        return self.get_batch(batch_size)

    @staticmethod
    def from_data_pipe(
            data_pipe: DataPipe, file_list: Optional[FileList] = None,
            **datafeeder_kwargs
    ) -> Union['DataFeederFromPipe',
               'DataFeederFileListFromPipe']:
        """
        Construct DataFeeder out of the pipe

        Parameters
        ----------
        data_pipe
            data pipe to use
        file_list
            file list to use; if provided, it will be DataFeederFileList
            otherwise it will be DataFeeder
        datafeeder_kwargs
            additional kwargs to pass to corresponding DataFeeder constructor

        Returns
        -------
        data_feeder
            data feeder with data pipe inside
        """
        if file_list is None:
            return DataFeederFromPipe(data_pipe=data_pipe, **datafeeder_kwargs)
        return DataFeederFileListFromPipe(
            data_pipe=data_pipe, file_list=file_list, **datafeeder_kwargs)


class DataFeederFileList(DataFeeder, FileListMixin):
    """
    Class for the feeding of data to the tensorflow graph

    Parameters
    ----------
    file_list
        built FileList instance
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
                 file_list_keys_mapping: Optional[Dict[str, str]] = None,
                 **datafeeder_kwargs):
        self.file_list = file_list
        self.file_list_keys_mapping = file_list_keys_mapping
        super().__init__(**datafeeder_kwargs)

    @project_artifacts.add_project_artifact(
        "file_list", "file_list.file_names_pairs")
    def build(self) -> "DataFeederFileList":
        super().build()
        self.file_list = self.file_list.filter_by_keys(
            self.file_list_keys_required, self.file_list_keys_optional,
            self.file_list_keys_mapping)
        self.samples_n = len(self.file_list)
        return self

    def build_generator(self) -> Union[Generator, Iterator]:
        """
        Create generator for file_list
        """

        return iter(self.file_list)

    @abc.abstractmethod
    def read_element_from_file_names(self, **fnames: Dict[str, str]) -> dict:
        """
        Take the dict of file names and return dictionary of data element

        Parameters
        ----------
        **fnames
            dict of strings with file names

        Returns
        -------
        element_data
            element data
        """

    def create_data_from_generator(self, data: dict):
        """
        Wrap read_element_from_file_names function to more specific name
        """
        return self.read_element_from_file_names(**data)


class DataFeederFromPipe(DataPipeMixin, DataFeeder):
    """
    DataFeeder that uses data_pipe for data generation

    Parameters
    ----------
    data_pipe
        data pipe object to use
    output_keys_mapping
        keys mapping to map the data_pipe nucleotide results to data feeder
        generated keys
    number_of_samples
        number of samples to generate
    """
    exclude_from_register = True
    exclude_args_from_log = ["data_pipe"]

    def __init__(self, data_pipe: DataPipe,
                 output_keys_mapping: Optional[dict] = None,
                 number_of_samples: Optional[int] = -1,
                 **datafeeder_kwargs):
        self.number_of_samples = number_of_samples
        DataPipeMixin.__init__(self, data_pipe=data_pipe,
                               output_keys_mapping=output_keys_mapping)
        DataFeeder.__init__(self, **datafeeder_kwargs)

    def build(self):
        super().build()
        _check_data_pipe_compatibility(self, self.data_pipe)
        self.data_pipe.build_dna()
        return self

    def build_generator(self) -> Union[Generator, Iterator]:
        number_of_generated_samples = 0
        if not self.number_of_samples or self.number_of_samples < 0:
            condition = lambda x: True
        else:
            condition = lambda x: (number_of_generated_samples
                                   < self.number_of_samples)

        while condition(number_of_generated_samples):
            data = self.process_sample_with_pipe()
            number_of_generated_samples += 1
            yield data


class DataFeederFileListFromPipe(DataPipeMixin, DataFeederFileList):
    """
    DataFeeder that uses data_pipe for data generation and file list

    Parameters
    ----------
    data_pipe
        data pipe object to use
    output_keys_mapping
        keys mapping to map the data_pipe nucleotide results to data feeder
        generated keys
    """
    exclude_from_register = True
    exclude_args_from_log = ["data_pipe", "file_list"]

    def __init__(self, data_pipe: DataPipe,
                 output_keys_mapping: Optional[dict] = None,
                 **datafeeder_kwargs):
        if "file_list_keys_mapping" in datafeeder_kwargs:
            raise ValueError("file_list_keys_mapping should be provided inside "
                             "of DataReaders for DataPipe data feeders!")
        DataPipeMixin.__init__(self, data_pipe=data_pipe,
                               output_keys_mapping=output_keys_mapping)
        DataFeederFileList.__init__(self, **datafeeder_kwargs)
        self.file_list_keys = self.file_list.keys()

    def build(self):
        super().build()
        _check_data_pipe_compatibility(self, self.data_pipe)
        self.data_pipe.build_dna()
        return self

    def read_element_from_file_names(self, **fnames: Dict[str, str]) -> dict:
        return self.process_sample_with_pipe(**fnames)


def _check_data_pipe_compatibility(data_feeder: DataFeeder,
                                   data_pipe: DataPipe):
    if data_pipe.is_tensorflow_pipe:
        raise ValueError("Currently data pipes for use as data feeder must "
                         "cannot use tensorflow elements!")

    if isinstance(data_feeder, DataFeederFileList) and not data_pipe.readers:
        raise ValueError("Provide DataReaders to read from file list!")
    if (not isinstance(data_feeder, DataFeederFileList)
            and data_pipe.readers):
        raise ValueError("Provide FileList to use DataReaders")
    if data_pipe.readers and data_pipe.read_from_tfrecords:
        raise ValueError(
            "Currently DataFeeder from tfrecords is not supported!")
