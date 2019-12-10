# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
DataPipe class to use as a collector for DataReaders and DataProcessors
"""

from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import tensorflow as tf

from nucleus7.core.gene_handler import GeneHandler
from nucleus7.data.processor import DataProcessor
from nucleus7.data.reader import DataReader
from nucleus7.data.reader import TfRecordsDataReader
from nucleus7.utils import mlflow_utils
from nucleus7.utils import model_utils
from nucleus7.utils import object_utils


class DataPipe(GeneHandler,
               model_utils.CustomSessionHandlerMixin):
    """
    Handler of DataReaders and DataProcessors.

    It will read the data from outer FileList if needed and then process it with
    the readers and processors in the way they were connected

    Currently all the nucleotides should have the same is_tensorflow flag set.
    Otherwise a ValueError will be thrown. Also data readers should all be
    tfrecords or all be not tfrecords readers. Otherwise ValueError will be
    thrown.

    Parameters
    ----------
    readers
        data readers to use
    processors
        data processors to use
    """
    nucleotide_type_dependency_map = {
        DataReader: [],
        DataProcessor: [
            DataReader,
            DataProcessor,
        ],
    }
    gene_name_and_nucleotide_super_cls = {
        "readers": DataReader,
        "processors": DataProcessor,
    }

    def __init__(self, *,
                 readers: Optional[Union[List[DataReader],
                                         Dict[str, DataReader],
                                         DataReader]] = None,
                 processors: Optional[Union[List[DataProcessor],
                                            Dict[str, DataProcessor],
                                            DataProcessor]] = None):
        self.processors = None  # type: Dict[str, DataProcessor]
        self.readers = None  # type: Dict[str, DataReader]
        super().__init__(readers=readers, processors=processors)
        if not self.readers and not self.processors:
            raise ValueError("At least one data reader or one data processor "
                             "must be provided!")
        self._save_target = None

    def build(self):
        super().build()
        self._check_nucleotide_compatibility()
        self._check_readers_compatibility()
        if self.inbound_nodes:
            raise ValueError("DataPipe cannot have inbound nodes!")
        return self

    @property
    def name(self):
        """
        Name of the data pipe
        """
        return "data_pipeline"

    @property
    def read_from_tfrecords(self) -> bool:
        """
        If the data should be read from tfrecords files

        Returns
        -------
        is_tf_records_pipe
            if the pipe uses all tfrecords readers
        """
        tfrecords_readers = [isinstance(each_reader, TfRecordsDataReader)
                             and each_reader.is_tensorflow
                             for each_reader in self.readers.values()]
        return all(tfrecords_readers)

    @property
    def is_tensorflow_pipe(self) -> bool:
        """
        Returns
        -------
        is_tensorflow_only
            True if all the nucleotides have is_tensorflow flag on it.
        """
        tensorflow_nucleotides = self._get_tensorflow_nucleotides()
        if tensorflow_nucleotides:
            return True
        return False

    @object_utils.assert_is_built
    @mlflow_utils.log_nucleotide_exec_time_to_mlflow(
        method_name="data_readers", log_on_train=False, log_on_eval=False)
    def process_readers(self, **fnames):
        """
        Process all readers

        Parameters
        ----------
        fnames
            file names to readers

        Returns
        -------
        read_data
            read data from fnames using readers
        """
        result = {}
        for each_reader_name, each_reader in sorted(self.readers.items()):
            result[each_reader_name] = each_reader(**fnames)
        return result

    @object_utils.assert_is_built
    @mlflow_utils.log_nucleotide_exec_time_to_mlflow(
        method_name="data_processors")
    def process_processors(self, **raw_data):
        """
        Process data using DataProcessors.

        Parameters
        ----------
        raw_data
            raw data to process

        Returns
        -------
        processed_data
            processed data using data processors
        """
        if not raw_data:
            raw_data = {}
        return self.process_gene(gene_name='processors',
                                 gene_inputs=raw_data)

    def __call__(self, **fnames):
        # pylint: disable=arguments-differ
        # parent __call__ method has more generic signature
        fnames = fnames or {}
        raw_data = self.process_readers(**fnames)
        data_encoded = self.process_processors(**raw_data)
        result = {}
        result.update(**raw_data, **data_encoded)
        return result

    @object_utils.assert_is_built
    def filter_sample_true(self, **data):
        """
        Apply all the filters on all the data and it evaluates to True if all
        filters pass, otherwise to False. In case of tensorflow nodes, it will
        have corresponding tensorflow bools

        Parameters
        ----------
        data
            data with keys as nucleotide names to proceed

        Returns
        -------
        flag
            True if all filters evaluate to True and False otherwise
        """
        if self.is_tensorflow_pipe:
            filter_all_true = tf.constant(True)
        else:
            filter_all_true = True
        for each_name, each_nucleotide_data in data.items():
            nucleotide = self.all_nucleotides[each_name]
            if hasattr(nucleotide, "data_filter_true"):
                filter_nucleotide = nucleotide.data_filter_true(
                    **each_nucleotide_data)
                if self.is_tensorflow_pipe:
                    filter_all_true = tf.logical_and(filter_all_true,
                                                     filter_nucleotide)
                else:
                    filter_all_true = filter_all_true and filter_nucleotide
        return filter_all_true

    def initialize_session(self):
        """
        Method to initialize the custom session variables for all the data
        nucleotides
        """
        for each_nucleotide in self.all_nucleotides.values():
            each_nucleotide.initialize_session()

    def _check_nucleotide_compatibility(self):
        tensorflow_nucleotides = self._get_tensorflow_nucleotides()
        not_tensorflow_nucleotides = self._get_not_tensorflow_nucleotides()
        if tensorflow_nucleotides and not_tensorflow_nucleotides:
            msg = ("Currently it is not possible to mix tensorflow and "
                   "not tensorflow data elements! "
                   "(tensorlow names: {}, not tensorflow names: {})"
                   ).format(tensorflow_nucleotides.keys(),
                            not_tensorflow_nucleotides.keys())
            raise ValueError(msg)

    def _get_tensorflow_nucleotides(self):
        tensorflow_nucleotides = {
            each_name: each_nucleotide
            for each_name, each_nucleotide in self.all_nucleotides.items()
            if each_nucleotide.is_tensorflow}
        return tensorflow_nucleotides

    def _get_not_tensorflow_nucleotides(self):
        not_tensorflow_nucleotides = {
            each_name: each_nucleotide
            for each_name, each_nucleotide in self.all_nucleotides.items()
            if not each_nucleotide.is_tensorflow}
        return not_tensorflow_nucleotides

    def _check_readers_compatibility(self):
        if not self.readers:
            return

        tfrecords_readers = [isinstance(each_reader, TfRecordsDataReader)
                             and each_reader.is_tensorflow
                             for each_reader in self.readers.values()]
        if not len(set(tfrecords_readers)) == 1:
            msg = ("Currently it is not possible to mix readers from tfrecords "
                   "and other sources! (readers: {})".format(self.readers))
            raise ValueError(msg)


class DataPipeMixin:
    """
    Mixin to use with DataPipe
    """

    def __init__(self, data_pipe: DataPipe,
                 output_keys_mapping: Optional[dict] = None):
        self.data_pipe = data_pipe
        self.output_keys_mapping = output_keys_mapping or {}
        self._last_nested_result = None

    @property
    def dna_helix(self):
        """
        Adds dna_helix property

        Returns
        -------
        dna_helix
            dna helix for the data pipe
        """
        return self.data_pipe.dna_helix

    @object_utils.classproperty
    def generated_keys_required(self):
        """
        Adds required generated_keys according to data pipe dna and the
        output_keys_mapping

        Returns
        -------
        generated_keys_required
            required generated keys
        """
        if hasattr(self, "data_pipe"):
            return self.data_pipe.get_flat_generated_keys_required(
                self.output_keys_mapping)
        return []

    @object_utils.classproperty
    def generated_keys_optional(self):
        """
        Adds optional generated_keys according to data pipe dna and the
        output_keys_mapping

        Returns
        -------
        generated_keys_optional
            optional generated keys
        """
        if hasattr(self, "data_pipe"):
            return self.data_pipe.get_flat_generated_keys_optional(
                self.output_keys_mapping)
        return []

    @object_utils.classproperty
    def dynamic_generated_keys(self):
        """
        if the mixin has dynamic generated keys

        Returns
        -------
        dynamic_generated_keys
            True if one of the data_pipe nucleotides has dynamic generated keys
            and False otherwise
        """
        if not hasattr(self, "data_pipe"):
            return False
        for each_nucleotide in self.data_pipe.all_nucleotides.values():
            if each_nucleotide.dynamic_generated_keys:
                return True
        return False

    def process_sample_with_pipe(self, **pipe_inputs):
        """
        Process sample inputs using pipe

        Parameters
        ----------
        pipe_inputs
            inputs to data pipe

        Returns
        -------
        results
            resulted data after application of the pipe
        """
        result_nested = self.data_pipe(**pipe_inputs)
        results_flat = self.data_pipe.flatten_results(
            result_nested, self.output_keys_mapping)
        self._last_nested_result = result_nested
        return results_flat

    def data_filter_true(self, **inputs) -> bool:
        """
        Override the main class data_filter_true with mixin one which uses
        data_pipe filters

        Parameters
        ----------
        **inputs
            inputs to data filter

        Returns
        -------
        filter_flag
            if no data_filters were added, returns all the time True, otherwise
            returns True if all the filters evaluate to True and False otherwise
        """
        # pylint: disable=unused-argument
        # inputs are not used directly, since they are mapped
        return self.data_pipe.filter_sample_true(**self._last_nested_result)

    def initialize_session(self):
        """
        Method to initialize the custom session variables for data pipe
        """
        self.data_pipe.initialize_session()
