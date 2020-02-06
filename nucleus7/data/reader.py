# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Data Reader interfaces to use with DataPipe
"""

import abc
from typing import Dict
from typing import Optional
from typing import Union

import tensorflow as tf

from nucleus7.data.base import BaseDataPipeNucleotide
from nucleus7.data.data_filter import DataFilterMixin
from nucleus7.data.file_list import FileListMixin
from nucleus7.utils import tf_data_utils
from nucleus7.utils import object_utils

_TFRECORDS_DATA_KEY = 'tfrecords'


class DataReader(BaseDataPipeNucleotide,
                 FileListMixin,
                 DataFilterMixin):
    """
    Base DataReader interface to use for DataPipe module which can read
    data from the file lists and then process it sample wise

    Parameters
    ----------
    file_list_keys_mapping
        mapping of the file list keys from original file_list to the keys passed
        to read method
    """
    register_name_scope = "data_reader"
    exclude_from_register = True
    _process_method_name = "read"

    def __init__(self, file_list_keys_mapping: Optional[Dict[str, str]] = None,
                 **nucleotide_kwargs):
        if nucleotide_kwargs.get("inbound_nodes"):
            raise ValueError("inbound_nodes are not used by DataReader!")
        self.file_list_keys_mapping = file_list_keys_mapping
        super().__init__(inbound_nodes=None, **nucleotide_kwargs)

    @abc.abstractmethod
    def read(self, **fnames: Union[dict, tf.data.Dataset]):
        """
        Read data from file names sample wise

        Parameters
        ----------
        fnames
            matched file names

        Returns
        -------
        data
            read data from matched file names
        """

    @object_utils.raise_exception_with_class_name
    def __call__(self, **fnames):
        # pylint: disable=arguments-differ
        # parent __call__ method has more generic signature
        fnames_filtered = self.remap_file_names(fnames)
        return self.read(**fnames_filtered)


# pylint: disable=abstract-method
# this uses abstract mixin, and is interface
class TfRecordsDataReader(DataReader, tf_data_utils.TfRecordsMixin):
    """
    Data Reader to read from tfrecords.

    Its main method will be applied to the tfrecords dataset.
    """
    file_list_keys = [_TFRECORDS_DATA_KEY]
    is_tensorflow = True

    def read(self, *, tfrecords: tf.data.Dataset):
        """
        Read from the tfrecord example
        """
        # pylint: disable=arguments-differ
        # parent method has more generic signature
        return self.parse_tfrecord_example(tfrecords)
