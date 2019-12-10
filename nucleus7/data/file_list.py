# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Interfaces for FileList
"""
import copy
import os
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from nucleus7.core.base import BaseClass
from nucleus7.core.base import MetaLogAndRegister
from nucleus7.data.data_filter import DataFilterMixin
from nucleus7.utils import file_utils
from nucleus7.utils import nest_utils
from nucleus7.utils import nucleotide_utils
from nucleus7.utils import object_utils

_EMPTY_CHAR = ""


class FileList(BaseClass, DataFilterMixin, metaclass=MetaLogAndRegister):
    """
    FileList is used to match file names for different tasks and to iterate
    over it. It also supports original shuffle and slicing

    Parameters
    ----------
    file_names
        mapping of key to glob expression with file names with this key
    downsample_factor
        specifies how to downsample the file list, e.g. if downsample_factor = 5
        then each 5th file will be used
    number_of_shards
        number of shards
    shard_index
        current shard index
    name
        class name

    Attributes
    ----------
    register_name_scope
        name scope for register and for the config logger

    Raises
    ------
    ValueError
        if downsample_factor is less then 1
    ValueError
        if shard_index >= number_of_shards or is not set
    """
    register_name_scope = "file_list"
    exclude_from_register = True

    # pylint: disable=too-many-arguments
    # file list takes so many arguments, more split will be more confusing
    def __init__(self,
                 file_names: Dict[str, str],
                 name: Union[str, None] = None,
                 downsample_factor: int = 1,
                 number_of_shards: int = 1,
                 shard_index: int = 0):
        super(FileList, self).__init__()
        self._file_names = file_names
        self.name = name or self.__class__.__name__
        if downsample_factor < 1:
            msg = "Downsample factor must be int and >= 1, got {}".format(
                downsample_factor)
            raise ValueError(msg)
        if number_of_shards < 1 or shard_index >= number_of_shards:
            msg = ("Impossible combination of shard_index ({}) "
                   "and number_of_shards({})".format(shard_index,
                                                     number_of_shards))
            raise ValueError(msg)
        self.downsample_factor = downsample_factor
        self.number_of_shards = number_of_shards
        self.shard_index = shard_index

    @property
    def file_names_pairs(self) -> Union[dict, list]:
        """
        Get the file names list of matching pairs

        Returns
        -------
        file_names
            dict holding file names
        """
        if not self.built:
            return self._file_names
        return nest_utils.dict_of_lists_to_list_of_dicts(self._file_names)

    def get(self) -> Dict[str, Union[List, str]]:
        """
        Get the file names as a dict

        Returns
        -------
        file_names
            dict holding file names
        """
        return self._file_names

    def keys(self) -> list:
        """
        Returns keys from file_names in sorted order

        Returns
        -------
        keys
            keys from file_names
        """
        return sorted(self._file_names)

    def build(self) -> "FileList":
        """
        Build the file list object and apply match from glob

        Returns
        -------
        self
            self for chaining

        Raises
        ValueError
            if the file list empty after the match
        """
        super().build()
        self.match()
        if self.is_empty():
            msg = "File list {} is empty after the match".format(self.name)
            raise ValueError(msg)
        self.filter()
        self.sort()
        if self.number_of_shards > 1:
            self.shard()
        if self.downsample_factor > 1:
            self.downsample()
        return self

    def match_fn(self, path: str, key: str) -> str:
        """
        This method will be used inside of self.match()

        It takes the whole path of file and extracts the pattern from it that
        is gonna to be used to match the file names. File names for multiple
        keys are matched when they have the same pattern, e.g. to match
        {'inputs': 'input1.ext', 'labels': 'labels1.ext'} this method should
        return '1'

        Default pattern corresponds to the basename without extension

        Parameters
        ----------
        path
            complete path of the file
        key
            key from file_names using for to create this pattern

        Returns
        -------
        match_pattern
            pattern to use for matching of the file name pairs, e.g.
            the basename without extension

        """
        # pylint: disable=unused-argument
        # key argument is there for interface even if it is not used here
        # pylint: disable=no-self-use
        # is an interface
        match_pattern = _get_basename_without_extension(path)
        return match_pattern

    def sort_fn(self, path: str, key: str) -> Union[str, int, tuple]:
        """
        This method will be used inside of self.sort()

        It takes the whole path of file and extracts the string from it that
        is gonna to be used to sort the file names


        Parameters
        ----------
        path
            complete path of the file
        key
            key from file_names using for sorting

        Returns
        -------
        pattern_for_sort
            pattern that is used for sorting of file names

        """
        return self.match_fn(path, key)

    def match(self) -> "FileList":
        """
        Match file names between different keys, e.g. find file names that
        correspond to same sample

        Returns
        -------
        self
            self for chaining
        """
        matched_file_names = file_utils.match_file_names_with_glob(
            file_names_with_glob=self._file_names, match_fn=self.match_fn)
        self._file_names = matched_file_names
        return self

    def add_data_filter(self, data_filter):
        super().add_data_filter(data_filter)
        if self.built:
            self.filter()

    @object_utils.assert_is_built
    def filter(self):
        """
        Filter the FileList according to DatFilter
        """
        if not self.data_filters:
            return

        initial_length = len(self)
        for each_index, each_item in enumerate(self[::-1]):
            filter_flag = self.data_filter_true(**each_item)
            index_to_delete = initial_length - each_index - 1
            if not filter_flag:
                del self[index_to_delete]

    @object_utils.assert_is_built
    def sort(self) -> "FileList":
        """
        Sort the file names using self.sort_fn

        It will sort the first key in according to sort_fn and then will select
        other keys according to sorted indices

        Returns
        -------
        self
            self for chaining
        """
        first_subset_key, first_subset = _get_first_key_and_subset(
            self._file_names)
        sorted_indices, _ = zip(
            *sorted(enumerate(first_subset),
                    key=lambda x: self.sort_fn(x[1], key=first_subset_key)))
        file_names_sorted = {
            each_key: [each_list[ind_sorted] for ind_sorted in sorted_indices]
            for each_key, each_list in self._file_names.items()}
        self._file_names = file_names_sorted
        return self

    @object_utils.assert_is_built
    def downsample(self) -> "FileList":
        """
        Downsample, e.g. select each kth, the file_names according to
        self.downsample factor

        Returns
        -------
        self
            self for chaining
        """
        file_names = self._file_names
        file_names_downsampled = {
            each_key: each_file_names_list[::self.downsample_factor]
            for each_key, each_file_names_list in file_names.items()
        }
        self._file_names = file_names_downsampled
        return self

    def shard(self) -> "FileList":
        """
        Shard the file list according to number_of_shards and shard_index, e.g.
        file_list will include only 1/number_of_shards elements

        Returns
        -------
        self
            file_list for particular shard index

        """
        start_index = self.shard_index
        shard_step = self.number_of_shards
        file_names_sharded = _slice_file_names(
            self._file_names, slice(start_index, None, shard_step))
        self._file_names = file_names_sharded
        return self

    def filter_by_keys(self, keys_required: Union[list, None] = None,
                       keys_optional: Union[list, None] = None,
                       file_list_keys_mapping: Optional[Dict[str, str]] = None
                       ) -> "FileList":
        """
        Filter the file names by required and optional keys


        Parameters
        ----------
        keys_required
            keys that are required to be inside of file names and will be
            filtered
        keys_optional
            optional keys that are not required to be inside of file names,
            but will be filtered if they are inside
        file_list_keys_mapping
            mapping for file lists keys

        Returns
        -------
        file_list_filtered
            file list with filtered keys

        Raises
        ------
        ValueError
            if not all keys_required are inside of file_list
        """
        file_names_filtered = _filter_and_remap_file_list_keys(
            self.get(), keys_required, keys_optional, file_list_keys_mapping)
        file_list_filtered = self.from_matched_file_names(
            file_names_filtered, name=self.name)
        return file_list_filtered

    @classmethod
    def from_matched_file_names(cls,
                                file_names: Dict[str, List[str]],
                                name: Optional[str] = None) -> "FileList":
        """
        Create the file list from file names, that were already matched

        Parameters
        ----------
        file_names
            matched file names
        name
            name of file list

        Returns
        -------
        file_list
            file list with file_names inside
        """
        new_cls_with_incremented_name = _create_file_list_subclass(cls)
        file_list = new_cls_with_incremented_name(
            file_names=file_names, name=name)
        # pylint: disable=protected-access
        # this method is a factory, so it set the _built attribute
        file_list._built = True
        return file_list

    @object_utils.assert_is_built
    def is_empty(self) -> bool:
        """
        Checks if the file list is empty

        Returns
        -------
        True or False
            if the file list has no entries

        """
        return len(self) == 0

    def __contains__(self, item: Union[str, list, tuple]) -> bool:
        """
        Checks if the item is inside of the self.keys or in case if item is
        a list or tuple, check if all the items from item are inside of
        self.keys

        Parameters
        ----------
        item
            single key or a list of them

        Returns
        -------
        True
            if item or all the single items are inside of the self.keys
        False
            otherwise

        """
        if not isinstance(item, (list, tuple)):
            return item in self.keys()

        return all([single_item in self.keys() for single_item in item])

    @object_utils.assert_is_built
    def __getitem__(
            self, index: Union[int, slice]
    ) -> Union[Dict[str, str], "FileList"]:
        """
        Returns dict with all keys from file_names and their values on the index

        If the index is a slice, it will return FileList object

        Parameters
        ----------
        index
            index to select or a slice

        Returns
        -------
        item
            dict with keys from file_names and values from index position

        """
        if isinstance(index, slice):
            file_names_sliced = _slice_file_names(self._file_names, index)
            file_list_sliced = self.from_matched_file_names(file_names_sliced)
            return file_list_sliced

        item = {}
        for each_key, each_value_list in self._file_names.items():
            item[each_key] = each_value_list[index]
        return item

    @object_utils.assert_is_built
    def __setitem__(self, index: int, value: dict):
        """
        Set the item with keys from file_names to the index position

        Parameters
        ----------
        index
            index
        value
            mapping {key: value} to be set

        Raises
        ------
        AssertionError
            if the keys from value do not belong to self._file_names keys

        """
        if not isinstance(value, dict) and set(value) == set(self._file_names):
            msg = "You can set the value with following keys: {}".format(
                self._file_names.keys())
            raise TypeError(msg)

        for each_value_key in value:
            self._file_names[each_value_key][index] = value[each_value_key]

    def __delitem__(self, key):
        for each_key in self._file_names:
            del self._file_names[each_key][key]

    @object_utils.assert_is_built
    def __len__(self) -> int:
        """
        Returns length of file list

        Returns
        -------
        len_of_file_list
            length of file list
        """
        if not self._file_names:
            return 0

        return len(_get_first_key_and_subset(self._file_names)[1])

    @object_utils.assert_is_built
    def __add__(self, other) -> "FileList":
        """
        Add two file lists

        Parameters
        ----------
        other
            other FileList

        Returns
        -------
        file_list_sum
            file list with file names as sum of self and other

        """
        object_utils.assert_object_is_built(other)
        self_len = len(self)
        other_len = len(other)
        empty_of_self_len = [_EMPTY_CHAR] * self_len
        empty_of_other_len = [_EMPTY_CHAR] * other_len
        file_names_sum = copy.deepcopy(self.get())

        all_keys = set(list(self.get()) + list(other.get()))

        for each_key in all_keys:
            if each_key not in self.get():
                file_names_sum[each_key] = empty_of_self_len
            if each_key in other.get():
                file_names_sum[each_key].extend(other.get()[each_key])
            else:
                file_names_sum[each_key].extend(empty_of_other_len)
        file_list_sum = self.from_matched_file_names(file_names_sum)
        return file_list_sum

    def __eq__(self, o: "FileList") -> bool:
        return self.get() == o.get()

    def __repr__(self):
        representation = "{} with keys {} and length {}".format(
            self.name, self.keys(), len(self))
        return representation


class FileListExtendedMatch(FileList):
    """
    FileList is used to match file names for different tasks and to iterate
    over it. It also supports original shuffle and slicing

    Also allows to provide the match_suffixes and match_prefixes to use for
    match

    Parameters
    ----------
    file_names
        mapping of key to glob expression with file names with this key
    match_suffixes
        dict with same keys as file_names and specifies the suffix for basename
        that will be removed before matching
    match_prefixes
        dict with same keys as file_names and specifies the prefix for basename
        that will be removed before matching
    """
    exclude_from_register = True

    def __init__(self,
                 file_names: Dict[str, Union[str, str]],
                 match_suffixes: Union[Dict[str, str], None] = None,
                 match_prefixes: Union[Dict[str, str], None] = None,
                 **file_list_kwargs):
        super().__init__(file_names=file_names, **file_list_kwargs)
        self.match_suffixes = match_suffixes or {}
        self.match_prefixes = match_prefixes or {}
        assert set.issubset(set(self.match_suffixes), set(self._file_names)), (
            "Keys for match_prefixes must be a subset of file_names "
            "(file_names: {}, match suffixes: {})".format(
                list(file_names.keys()), list(match_suffixes.keys()))
        )
        assert set.issubset(set(self.match_prefixes), set(self._file_names)), (
            "Keys for match_prefixes must be a subset of file_names "
            "(file_names: {}, match prefixes: {})".format(
                list(file_names.keys()), list(match_prefixes.keys()))
        )

    def match_fn(self, path: str, key: str):
        """
        This method will be used inside of self.match()

        It uses basename without extension and removes suffixes from
        self.matching_suffixes and prefixes from self.matching_prefixes

        Parameters
        ----------
        path
            complete path of the file
        key
            key from file_names using for to create this pattern

        Returns
        -------
        match_pattern
            pattern to use for matching of the file name pairs, e.g.
            the basename without extension

        """
        match_suffix = self.match_suffixes.get(key, None)
        match_prefix = self.match_prefixes.get(key, None)
        match_pattern = _get_basename_without_extension(path)
        match_pattern = file_utils.remove_prefix(match_pattern, match_prefix)
        match_pattern = file_utils.remove_suffix(match_pattern, match_suffix)
        return match_pattern


class FileListMixin:
    """
    Mixin to add the file list interface to other classes
    """
    file_list_keys = []

    # pylint: disable=no-self-argument
    # classproperty is class property and so cls must be used
    @object_utils.classproperty
    def file_list_keys_required(cls) -> List[str]:
        """
        All required file list keys

        Returns
        -------
        required_file_list_keys
            required file list keys, e.g. keys that doesnt have "_" prefix
        """
        return [k for k in cls.file_list_keys if not k.startswith('_')]

    # pylint: disable=no-self-argument
    # classproperty is class property and so cls must be used
    @object_utils.classproperty
    def file_list_keys_optional(cls) -> List[str]:
        """
        All optional file list keys

        Returns
        -------
        optional_file_list_keys
            optional file list keys with stripped "_" prefix,
            e.g. keys that have "_" prefix
        """
        return [k[1:] for k in cls.file_list_keys if k.startswith('_')]

    # pylint: disable=no-self-argument
    # classproperty is class property and so cls must be used
    @object_utils.classproperty
    def file_list_keys_all(cls) -> List[str]:
        """
        Add file list keys

        Returns
        -------
        all_file_list_keys
            all file list keys including required and optional with stripped
            "_" prefix
        """

        return cls.file_list_keys_required + cls.file_list_keys_optional

    def remap_file_names(self, file_names) -> dict:
        """
        Remap file list keys according to file_list_keys_mapping

        Parameters
        ----------
        file_names
            file names to remap

        Returns
        -------
        remapped_file_names
            remapped file names
        """
        file_list_keys_mapping = getattr(self, "file_list_keys_mapping", {})
        remapped_file_list = _filter_and_remap_file_list_keys(
            file_names, self.file_list_keys_required,
            self.file_list_keys_optional, file_list_keys_mapping)

        return remapped_file_list


def _filter_and_remap_file_list_keys(
        file_names: dict, keys_required: Optional[List[str]] = None,
        keys_optional: Optional[List[str]] = None,
        file_list_keys_mapping: Optional[Dict[str, str]] = None):
    file_list_keys_mapping = file_list_keys_mapping or {}
    keys_required = keys_required or []
    keys_optional = keys_optional or []
    file_names_remapped = nucleotide_utils.remap_single_input(
        file_names, file_list_keys_mapping)
    if not set(keys_required).issubset(set(file_names_remapped.keys())):
        msg = (
            "Provided required file list keys are not inside of remapped "
            "file_list! "
            " (file_list keys: {}, mapping: {}, required_keys: {})".format(
                file_names.keys(), file_list_keys_mapping, keys_required))
        raise ValueError(msg)
    file_names_filtered = {
        each_file_list_key: file_names_remapped[each_file_list_key]
        for each_file_list_key in keys_required
    }
    file_names_filtered.update({
        each_file_list_key: file_names_remapped[each_file_list_key]
        for each_file_list_key in keys_optional
        if each_file_list_key in file_names_remapped
    })
    return file_names_filtered


def _get_first_key_and_subset(file_names: dict):
    first_key = sorted(file_names.keys())[0]
    return first_key, file_names[first_key]


def _get_basename_without_extension(path: str):
    basename = os.path.basename(path)
    basename_without_extension = os.path.splitext(basename)[0]
    return basename_without_extension


def _slice_file_names(file_names: Dict[str, List[str]],
                      slice_indexes: slice) -> Dict[str, List[str]]:
    file_names_sliced = {}
    for each_key, each_value_list in file_names.items():
        file_names_sliced[each_key] = each_value_list[slice_indexes]
    return file_names_sliced


def _create_file_list_subclass(cls: type):
    new_class_name = _increment_class_name(cls.__name__)
    new_class = type(new_class_name, (cls,),
                     {"__init__": cls.__init__,
                      "exclude_from_register": True})
    return new_class


def _increment_class_name(class_name: str) -> str:
    class_name_split = class_name.rsplit("_", 1)
    if len(class_name_split) == 1 or not class_name_split[-1].isdigit():
        return "_".join([class_name, "1"])

    current_index = int(class_name_split[-1])
    new_index = current_index + 1
    return "_".join([class_name_split[0], str(new_index)])
