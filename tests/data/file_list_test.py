# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import copy
import os
import random
from unittest.mock import MagicMock

from absl.testing import parameterized

from nucleus7.data.data_filter import DataFilter
from nucleus7.data.file_list import FileList
from nucleus7.data.file_list import FileListExtendedMatch
from nucleus7.data.file_list import FileListMixin
from nucleus7.test_utils.model_dummies import FileListDummy
from nucleus7.test_utils.test_utils import TestCaseWithTempDir


class _FileListMatchDummy(FileList):
    exclude_from_register = True

    def match_fn(self, path: str, key: str):
        basename = os.path.splitext(os.path.basename(path))[0]
        match_pattern = basename[:6] + basename[7:]
        return match_pattern


class _FileListMixinDummy(FileListMixin):
    exclude_from_register = True
    file_list_keys = ['key1', 'key2', '_key3']


class _DummyDataFilterOddInput1(DataFilter):
    exclude_from_register = True

    def predicate(self, input1: str, input2: str) -> bool:
        input1_number = int(os.path.splitext(input1)[0].split("_")[-1])
        return bool(input1_number % 2)


class TestFileList(TestCaseWithTempDir, parameterized.TestCase):

    def setUp(self):
        self.number_of_files = 20
        self.number_of_files2 = 10
        self.number_of_files3 = 10
        self.file_names = {'input1': ['input_1_{:03d}.ext'.format(i)
                                      for i in range(self.number_of_files)],
                           'input2': ['input_2_{:03d}.ext'.format(i)
                                      for i in range(self.number_of_files)]}
        self.file_names_glob = {'input1': 'input_1_*.ext',
                                'input2': 'input_2_*.ext'}
        self.file_names2 = {'input1': ['input2_1_{:03d}.ext'.format(i)
                                       for i in range(self.number_of_files2)],
                            'input2': ['input2_2_{:03d}.ext'.format(i)
                                       for i in range(self.number_of_files2)]}

        self.file_names3 = {'input1': ['input3_1_{:03d}.ext'.format(i)
                                       for i in range(self.number_of_files3)],
                            'input3': ['input_3_{:03d}.ext'.format(i)
                                       for i in range(self.number_of_files3)]
                            }
        random.seed(4567)
        super().setUp()

    def test_getitem_index(self):
        file_names = self.file_names
        file_list = FileListDummy(file_names=file_names).build()

        file_names_1_must = {'input1': 'input_1_001.ext',
                             'input2': 'input_2_001.ext'}

        self.assertDictEqual(
            file_names_1_must,
            file_list[1]
        )

    def test_getitem_slice(self):
        file_names = self.file_names
        file_names_2_10_must = {k: v[2:10] for k, v in file_names.items()}
        file_list = FileListDummy(file_names=file_names).build()
        file_list_sliced = file_list[2:10]
        self.assertIsInstance(file_list_sliced, FileList)
        self.assertTrue(file_list_sliced.built)
        self.assertDictEqual(
            file_names_2_10_must,
            file_list_sliced.get()
        )

    def test_len(self):
        file_names = self.file_names
        file_list = FileListDummy(file_names=file_names).build()
        self.assertEqual(len(file_list), self.number_of_files)

    def test_contains(self):
        file_names = self.file_names
        file_list = FileListDummy(file_names=file_names).build()
        self.assertTrue('input1' in file_list)
        self.assertTrue('input2' in file_list)
        self.assertFalse('input3' in file_list)
        self.assertTrue(['input1', 'input2'] in file_list)
        self.assertTrue(['input1'] in file_list)
        self.assertFalse(['input1', 'input3'] in file_list)

    def test_setitem(self):
        file_names = self.file_names
        file_list = FileListDummy(file_names=file_names).build()

        file_names_10_set_must = copy.deepcopy(file_list.get())
        file_names_10_set_must['input1'][10] = 10
        file_names_10_set_must['input2'][10] = 20
        file_list[10] = {'input1': 10, 'input2': 20}
        self.assertDictEqual(
            file_names_10_set_must,
            file_list.get()
        )
        with self.assertRaises(TypeError):
            file_list[11] = 10

        file_list[15] = {'input1': 15}
        file_names_10_set_must['input1'][15] = 15
        self.assertDictEqual(
            file_names_10_set_must,
            file_list.get()
        )

    def test_delitem(self):
        file_names = self.file_names
        file_list = FileListDummy(file_names=file_names).build()
        del file_list[10]

        file_names_must = {k: v[:10] + v[11:]
                           for k, v in self.file_names.items()}
        self.assertEqual(len(file_list), self.number_of_files - 1)
        self.assertDictEqual(file_names_must,
                             file_list.get())

    def test_filter(self):
        data_filter = _DummyDataFilterOddInput1()
        file_names = copy.deepcopy(self.file_names)
        file_list = FileListDummy(file_names=file_names)
        file_list.add_data_filter(data_filter)
        file_list.build()
        file_names_must = {k: v[1::2]
                           for k, v in self.file_names.items()}
        self.assertEqual(10,
                         len(file_list))
        self.assertDictEqual(file_names_must,
                             file_list.get())

    def test_match(self):
        file_names_orig = self.file_names
        file_names_glob = self.file_names_glob
        temp_dir = self.get_temp_dir()
        file_names_orig_with_dir = _add_temp_dir_to_fnames(
            temp_dir, file_names_orig)
        file_names_glob = _add_temp_dir_to_glob_fnames(
            temp_dir, file_names_glob)

        _create_temp_files(file_names_orig_with_dir)

        file_list = _FileListMatchDummy(file_names=file_names_glob).build()
        file_list.sort()
        self.assertDictEqual(file_names_orig_with_dir,
                             file_list.get())

    @parameterized.parameters({"downsample_factor": 0},
                              {"downsample_factor": 1},
                              {"downsample_factor": 10})
    def test_build(self, downsample_factor):
        file_names = self.file_names
        if downsample_factor < 1:
            with self.assertRaises(ValueError):
                FileListDummy(file_names=file_names,
                              downsample_factor=downsample_factor)
            return

        file_list = FileListDummy(file_names=file_names,
                                  downsample_factor=downsample_factor)
        file_list.match = MagicMock(return_value=None)
        file_list.sort = MagicMock(return_value=None)
        file_list.downsample = MagicMock(return_value=None)
        file_list.build()
        self.assertEqual(1,
                         file_list.match.call_count)
        self.assertEqual(1,
                         file_list.sort.call_count)
        if downsample_factor > 1:
            self.assertEqual(1,
                             file_list.downsample.call_count)
        else:
            self.assertEqual(0,
                             file_list.downsample.call_count)

    def test_sort(self):
        file_names = self.file_names
        file_names_orig = copy.deepcopy(file_names)
        file_list = FileListDummy(file_names=file_names).build()
        random.shuffle(file_list)
        self.assertNotEqual(file_names_orig,
                            file_list.get())
        file_list.sort()
        self.assertDictEqual(file_names_orig,
                             file_list.get())

    def test_sort_tuple(self):
        def _sort_fn(path, key):
            basename = os.path.splitext(os.path.basename(path))[0]
            keys = basename.split('_')[::-1]
            return tuple(keys)

        file_names = {'input': ['123_10', '123_11', '123_13',
                                '124_01', '124_02', '124_13']}
        file_names_sorted_must = {
            'input': ['124_01', '124_02', '123_10', '123_11',
                      '123_13', '124_13']}
        file_names_orig = copy.deepcopy(file_names)
        file_list = FileListDummy(file_names=file_names).build()
        file_list.sort_fn = MagicMock(side_effect=_sort_fn)
        random.shuffle(file_list)
        self.assertNotEqual(file_names_orig,
                            file_list.get())
        file_list.sort()
        self.assertDictEqual(file_names_sorted_must,
                             file_list.get())

    def test_shuffle(self):
        file_names = self.file_names
        file_names_orig = copy.deepcopy(file_names)
        file_list = FileListDummy(file_names=file_names).build()
        random.shuffle(file_list)
        self.assertNotEqual(file_names_orig,
                            file_list.get())

        file_names_from_list = file_list.get()
        file_names_from_list_as_set = {k: set(v)
                                       for k, v in file_names_from_list.items()}
        file_names_as_set_must = {k: set(v)
                                  for k, v in file_names_orig.items()}
        self.assertDictEqual(file_names_as_set_must,
                             file_names_from_list_as_set)

    @parameterized.parameters({"downsample_factor": 1},
                              {"downsample_factor": 10})
    def test_downsample(self, downsample_factor):
        file_names = self.file_names
        file_names_orig = copy.deepcopy(file_names)
        file_list = FileListDummy(file_names=file_names,
                                  downsample_factor=downsample_factor).build()
        if downsample_factor == 1:
            file_names_must = file_names_orig
        else:
            file_names_must = {
                each_key: each_list[::10]
                for each_key, each_list in file_names_orig.items()}

        self.assertDictEqual(file_names_must,
                             file_list.get())

    def test_add(self):
        file_names1 = self.file_names
        file_names1_orig = copy.deepcopy(file_names1)
        file_list1 = FileListDummy(file_names=file_names1).build()

        file_names2 = self.file_names2
        file_names2_orig = copy.deepcopy(file_names2)
        file_list2 = FileListDummy(file_names=file_names2).build()

        file_names3 = self.file_names3
        file_names3_orig = copy.deepcopy(file_names3)
        file_list3 = FileListDummy(file_names=file_names3).build()

        file_list1_2 = file_list1 + file_list2
        file_list1_2_3 = file_list1 + file_list2 + file_list3
        self.assertIsInstance(file_list1_2,
                              FileList)
        self.assertIsInstance(file_list1_2_3,
                              FileList)

        self.assertTrue(file_list1_2.built)
        self.assertTrue(file_list1_2_3.built)

        self.assertDictEqual(file_names1_orig,
                             file_list1.get())
        self.assertDictEqual(file_names2_orig,
                             file_list2.get())
        self.assertDictEqual(file_names3_orig,
                             file_list3.get())

        self.assertEqual(len(file_list1) + len(file_list2),
                         len(file_list1_2))
        self.assertEqual(len(file_list1) + len(file_list2) + len(file_list3),
                         len(file_list1_2_3))

        self.assertTrue(
            all([len(file_name_list) == len(file_list1_2)
                 for file_name_list in file_list1_2.get().values()]))
        self.assertTrue(
            all([len(file_name_list) == len(file_list1_2_3)
                 for file_name_list in file_list1_2_3.get().values()]))

        file_names_1_2_must = {}
        for each_key in file_names1_orig.keys():
            file_names_1_2_must[each_key] = (file_names1_orig[each_key]
                                             + file_names2_orig[each_key])
        file_names_1_2_3_must = copy.deepcopy(file_names_1_2_must)

        empty_list_3 = [""] * len(file_list3)
        empty_list_1_2 = [""] * len(file_list1_2)

        all_keys_1_2_3 = set(list(file_names_1_2_must) + list(file_names3_orig))

        for each_key in all_keys_1_2_3:
            if each_key not in file_names_1_2_must:
                file_names_1_2_3_must[each_key] = empty_list_1_2
            if each_key in file_names3_orig:
                file_names_1_2_3_must[each_key].extend(
                    file_names3_orig[each_key])
            else:
                file_names_1_2_3_must[each_key].extend(empty_list_3)

        self.assertDictEqual(file_names_1_2_3_must,
                             file_list1_2_3.get())

    def test_filter_by_keys(self):
        file_names = self.file_names
        file_names_orig = copy.deepcopy(file_names)
        file_list = FileListDummy(file_names=file_names).build()
        keys_required = ['input1']
        keys_optional = ['input2']
        file_list_filtered = file_list.filter_by_keys(
            keys_required=keys_required)
        self.assertDictEqual({'input1': file_names_orig['input1']},
                             file_list_filtered.get())

        file_list_filtered = file_list.filter_by_keys(
            keys_optional=keys_optional)
        self.assertDictEqual({'input2': file_names_orig['input2']},
                             file_list_filtered.get())

        file_list_filtered = file_list.filter_by_keys(
            keys_required=keys_required, keys_optional=keys_optional)
        self.assertDictEqual(file_names_orig,
                             file_list_filtered.get())

    def test_filter_by_keys_with_mapping(self):
        file_names = {'input1': ['input_1_{:03d}.ext'.format(i)
                                 for i in range(self.number_of_files)],
                      'input2': ['input_2_{:03d}.ext'.format(i)
                                 for i in range(self.number_of_files)],
                      'input3': ['input_2_{:03d}.ext'.format(i)
                                 for i in range(self.number_of_files)]}
        file_names_orig = copy.deepcopy(file_names)
        file_list_keys_mapping = {"input1": "input1_r",
                                  "input3": "_"}
        file_list = FileListDummy(file_names=file_names).build()
        keys_required = ['input1']
        keys_optional = ['input2']
        with self.assertRaises(ValueError):
            _ = file_list.filter_by_keys(
                keys_required=keys_required,
                keys_optional=keys_optional,
                file_list_keys_mapping=file_list_keys_mapping)

        keys_required = ['input1_r']
        file_list_filtered = file_list.filter_by_keys(
            keys_required=keys_required,
            keys_optional=keys_optional,
            file_list_keys_mapping=file_list_keys_mapping)

        file_names_must = {'input1_r': file_names_orig['input1'],
                           'input2': file_names_orig['input2']}
        self.assertDictEqual(file_names_must,
                             file_list_filtered.get())

    def test_from_matched_file_names(self):
        file_names = self.file_names
        file_names_orig = copy.deepcopy(file_names)
        file_list = FileListDummy(file_names=file_names).build()
        file_list_new = file_list.from_matched_file_names(file_list.get())
        self.assertIsNot(file_list, file_list_new)
        self.assertDictEqual(file_names_orig,
                             file_list_new.get())

    @parameterized.parameters({'number_of_shards': 10, 'shard_index': 2},
                              {'number_of_shards': 7, 'shard_index': 6},
                              {'number_of_shards': 2, 'shard_index': 2},
                              {'number_of_shards': -1, 'shard_index': 2})
    def test_shard(self, number_of_shards, shard_index):
        file_names = self.file_names
        file_names_orig = copy.deepcopy(file_names)
        if -1 <= number_of_shards <= shard_index:
            with self.assertRaises(ValueError):
                FileListDummy(file_names=file_names,
                              number_of_shards=number_of_shards,
                              shard_index=shard_index)
            return

        file_list = FileListDummy(file_names=file_names,
                                  number_of_shards=number_of_shards,
                                  shard_index=shard_index).build()
        if number_of_shards <= 1:
            self.assertDictEqual(file_names_orig,
                                 file_list.get())
            return

        start_ind = shard_index
        shard_step = number_of_shards
        file_names_sharded_must = {k: v[start_ind:None:shard_step]
                                   for k, v in file_names_orig.items()}
        self.assertDictEqual(file_names_sharded_must,
                             file_list.get())


class TestFileListExtendedMatch(TestCaseWithTempDir,
                                parameterized.TestCase):
    def setUp(self):
        self.number_of_files = 10
        self.file_names = {
            'input1': ['data_{:03d}.ext'.format(i)
                       for i in range(self.number_of_files)],
            'input2': ['data_{:03d}.ext'.format(i)
                       for i in range(self.number_of_files)]}

        self.file_names_glob = {'input1': '*data_*.ext',
                                'input2': '*data_*.ext'}

        self.match_prefixes = {'inputs': 'prefix_input_'}
        self.match_suffixes = {'labels': 'labels_'}

        self.file_names_with_prefixes = {
            k: [self.match_prefixes.get(k, '') + fname
                for fname in fnames]
            for k, fnames in self.file_names.items()
        }
        self.file_names_with_suffixes = {
            k: [fname + self.match_suffixes.get(k, '')
                for fname in fnames]
            for k, fnames in self.file_names.items()
        }
        self.file_names_with_prefixes_and_suffixes = {
            k: [self.match_prefixes.get(k, '') + fname
                + self.match_suffixes.get(k, '')
                for fname in fnames]
            for k, fnames in self.file_names.items()
        }
        super().setUp()

    @parameterized.parameters({"add_prefixes": False, "add_suffixes": False},
                              {"add_prefixes": False, "add_suffixes": True},
                              {"add_prefixes": True, "add_suffixes": False},
                              {"add_prefixes": True, "add_suffixes": True})
    def test_match(self, add_prefixes, add_suffixes):
        if add_prefixes and add_suffixes:
            file_names = self.file_names_with_prefixes_and_suffixes
        elif add_prefixes:
            file_names = self.file_names_with_prefixes
        elif add_suffixes:
            file_names = self.file_names_with_suffixes
        else:
            file_names = self.file_names
        file_names_glob = self.file_names_glob
        temp_dir = self.get_temp_dir()
        file_names_orig_with_dir = _add_temp_dir_to_fnames(temp_dir, file_names)
        file_names_glob = _add_temp_dir_to_glob_fnames(
            temp_dir, file_names_glob)
        _create_temp_files(file_names_orig_with_dir)
        file_list = FileListExtendedMatch(
            file_names=file_names_glob).build()
        self.assertDictEqual(file_names_orig_with_dir,
                             file_list.get())


class TestFileListMixin(TestCaseWithTempDir):

    def test_file_list_keys_required(self):
        mixin = _FileListMixinDummy
        self.assertListEqual(['key1', 'key2'],
                             mixin.file_list_keys_required)

    def test_file_list_keys_optional(self):
        mixin = _FileListMixinDummy
        self.assertListEqual(['key3'],
                             mixin.file_list_keys_optional)

    def test_file_list_keys_all(self):
        mixin = _FileListMixinDummy
        self.assertListEqual(['key1', 'key2', 'key3'],
                             mixin.file_list_keys_all)

    def test_remap_file_names(self):
        mixin = _FileListMixinDummy()
        data = {"key1": 10, "key2": 20, "key3": 30}
        data_orig = copy.deepcopy(data)
        data_remapped = mixin.remap_file_names(data)
        self.assertDictEqual(data_orig, data_remapped)

        mixin.file_list_keys_mapping = {"key1": "key1_r",
                                        "key2": "_"}
        mixin.file_list_keys = ["key1_r", "_key3", "_key2"]
        data_remapped_must = {"key1_r": 10,
                              "key3": 30}
        data_remapped = mixin.remap_file_names(data_orig)
        self.assertDictEqual(data_remapped_must,
                             data_remapped)


def _create_temp_files(file_names: dict):
    for each_key, each_flist in file_names.items():
        for each_fname in each_flist:
            with open(each_fname, 'w') as f:
                f.write("")


def _add_temp_dir_to_fnames(temp_dir: str, file_names: dict) -> dict:
    file_names_with_dir = {
        each_key: [
            os.path.join(temp_dir, fname) for fname in each_fname_list]
        for each_key, each_fname_list in file_names.items()
    }
    return file_names_with_dir


def _add_temp_dir_to_glob_fnames(temp_dir: str, file_names: dict) -> dict:
    file_names_with_dir = {
        each_key: os.path.join(temp_dir, each_fglob)
        for each_key, each_fglob in file_names.items()}
    return file_names_with_dir
