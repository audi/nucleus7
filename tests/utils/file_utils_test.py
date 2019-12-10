# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import os
import random
import tempfile
from unittest.mock import patch

from absl.testing import parameterized

from nucleus7.test_utils.test_utils import TestCaseWithTempDir
from nucleus7.utils import file_utils
from nucleus7.utils import nest_utils


class TestFIleUtils(parameterized.TestCase, TestCaseWithTempDir):

    @staticmethod
    def _get_file_list(shuffle=True):
        random.seed(4564)
        suffixes = {'labels': '_labels'}
        fnames_input = ['f_{}.ext'.format(i) for i in range(120)]
        fnames_labels = ['f_{}_labels.ext'.format(i) for i in range(100)]
        if shuffle:
            random.shuffle(fnames_input)
            random.shuffle(fnames_labels)
        file_list = {'input': fnames_input, 'labels': fnames_labels}
        return file_list, suffixes

    @staticmethod
    def _get_file_list_as_list():
        file_list = [{'image': ['image1.ext', 'image2.ext'],
                      'labels': ['image1_label.ext', 'image3_label.ext']},
                     {'image': ['image10.ext', 'image20.ext'],
                      'labels2': ['image10_label2.ext', 'image20_label2.ext']}]
        suffixes = {'labels': '_label', 'labels2': '_label2'}
        return file_list, suffixes

    @patch('nucleus7.utils.file_utils.maybe_fnames_from_glob',
           side_effect=lambda x: x)
    def test_match_file_names(self, f):
        file_list, suffixes = self._get_file_list()
        file_list_new, keys_for_fnames = file_utils.match_file_names(
            file_list, suffixes)

        self.assertSetEqual(set(file_list_new.keys()), set(file_list))
        self.assertEqual(len(file_list_new['labels']),
                         len(file_list_new['input']))
        self.assertSetEqual(set(file_list['labels']),
                            set(file_list_new['labels']))
        labels_must = [file_utils.add_suffix(f, '_labels')
                       for f in file_list_new['input']]
        self.assertListEqual(file_list_new['labels'], labels_must)
        self.assertEqual(len(keys_for_fnames['labels']),
                         len(keys_for_fnames['input']))
        self.assertListEqual(list(keys_for_fnames['labels']), [1] * 100)
        self.assertListEqual(list(keys_for_fnames['input']), [1] * 100)

    @patch('nucleus7.utils.file_utils.maybe_fnames_from_glob',
           side_effect=lambda x: x)
    def test_match_file_names_as_list(self, f):
        file_list, suffixes = self._get_file_list_as_list()
        file_list_new, keys_for_fnames = file_utils.match_file_names(
            file_list, suffixes)
        file_list_must = {
            'image': ['image1.ext', 'image10.ext', 'image20.ext'],
            'labels': ['image1_label.ext', '', ''],
            'labels2': ['', 'image10_label2.ext', 'image20_label2.ext']}
        keys_for_fnames_must = {'image': [1, 1, 1],
                                'labels': [1, 0, 0],
                                'labels2': [0, 1, 1]}
        self.assertDictEqual(file_list_new, file_list_must)
        self.assertDictEqual(keys_for_fnames, keys_for_fnames_must)

    @parameterized.parameters({'use_keys': ['input', 'labels']},
                              {'use_keys': ['input']})
    def test_match_file_names_with_glob(self, use_keys=['input']):
        def match_fn(path, key):
            basename = os.path.splitext(os.path.basename(path))[0]
            match_pattern = basename.replace("_labels", '')
            return match_pattern

        file_names, _ = self._get_file_list(shuffle=False)
        temp_dir = self.get_temp_dir()
        file_names_with_dir = {
            each_key: [
                os.path.join(temp_dir, fname) for fname in each_fname_list]
            for each_key, each_fname_list in file_names.items()
        }
        file_names_glob = {'input': os.path.join(temp_dir, "f_*[0-9].ext"),
                           'labels': os.path.join(temp_dir, "f_*_labels.ext")}

        file_names_with_dir = {k: v for k, v in file_names_with_dir.items()
                               if k in use_keys}
        file_names_glob = {k: v for k, v in file_names_glob.items()
                           if k in use_keys}

        self._create_temp_files(file_names_with_dir)
        matched_file_names = file_utils.match_file_names_with_glob(
            file_names_glob, match_fn
        )

        file_names_pairs_orig = nest_utils.dict_of_lists_to_list_of_dicts(
            file_names_with_dir)
        matched_file_names_pairs = nest_utils.dict_of_lists_to_list_of_dicts(
            matched_file_names)
        file_names_pairs_orig_sorted = sorted(file_names_pairs_orig,
                                              key=lambda x: x['input'])
        matched_file_names_pairs_sorted = sorted(matched_file_names_pairs,
                                                 key=lambda x: x['input'])
        self.assertListEqual(file_names_pairs_orig_sorted,
                             matched_file_names_pairs_sorted)

    def test_get_incremented_path(self):
        self.assertEqual("file/one.ext", file_utils.get_incremented_path(
            "file/one.ext", add_index=False))
        self.assertEqual("file/one-1.ext", file_utils.get_incremented_path(
            "file/one.ext", add_index=True))
        self.assertEqual("file/one-1.ext", file_utils.get_incremented_path(
            "file/one-1.ext", add_index=False))
        self.assertEqual("file/one-1-1.ext", file_utils.get_incremented_path(
            "file/one-1.ext", add_index=True))
        _, file_name = tempfile.mkstemp(suffix='.ext')
        file_name_incremented1 = file_utils.add_suffix(file_name, '-1')
        file_name_incremented1_1 = file_utils.add_suffix(file_name, '-1-1')
        file_name_incremented2 = file_utils.add_suffix(file_name, '-2')
        file_name_incremented1000 = file_utils.add_suffix(file_name, '-1000')
        file_name_incremented1001 = file_utils.add_suffix(file_name, '-1001')

        self.assertEqual(file_name_incremented1,
                         file_utils.get_incremented_path(
                             file_name, add_index=True))
        self.assertEqual(file_name_incremented1,
                         file_utils.get_incremented_path(
                             file_name, add_index=False))
        with open(file_name_incremented1, 'w') as f:
            f.write('')

        self.assertEqual(file_name_incremented1_1,
                         file_utils.get_incremented_path(
                             file_name_incremented1, add_index=True))
        self.assertEqual(file_name_incremented2,
                         file_utils.get_incremented_path(
                             file_name_incremented1, add_index=False))

        with open(file_name_incremented1000, 'w') as f:
            f.write('')
        self.assertEqual(file_name_incremented1001,
                         file_utils.get_incremented_path(
                             file_name_incremented1000, add_index=False))

    def test_get_incremented_path_with_existing_index(self):
        _, file_name = tempfile.mkstemp(suffix='-1.ext')
        file_name_incremented1 = file_utils.add_suffix(file_name, '-1')
        file_name_incremented2 = file_utils.add_suffix(file_name, '-2')
        file_name_incremented1000 = file_utils.add_suffix(file_name, '-1000')
        file_name_incremented1001 = file_utils.add_suffix(file_name, '-1001')
        with open(file_name_incremented1, 'w') as f:
            f.write('')
        file_name = file_name_incremented1
        self.assertEqual(file_name_incremented2,
                         file_utils.get_incremented_path(
                             file_name, add_index=False))
        self.assertEqual(file_name_incremented2,
                         file_utils.get_incremented_path(
                             file_name_incremented1, add_index=False))

        with open(file_name_incremented1000, 'w') as f:
            f.write('')
        self.assertEqual(file_name_incremented1001,
                         file_utils.get_incremented_path(
                             file_name_incremented1000, add_index=False))

    def test_add_suffix(self):
        self.assertEqual('abc_suffix',
                         file_utils.add_suffix('abc', '_suffix'))
        self.assertEqual('abc_suffix.ext',
                         file_utils.add_suffix('abc.ext', '_suffix'))
        self.assertEqual(
            os.path.join('tep_dir', 'abc_suffix.ext'),
            file_utils.add_suffix(os.path.join('tep_dir', 'abc.ext'),
                                  '_suffix'))

    def test_remove_suffix(self):
        self.assertEqual('abc',
                         file_utils.remove_suffix('abc_suffix', '_suffix'))
        self.assertEqual('abc.ext',
                         file_utils.remove_suffix('abc_suffix.ext', '_suffix'))
        self.assertEqual('abc_suffix_wrong.ext',
                         file_utils.remove_suffix('abc_suffix_wrong.ext',
                                                  '_suffix'))
        self.assertEqual(
            os.path.join('temp_dir', 'abc.ext'),
            file_utils.remove_suffix(os.path.join('temp_dir', 'abc_suffix.ext'),
                                     '_suffix'))

    def test_remove_prefix(self):
        self.assertEqual('abc',
                         file_utils.remove_prefix('prefix_abc', 'prefix_'))
        self.assertEqual('abc.ext',
                         file_utils.remove_prefix('prefix_abc.ext', 'prefix_'))
        self.assertEqual('wrong_prefix_abc.ext',
                         file_utils.remove_prefix('wrong_prefix_abc.ext',
                                                  'prefix_'))
        self.assertEqual(
            os.path.join('temp_dir', 'abc.ext'),
            file_utils.remove_prefix(os.path.join('temp_dir', 'prefix_abc.ext'),
                                     'prefix_'))

    @staticmethod
    def _create_temp_files(file_names: dict):
        for each_key, each_flist in file_names.items():
            for each_fname in each_flist:
                with open(each_fname, 'w') as f:
                    f.write("")
