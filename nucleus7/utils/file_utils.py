# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Utils to work with files
"""
from functools import partial
import glob
import logging
import os
import re
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union


def get_subdirs(root_dir: str) -> list:
    """
    Get all subdirectories inside of root_dir

    Parameters
    ----------
    root_dir
        name of root directory to search for subdirectories

    Returns
    -------
    subdirectories
        subdirectories

    """
    subdirs = sorted([os.path.join(root_dir, o) for o in os.listdir(root_dir)
                      if os.path.isdir(os.path.join(root_dir, o))])
    return subdirs


def match_file_names(file_names: Union[list, dict],
                     suffixes: Union[dict, None] = None,
                     basename_fn: Union[Callable, dict] = None
                     ) -> Tuple[dict, dict]:
    """
    Arrange file names for all keys from file_list and return them in sorted
    order together with data, which keys are provided for each sample

    If one file from file pairs not exist, it will be skipped.

    >>> file_list = {'image': ['image1', 'image2'],
                     'labels': ['image1_label', 'image3_label']}
    >>> file_list_reassigned, keys_for_fnames = match_file_names(file_list)
    file_list_reassigned = {'image': '[image1'], 'labels': [image1_label]}
    keys_for_fnames = {'image': [1],
                       'labels': [1]}

    Also if file_list is list of subsets and each subset can have
    different keys. In that case it will be resolved as follows

    >>> file_list = [{'image': ['image1', 'image2'],
                     'labels': ['image1_label', 'image3_label']},
                     {'image': ['image10', 'image20'],
                     'labels2': ['image10_label', 'image20_label']}]
    >>> file_list_reassigned, keys_for_fnames = match_file_names(file_list)
    file_list_reassigned = {'image': ['image1', 'image10', 'image20'],
                            'labels': ['image1_label', '', ''],
                            'labels2': ['', 'image10_label', 'image20_label']}
    keys_for_fnames = {'image': [1, 1, 1],
                       'labels': [1, 0, 0],
                       'labels2': [0, 1, 1]}

    Parameters
    ----------
    file_names
        dict with glob expressions or file names corresponding to each key;
        corresponding files for should have same base names except of
        file_suffixes[key], e.g. {base_name}{fname_suffix}{.ext}
    suffixes
        dict of suffixes that will be extracted from fname to match them;
        can have same keys as file_list
    basename_fn
        function to get the matching base name from file name itself; if not
        defined, only base names with suffix will be matched; if is a dict,
        should have same

    Returns
    -------
    file_list_reassigned
        reassigned dict with same keys as file_list but with corresponding
        files is same order
    keys_for_fnames
        dict of lists with all possible keys from file_list and 1's if that
        sample has corresponding file name and key and 0's otherwise; this
        can be used e.g. for masking of losses
    """
    suffixes = suffixes or {}
    file_names = [file_names] if isinstance(file_names, dict) else file_names
    all_keys = list({k for file_names_subset in file_names
                     for k in file_names_subset.keys()})

    file_list_reassigned = {}
    keys_for_fnames = {}

    for file_list_subset in file_names:
        basenames_common, file_names_reassigned_subset = (
            _match_file_names_subset(basename_fn, file_list_subset, suffixes))
        for key in all_keys:
            file_list_reassigned.setdefault(key, [])
            keys_for_fnames.setdefault(key, [])
            if key not in file_names_reassigned_subset:
                file_list_reassigned[key].extend([''] * len(basenames_common))
                keys_for_fnames[key].extend([0] * len(basenames_common))
            else:
                file_list_reassigned[key].extend(
                    file_names_reassigned_subset[key])
                keys_for_fnames[key].extend([1] * len(basenames_common))
    return file_list_reassigned, keys_for_fnames


def get_incremented_path(path: str, add_index: bool = False,
                         separator: str = '-') -> str:
    """
    Get the path to a filename which does not exist by incrementing path.

    Examples
    --------
    >>> get_incremented_path('/etc/issue')
    '/etc/issue-1'
    >>> get_incremented_path('whatever/1337bla.py')
    'whatever/1337bla.py'
    """

    path_without_extension, extension = os.path.splitext(path)
    path_dir, basename = os.path.split(path_without_extension)
    basename, basename_without_index = _get_basename_without_index(
        add_index, basename, separator)
    existing_file_names = get_existing_fnames_with_index(basename_without_index,
                                                         path_dir, extension,
                                                         separator)
    if not existing_file_names:
        non_existing_path = os.path.join(path_dir, basename + extension)
        if os.path.exists(non_existing_path):
            return add_suffix(non_existing_path, '-1')
        return non_existing_path

    new_index = _get_index_from_path(existing_file_names[-1], separator) + 1
    incremented_file_name = "{}{}{}{}".format(
        basename_without_index, separator, new_index, extension)
    incremented_path = os.path.join(path_dir, incremented_file_name)
    return incremented_path


def add_suffix(fname: str, suffix: str) -> str:
    """
    Add suffix fo filename just before extension

    Parameters
    ----------
    fname
        file name to add the suffix
    suffix
        suffix to add

    Returns
    -------
    fname_with_suffix
        file name with suffix

    Examples
    --------
    >>> add_suffix('file/name.ext', '_suffix')
    'file/name_suffix.ext'
    """
    fname_, ext = os.path.splitext(fname)
    return ''.join((fname_, suffix, ext))


def maybe_fnames_from_glob(fnames: Union[str, list]) -> list:
    """
    Restore the file names from glob pattern if needed
    """
    fnames_expr = [fnames] if isinstance(fnames, str) else fnames
    fnames = []
    for expr in fnames_expr:
        fnames.extend(sorted(glob.glob(expr)))
    return fnames


def get_basename_with_depth(fname: Union[str, bytes], path_depth: int
                            ) -> Tuple[str, str]:
    """
    Return basename with depth of fname_depth from full file path

    Parameters
    ----------
    fname
        complete path to file
    path_depth
        how many directories back from the file must be returned

    Returns
    -------
    basename_with_subdirectories
        basename with subdirectories till depth of path_depth

    Example
    -------
    >>> get_basename_with_depth("long/path/to/file.txt", 0)
    "", "file.txt"
    >>> get_basename_with_depth("long/path/to/file.txt", 2)
    "path/to", "file.txt"
    """
    if isinstance(fname, bytes):
        fname = fname.decode()
    fname_splitted = os.path.normpath(fname).lstrip(
        os.path.sep).split(os.path.sep)
    subdir = ("" if path_depth == 0
              else os.path.join(*fname_splitted[-(path_depth + 1):-1]))
    file_name = fname_splitted[-1]
    return subdir, file_name


def match_file_names_with_glob(file_names_with_glob: Dict[str, str],
                               match_fn: Callable[[str, str], str]
                               ) -> Dict[str, List[str]]:
    """
    Match


    Parameters
    ----------
    file_names_with_glob
    match_fn

    Returns
    -------

    """
    file_names_from_glob = {
        each_key: set(glob.glob(each_glob_expression))
        for each_key, each_glob_expression in file_names_with_glob.items()
    }
    file_names_from_glob_and_match_names = {
        each_key: {
            match_fn(each_real_file_name, key=each_key): each_real_file_name
            for each_real_file_name in real_file_names}
        for each_key, real_file_names in file_names_from_glob.items()
    }
    file_names_from_glob_matches = [
        set(match_names) for match_names in
        file_names_from_glob_and_match_names.values()]
    common_match_names = set.intersection(*file_names_from_glob_matches)
    matched_file_names = {each_key: [] for each_key in file_names_with_glob}
    for match_name in common_match_names:
        for each_key in file_names_with_glob:
            real_file_name_for_match_and_key = (
                file_names_from_glob_and_match_names[each_key][match_name])
            matched_file_names[each_key].append(
                real_file_name_for_match_and_key)
    return matched_file_names


def remove_prefix(path: str, prefix: Union[str, None]) -> str:
    """
    Remove the prefix from the file path if file name starts with it

    Parameters
    ----------
    path
        path with prefix
    prefix
        prefix to remove

    Returns
    -------
    name_without_prefix
        name with removed prefix

    """
    if not prefix:
        return path
    directory, basename = os.path.split(path)
    basename, ext = os.path.splitext(basename)
    if basename.startswith(prefix):
        basename = basename.replace(prefix, "", 1)
        path = os.path.join(directory, basename + ext)
    return path


def remove_suffix(path: str, suffix: Union[str, None]) -> str:
    """
    Remove the suffix from the path if file name without extension ends with it

    Parameters
    ----------
    path
        name with prefix
    suffix
        suffix to remove

    Returns
    -------
    name_without_suffix
        name with removed suffix

    """
    if not suffix:
        return path
    directory, basename = os.path.split(path)
    basename, ext = os.path.splitext(basename)
    if basename.endswith(suffix):
        word_split = basename.rsplit(suffix, 1)
        basename = "".join(word_split[:-1])
        path = os.path.join(directory, basename + ext)
    return path


def get_existing_fnames_with_index(basename_without_index: str, path_dir: str,
                                   extension: str = "",
                                   separator: str = '-') -> List[str]:
    """
    Get the file names with index, e.g. all file names with {path}-{index}
    format.

    Parameters
    ----------
    basename_without_index
        base name without index
    extension
        file extension
    path_dir
        path to the directory where to search
    separator
        separator for the index

    Returns
    -------
    existing_file_names
        existing file names sorted in index order
    """
    glob_expr = "{}{}*{}".format(basename_without_index, separator, extension)
    regex_expr = "{}{}[0-9]{{1,}}{}".format(
        basename_without_index, separator, extension)
    existing_file_names = glob.glob(os.path.join(path_dir, glob_expr))
    existing_file_names = [f for f in existing_file_names
                           if re.search(os.path.join(path_dir, regex_expr), f)]

    existing_file_names_sorted = sorted(
        existing_file_names,
        key=partial(_get_index_from_path, separator=separator))
    return existing_file_names_sorted


def _get_basename_without_index(add_index: bool, basename: str, separator: str
                                ) -> Tuple[str, str]:
    basename_without_index = basename
    if not add_index:
        if separator in basename:
            index = basename.split(separator)[-1]
            try:
                _ = int(index)
                basename_without_index = separator.join(
                    basename.split(separator)[:-1])
            except ValueError:
                pass
    else:
        basename = add_suffix(basename, '-1')
    return basename, basename_without_index


def _match_file_names_subset(basename_fn: Callable[[str], str],
                             file_list_subset: dict, suffixes: dict
                             ) -> Tuple[list, dict]:
    def _basename_fn(fname):
        return os.path.basename(os.path.splitext(fname)[0])

    logger = logging.getLogger(__name__)
    basename_fn = basename_fn or _basename_fn

    basename2fname_subset = {}
    fnames_without_suffix_subset = {}
    # TODO (oleksandr.vorobiov@audi.de): rewrite part with key selection
    if isinstance(basename_fn, dict):
        key_basename_subset = [k for k in basename_fn.keys()
                               if k in file_list_subset.keys()]
    else:
        key_basename_subset = None
    if key_basename_subset:
        key_basename_subset = key_basename_subset[0]
    for key, fnames in file_list_subset.items():
        if isinstance(basename_fn, dict):
            key_basename = (key if (key in basename_fn
                                    or not key_basename_subset)
                            else key_basename_subset)
            basename_fn_ = basename_fn.get(key_basename, _basename_fn)
        else:
            basename_fn_ = basename_fn
        suffix = suffixes.get(key, '')
        fnames = maybe_fnames_from_glob(fnames)
        basename2fname_subset[key] = {
            remove_suffix(basename_fn_(f), suffix): f
            for f in fnames}
        fnames_without_suffix_subset[key] = set(
            basename2fname_subset[key].keys())
    basenames_common = sorted(set.intersection(
        *fnames_without_suffix_subset.values()))
    logger.info('Found %s file pairs for keys %s',
                len(basenames_common), list(file_list_subset.keys()))
    file_names_reassigned_subset = {
        k: [basename2fname_subset[k][basename]
            for basename in basenames_common]
        for k in file_list_subset}
    return basenames_common, file_names_reassigned_subset


def _get_index_from_path(path_, separator):
    basename_ = os.path.basename(path_)
    basename_ = os.path.splitext(basename_)[0]
    index_ = int(basename_.split(separator)[-1])
    return index_
