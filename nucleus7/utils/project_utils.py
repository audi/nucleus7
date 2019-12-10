# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Utils to work with project
"""

import datetime
import importlib
import json
import logging
import os
import time
from typing import Union

from nucleus7.core import config_logger
from nucleus7.utils import git_utils
from nucleus7.utils import io_utils
from nucleus7.utils import nest_utils
from nucleus7.utils import run_utils

_CONFIG_LIST_INSERT_IDENTIFIER = "__BASE__"
_CONFIG_DICT_UPDATE_IDENTIFIER = "__UPDATE_CONFIG__"
_PROJECT_META_FILE_NAME = "nucleus7_project.json"
_PROJECT_NAME_KEY = "PROJECT_NAME"


class ProjectLock:
    """
    Class to lock directory

    Can be used to ensure that multiple processes will perform operations
    sequentially.

    Parameters
    ----------
    lock_dir
        path where to store lockfile
    lockname
        file name of lock file
    """

    def __init__(self, lock_dir: str, lockname: str = 'lockfile'):
        self._lock_fname = os.path.join(lock_dir, lockname)

    def lock_or_wait(self):
        """
        Create lock file if it does not exist or wait until lock file
        is removed
        """
        logger = logging.getLogger(__name__)
        try:
            os.mkdir(self._lock_fname)
            logger.info('Lock file %s created', self._lock_fname)
        except FileExistsError:
            logger.info('Lock file %s found, wait 0.5 sec', self._lock_fname)
            time.sleep(0.5)
            self.lock_or_wait()

    def release(self):
        """
        Remove the lock file
        """
        logger = logging.getLogger(__name__)
        logger.info('Lock file %s released', self._lock_fname)
        os.rmdir(self._lock_fname)


def collect_project_meta_info() -> dict:
    """
    Collect the meta information of the project including run type, git hashes
    of imported objects etc.

    Returns
    -------
    project_meta
        dict holding meta information of the project for this run
    """
    meta = dict()
    meta['HOSTNAME'] = os.environ.get('HOSTNAME')
    meta['run_start_time'] = datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S")
    cluster_spec = run_utils.get_cluster_spec()
    meta['CLUSTER_SPEC'] = cluster_spec
    meta['USER'] = os.environ.get('USER')
    source_versions = get_source_versions()
    meta['source_versions'] = source_versions
    return meta


def collect_and_add_project_meta_info(project_dir: str):
    """
    Collect the meta information of the project including run type, git hashes
    of imported objects etc. and add it to existing meta information file inside
    of project directory

    Parameters
    ----------
    project_dir
        project directory
    """
    meta_fname = os.path.join(project_dir, _PROJECT_META_FILE_NAME)
    if not os.path.isfile(meta_fname):
        meta_all = {}
    else:
        meta_all = io_utils.load_json(meta_fname)
    project_name = get_project_name_from_directory(project_dir)
    meta_all.setdefault(_PROJECT_NAME_KEY, project_name)
    meta_run = collect_project_meta_info()
    for k in meta_run:
        if k in meta_all:
            meta_all[k] = meta_all[k] + [meta_run[k]]
        else:
            meta_all[k] = [meta_run[k]]
    with open(meta_fname, 'w') as file:
        json.dump(meta_all, file, indent=4, sort_keys=True)


def add_runtype_to_project_meta_info(project_dir: str):
    """
    Add current runtype to the project meta information json

    Parameters
    ----------
    project_dir
        project directory
    """
    meta_fname = os.path.join(project_dir, _PROJECT_META_FILE_NAME)
    if not os.path.isfile(meta_fname):
        meta = {}
    else:
        meta = io_utils.load_json(meta_fname)
    used_runs = meta.get('runs', [])
    runtype = _get_or_increase_runtype(used_runs)
    meta['runs'] = used_runs + [runtype]
    with open(meta_fname, 'w') as file:
        json.dump(meta, file, indent=4, sort_keys=True)


def get_source_versions() -> dict:
    """
    Return the source versions for all the modules used to construct
    nucleus7 objects

    Returns
    -------
    source_versions
        dict with package names as keys and its version or git has as value
    """
    logged_configs = config_logger.get_logged_configs()
    configs_flatten = nest_utils.flatten_nested_struct(logged_configs)
    class_names = {v for k, v in configs_flatten.items()
                   if "class_name" in k.split('//')}
    source_versions = {"nucleus7": _get_package_version("nucleus7")}

    for each in class_names:
        package = each.split('.')[0]
        if package not in source_versions:
            package_version = _get_package_version(package)
            source_versions[package] = package_version
    return source_versions


def log_exported_model_info(path_to_log: str, global_step: int):
    """
    Log the exported model information to file "model.info"

    Parameters
    ----------
    path_to_log
        path to log directory to store the file
    global_step
        global step of the export

    """
    global_step_fname = os.path.join(path_to_log, 'model.info')
    with open(global_step_fname, 'w') as file:
        file.write("global_step={:d}".format(global_step))


def update_config_with_other_config(config1: dict, config2: dict) -> dict:
    """
    Update one config with other

    Parameters
    ----------
    config1
        base config, e.g. config that will be updated
    config2
        config tha will update

    Returns
    -------
    updated_config
        updated config
    """
    updated_config = {}
    updated_config.update(config1)
    for each_key_config2, each_subconfig2 in config2.items():
        subconfig_updated = _update_subconfig(
            config1.get(each_key_config2, None), each_subconfig2)
        updated_config[each_key_config2] = subconfig_updated
    return updated_config


def get_project_name_from_directory(project_dir: str) -> str:
    """
    Returns project name given project directory

    Parameters
    ----------
    project_dir
        project directory

    Returns
    -------
    project_name
        project name
    """
    project_name = None
    project_meta_file_name = os.path.join(project_dir, _PROJECT_META_FILE_NAME)
    if os.path.exists(project_meta_file_name):
        project_meta = io_utils.load_json(project_meta_file_name)
        project_name = project_meta.get(_PROJECT_NAME_KEY)
    return project_name or os.path.split(project_dir.rstrip('/'))[-1]


def _update_subconfig(config1: Union[list, dict], config2: Union[list, dict]
                      ) -> Union[list, dict]:
    if not config1:
        return config2
    if not config2:
        return config1
    if _define_subconfig_type(config1, config2) == "list":
        return _update_list_subconfigs(config1, config2)
    return _update_dict_subconfigs(config1, config2)


def _define_subconfig_type(config1, config2) -> str:
    if any((isinstance(config1, list), isinstance(config2, list))):
        return "list"
    return "dict"


def _update_list_subconfigs(config1: list, config2: list) -> list:
    if not isinstance(config1, (list, tuple)):
        config1 = [config1]
    if not isinstance(config2, (list, tuple)):
        config2 = [config2]

    if _CONFIG_LIST_INSERT_IDENTIFIER in config1:
        config1 = list(config1)
        config1.remove(_CONFIG_LIST_INSERT_IDENTIFIER)

    try:
        base_index = config2.index(_CONFIG_LIST_INSERT_IDENTIFIER)
    except ValueError:
        base_index = -1
    if base_index == -1:
        return config2

    config_updated = list(config2)
    config_updated.remove(_CONFIG_LIST_INSERT_IDENTIFIER)
    for each_item in config1[::-1]:
        config_updated.insert(base_index, each_item)
    return config_updated


def _update_dict_subconfigs(config1: dict, config2: dict) -> dict:
    if config2.get(_CONFIG_DICT_UPDATE_IDENTIFIER):
        config_updated = {}
        config_updated.update(config1)
        config_updated.update(config2)
        del config_updated[_CONFIG_DICT_UPDATE_IDENTIFIER]
        return config_updated
    return config2


def _get_package_version(package_name: str) -> str:
    module = importlib.import_module(package_name)
    try:
        return module.__version__
    except AttributeError:
        return git_utils.get_git_revision_hash_from_module_path(
            None, package_name)


def _get_or_increase_runtype(used_runs):
    runtype = run_utils.get_run_type()
    if runtype in used_runs:
        same_run_types = [t for t in used_runs if runtype in t]
        run_nums_continued = [t.split('/')[1:] for t in same_run_types]
        run_num = max([int(t[0]) if t else 0 for t in run_nums_continued]) + 1
        runtype = "{}/{}".format(runtype, run_num)
    return runtype
