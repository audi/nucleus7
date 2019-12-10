# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Project directory structure

nucleus7 project has following structure:

project_dir:

    - training:

        - configs - configuration for the training (must be specified to start
          the training)
        - artifacts - artifacts of the training, e.g. dna, file_lists etc.
        - callbacks - callbacks can write here
        - summaries - tensorflow summaries

    - inference:

        - configs - global configuration for the inference; values from
          run-{}/configs ca override it
        - run-{}:

            - configs - configs for this run - if not specified, configs from
              inference/configs will be used
            - artifacts - artifacts of the inference, e.g. dna, file_lists etc.
            - results - results of the inference, e.g. callbacks write here

    - saved_models - saved models in SavedModels format
    - checkpoints - checkpoints like meta graph and the weights
    - (optional, if no mlflow URI was specified) mlruns - mlflow tracking uri

"""
from collections import namedtuple
import glob
import logging
import os
import pathlib
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Set
import warnings

from nucleus7.core import project_global_config
from nucleus7.utils import file_utils
from nucleus7.utils import io_utils
from nucleus7.utils import project_utils
from nucleus7.utils import run_utils

_TrainerDirs = namedtuple('TrainerDirs',
                          ['configs', 'artifacts', 'callbacks', 'summaries',
                           'checkpoints', 'saved_models'])

_InferenceDirs = namedtuple('InferenceDirs',
                            ['configs', 'artifacts', 'callbacks',
                             'checkpoints', 'saved_models'])

_KPIDirs = namedtuple('KPIDirs',
                      ['configs', 'artifacts', 'callbacks'])

_DataExtractDirs = namedtuple('DataExtractDirs',
                              ['configs', 'artifacts', 'callbacks'])

_NEW_RUN_FLAG = "NEW_RUN"
_RUN_DIR_IDENTIFIER = "run"
_TRAINING_DIR = "training"
_INFERENCE_DIR = "inference"
_KPI_DIR = "kpi_evaluation"
_DATA_EXTRACT_DIR = "data_extraction"
_LAST_RUN_NAME = "last_run"
_CONFIGS_DIR = "configs"

_TRAINING_FILE_NAMES_WITH_CONFIGS = [
    'datasets',
    'model',
    'trainer',
]

_TRAINING_DIRECTORIES_WITH_CONFIGS = [
    'callbacks',
    'callbacks_eval',
    'callbacks_train',
    'kpi',
    'losses',
    'metrics',
    'plugins',
    'postprocessors',
    'summaries',
]

_INFERENCE_FILE_NAMES_WITH_CONFIGS = [
    'datafeeder',
    'inferer',
]
_INFERENCE_DIRECTORIES_WITH_CONFIGS = [
    'callbacks',
    'kpi',
]

_KPI_FILE_NAMES_WITH_CONFIGS = [
    'datafeeder',
]
_KPI_DIRECTORIES_WITH_CONFIGS = [
    'callbacks',
    'kpi',
]

_DATA_EXTRACT_FILE_NAMES_WITH_CONFIGS = [
    'datafeeder',
]
_DATA_EXTRACT_DIRECTORIES_WITH_CONFIGS = [
    'callbacks',
]

_CONFIG_MAIN_FILE_NAME = "config_main.json"


class ProjectDirs:
    """
    Class holding the sub directory structure for different project types

    Attributes
    ----------
    TRAINER
        trainer directory structure
    INFERER
        inferer directory structure
    KPI
        kpi directory structure
    DATA_EXTRACT
        data extraction directory structure
    """
    # pylint: disable=too-few-public-methods
    # it is only a container, so no methods required
    TRAINER = _TrainerDirs(
        configs=os.path.join(_TRAINING_DIR, _CONFIGS_DIR),
        summaries=os.path.join(_TRAINING_DIR, 'summaries'),
        callbacks=os.path.join(_TRAINING_DIR, 'callbacks'),
        artifacts=os.path.join(_TRAINING_DIR, 'artifacts'),
        checkpoints='checkpoints',
        saved_models='saved_models',
    )
    INFERER = _InferenceDirs(
        configs=os.path.join(_INFERENCE_DIR, _CONFIGS_DIR),
        callbacks=os.path.join(_INFERENCE_DIR, _NEW_RUN_FLAG, 'results'),
        artifacts=os.path.join(_INFERENCE_DIR, _NEW_RUN_FLAG, 'artifacts'),
        checkpoints='checkpoints',
        saved_models='saved_models',
    )
    KPI = _KPIDirs(
        configs=os.path.join(_KPI_DIR, _CONFIGS_DIR),
        callbacks=os.path.join(_KPI_DIR, _NEW_RUN_FLAG, 'results'),
        artifacts=os.path.join(_KPI_DIR, _NEW_RUN_FLAG, 'artifacts'),
    )
    DATA_EXTRACT = _DataExtractDirs(
        configs=os.path.join(_DATA_EXTRACT_DIR, _CONFIGS_DIR),
        callbacks=os.path.join(_DATA_EXTRACT_DIR, _NEW_RUN_FLAG, 'extracted'),
        artifacts=os.path.join(_DATA_EXTRACT_DIR, _NEW_RUN_FLAG, 'artifacts'),
    )


def create_trainer_project_dirs(
        project_dir: str,
        continue_training: bool = False) -> _TrainerDirs:
    """
    Create project directories for training if needed and check if training
    project already exists

    Create following directories under project_dir:

    - training:

        - artifacts - artifacts of the training, e.g. dna, file_lists etc.
        - callbacks/{train/eval} - callbacks can write here
        - summaries/{train/eval} - tensorflow summaries

    - saved_models - saved models in timestamp
      subfolders and inference_graph.meta together with
      input_output_names.json
    - checkpoints - checkpoints like meta graph and the weights
    - (optional, if no mlflow URI was specified) mlruns - mlflow tracking uri

    Parameters
    ----------
    project_dir
        path to project directory
    continue_training
        if the training should be continued in same project folder; if no
        project directories exist, it will have no effect, otherwise if this set
        to False, no training will be started

    Returns
    -------
    training_dirs
        trainer directories

    Raises
    ------
    FileExistsError
        if this run was already used and exist inside of nucleus7_project.json
        file under runs key
    FileExistsError
        if project_dir has other content as
    """
    lock = project_utils.ProjectLock(project_dir)
    lock.lock_or_wait()
    io_utils.maybe_mkdir(os.path.join(project_dir, _TRAINING_DIR))
    try:
        _validate_training_project(project_dir, continue_training)
    # pylint: disable=invalid-name
    # is common practice to call exceptions as e
    except Exception as e:
        lock.release()
        raise e

    training_dirs = _create_project_directories(project_dir,
                                                ProjectDirs.TRAINER)
    for each_mode in ["train", "eval"]:
        io_utils.maybe_mkdir(os.path.join(training_dirs.summaries, each_mode))
        io_utils.maybe_mkdir(os.path.join(training_dirs.callbacks, each_mode))
    project_utils.collect_and_add_project_meta_info(project_dir)
    project_utils.add_runtype_to_project_meta_info(project_dir)
    lock.release()
    return training_dirs


def create_inference_project_dirs(project_dir: str,
                                  continue_last: bool = False,
                                  run_name: Optional[str] = None,
                                  ) -> _InferenceDirs:
    """
    Create project directories for inference if needed and check if inference
    project already exists

    Generates following project structure inside of project_dir:

    - inference:

        - configs - global configuration for the inference; values from
          run-{}/configs ca override it
        - run-{}:

            - artifacts - artifacts of the inference, e.g. dna, file_lists etc.
            - results - results of the inference, e.g. callbacks write here

    run folder will be selected as following:
        - if there are no run- subfolders inside of the inference folder
          (or inference folder does not exist), it will create a run-1 folder
        - if there is a run-N folder:

            - if it is empty, it will be used
            - if there are any other files and folders inside without _ prefix,
              a new run-{N+1} folder will be created

    Parameters
    ----------
    project_dir
        path to project directory
    run_name
        name of the run; if not specified, then incremented run-N folder is
        used; if specified, must contain at most config folder
    continue_last
        if last project must be continued

    Returns
    -------
    inference_dirs
        inference directories
    """
    inference_dirs = _create_project_dirs_with_run_subfolders(
        project_dir, _INFERENCE_DIR, ProjectDirs.INFERER,
        run_name=run_name, continue_last=continue_last)
    return inference_dirs


def create_kpi_project_dirs(project_dir: str,
                            continue_last: bool = False,
                            run_name: Optional[str] = None,
                            ) -> _KPIDirs:
    """
    Create project directories for kpi calculation if needed and check if
    kpi project already exists

    Generates following project structure inside of project_dir:

    - kpi_evaluation:

        - configs - global configuration for the kpi calculation; values from
          run-{}/configs can override it
        - run-{}:

            - artifacts - artifacts of the kpi calculation, e.g. dna,
              file_lists etc.
            - results - results of the kpi calculation, e.g. callbacks and kpi
              evaluator write here

    run folder will be selected as following:
        - if there are no run- subfolders inside of the kpi_evaluation folder
          (or kpi_evaluation folder does not exist), it will create a
          run-1 folder
        - if there is a run-N folder:

            - if it is empty, it will be used
            - if there are any other files and folders inside without _ prefix,
              a new run-{N+1} folder will be created

    Parameters
    ----------
    project_dir
        path to project directory
    run_name
        name of the run; if not specified, then incremented run-N folder is
        used; if specified, must contain at most config folder
    continue_last
        if last project must be continued

    Returns
    -------
    kpi_dirs
        kpi calculation directories
    """
    kpi_dirs = _create_project_dirs_with_run_subfolders(
        project_dir, _KPI_DIR, ProjectDirs.KPI,
        run_name=run_name, continue_last=continue_last)
    return kpi_dirs


def create_data_extraction_project_dirs(project_dir: str, run_name: str,
                                        continue_last: bool = False
                                        ) -> _DataExtractDirs:
    """
    Create project directories inside of run_name directory. If this directory
    does not exist, it will be created otherwise it should be empty or contain
    only configs folder.

    Generates following project structure inside of project_dir:

    - data_extraction:

        - configs - global configuration for the data extraction; values from
          {run_name}/configs can override it
        - {run_name}}:

            - artifacts - artifacts of the kpi calculation, e.g. dna,
              file_lists etc.
            - extracted - extracted data of the kpi calculation, e.g. callbacks
              write here

    Parameters
    ----------
    project_dir
        path to project directory
    run_name
        name of the data extraction run
    continue_last
        if last project must be continued

    Returns
    -------
    data_extraction_dirs
        kpi calculation directories
    """
    _verify_data_extraction_run_name(
        project_dir, run_name, continue_last=continue_last)

    project_structure = _update_project_structure_with_run_dir(
        run_name, ProjectDirs.DATA_EXTRACT)
    data_extraction_dirs = _create_project_directories(
        project_dir, project_structure)
    return data_extraction_dirs


def read_train_configs_from_directories(
        project_dir: str,
        verify_unused_configs: bool = False) -> dict:
    """
    Read the training configuration from project directory and its configuration
    subdirectory.

    Parameters
    ----------
    project_dir
        project directory
    verify_unused_configs
        if there are other then single_config_fnames json files or directories
        other than multiple_config_fnames inside of project_dir and its name
        starts not with '_', error will be raised

    Returns
    -------
    train_config
        read configuration

    Raises
    ------
    FileExistsError
        if verify_unused_config = True and
        if there are other then single_config_fnames json files or directories
        other than multiple_config_fnames inside of project_dir and its name
        starts not with '_', error will be raised

    """
    config_dir = os.path.join(project_dir, ProjectDirs.TRAINER.configs)
    if not os.path.exists(config_dir) or not os.listdir(config_dir):
        config_dir_realpath = os.path.realpath(config_dir)
        project_dir_realpath = os.path.realpath(project_dir)
        msg = ("Configs folder {} not found inside of {} project directory! "
               "Please include it to start the training"
               ).format(os.path.relpath(config_dir_realpath,
                                        project_dir_realpath),
                        project_dir_realpath)
        raise FileNotFoundError(msg)

    if "augmenter.json" in os.listdir(config_dir):
        raise ValueError("Augmenter is deprecated! Use DataPipe with "
                         "RandomAugmentationTf instead!")

    train_config = _read_configs_from_directory(
        config_dir, _TRAINING_FILE_NAMES_WITH_CONFIGS,
        _TRAINING_DIRECTORIES_WITH_CONFIGS,
        verify_unused_configs=verify_unused_configs)
    project_global_config.add_global_project_config_from_project_dir(config_dir)
    return train_config


def read_inference_configs_from_directories(
        project_dir: str,
        run_name: Optional[str] = None,
        verify_unused_configs: bool = False,
        additional_dir_with_configs: Optional[str] = None,
        continue_last: bool = False) -> dict:
    """
    Read the inference configuration from project inference directory and the
    run subdirectory

    Parameters
    ----------
    project_dir
        project directory
    run_name
        name of the run; if not specified, then incremented run-N folder is
        used; if specified, must contain at most config folder
    verify_unused_configs
        if there are other then single_config_fnames json files or directories
        other than multiple_config_fnames inside of project_dir and its name
        starts not with '_', error will be raised
    additional_dir_with_configs
        path to additional configs to use
    continue_last
        if last project must be continued

    Returns
    -------
    inference_config
        read configuration

    Raises
    ------
    FileNotFoundError
        if there are no configs folder inside of inference subfolder or inside
        of inference/run-{valid new N} or folder exist but are empty

    """
    run_subfolder = _select_inference_run_subfolder(
        project_dir, _INFERENCE_DIR, ProjectDirs.INFERER,
        run_name=run_name,
        continue_last=continue_last)
    configs_dirs_ro = _create_config_dirs_ro(
        project_dir, _INFERENCE_DIR, run_subfolder, additional_dir_with_configs)
    inference_config = _read_configs_from_dirs_with_ro(
        configs_dirs_ro,
        single_config_fnames=_INFERENCE_FILE_NAMES_WITH_CONFIGS,
        multiple_config_fnames=_INFERENCE_DIRECTORIES_WITH_CONFIGS,
        verify_unused_configs=verify_unused_configs)
    return inference_config


def read_kpi_configs_from_directories(
        project_dir: str,
        run_name: Optional[str] = None,
        verify_unused_configs: bool = False,
        additional_dir_with_configs: Optional[str] = None,
        continue_last: bool = False) -> dict:
    """
    Read the kpi configuration from project kpi_evaluation directory and the
    run subdirectory

    Parameters
    ----------
    project_dir
        project directory
    run_name
        name of the run; if not specified, then incremented run-N folder is
        used; if specified, must contain at most config folder
    verify_unused_configs
        if there are other then single_config_fnames json files or directories
        other than multiple_config_fnames inside of project_dir and its name
        starts not with '_', error will be raised
    additional_dir_with_configs
        path to additional configs to use
    continue_last
        if last project must be continued

    Returns
    -------
    kpi_config
        read configuration

    Raises
    ------
    FileNotFoundError
        if there are no configs folder inside of kpi_evaluation subfolder or
        inside of kpi_evaluation/run-{valid new N} or folder exist but are
        empty

    """
    run_subfolder = _select_inference_run_subfolder(
        project_dir, _KPI_DIR, ProjectDirs.KPI, run_name=run_name,
        continue_last=continue_last)
    configs_dirs_ro = _create_config_dirs_ro(
        project_dir, _KPI_DIR, run_subfolder, additional_dir_with_configs)
    kpi_config = _read_configs_from_dirs_with_ro(
        configs_dirs_ro,
        single_config_fnames=_KPI_FILE_NAMES_WITH_CONFIGS,
        multiple_config_fnames=_KPI_DIRECTORIES_WITH_CONFIGS,
        verify_unused_configs=verify_unused_configs)
    return kpi_config


def read_data_extraction_configs_from_directories(
        project_dir: str,
        run_name: str,
        verify_unused_configs: bool = False,
        additional_dir_with_configs: Optional[str] = None,
        continue_last: bool = False) -> dict:
    """
    Read the data extraction configuration from project data_extraction
    directory and the run subdirectory

    Parameters
    ----------
    project_dir
        project directory
    run_name
        name of the data extraction run
    verify_unused_configs
        if there are other then single_config_fnames json files or directories
        other than multiple_config_fnames inside of project_dir and its name
        starts not with '_', error will be raised
    additional_dir_with_configs
        path to additional configs to use
    continue_last
        if last project must be continued

    Returns
    -------
    data_extraction_config
        read configuration

    Raises
    ------
    FileNotFoundError
        if there are no configs folder inside of data_extraction subfolder or
        inside of data_extraction/run-{valid new N} or folder exist but are
        empty

    """
    _verify_data_extraction_run_name(project_dir, run_name,
                                     continue_last=continue_last)
    configs_dirs_ro = _create_config_dirs_ro(
        project_dir, _DATA_EXTRACT_DIR, run_name, additional_dir_with_configs)
    data_extraction_config = _read_configs_from_dirs_with_ro(
        configs_dirs_ro,
        single_config_fnames=_DATA_EXTRACT_FILE_NAMES_WITH_CONFIGS,
        multiple_config_fnames=_DATA_EXTRACT_DIRECTORIES_WITH_CONFIGS,
        verify_unused_configs=verify_unused_configs)
    return data_extraction_config


def _create_config_dirs_ro(project_dir: str, subdir: str,
                           run_subfolder: Optional[str] = None,
                           additional_config_dir: Optional[str] = None):
    configs_dirs_ro = []
    configs_dir_project = os.path.join(project_dir, subdir, _CONFIGS_DIR)
    if os.path.isdir(configs_dir_project):
        configs_dirs_ro.append(configs_dir_project)
    if run_subfolder:
        configs_dir_run = os.path.join(
            project_dir, subdir, run_subfolder, _CONFIGS_DIR)
        if os.path.isdir(configs_dir_run):
            configs_dirs_ro.append(configs_dir_run)
    if additional_config_dir and os.path.isdir(additional_config_dir):
        configs_dirs_ro.append(additional_config_dir)
    _verify_at_least_one_valid_single_config_dir(configs_dirs_ro)
    return configs_dirs_ro


def _select_inference_run_subfolder(project_dir, subdir,
                                    project_structure,
                                    run_name: Optional[str] = None,
                                    continue_last=False):
    logger = logging.getLogger(__name__)
    allowed_content_for_run = ["configs"]
    if continue_last:
        allowed_content_for_run.extend(
            _get_allowed_content_for_run_to_continue(project_structure))
    run_subfolder = _select_run_subfolder(
        project_dir, subdir, run_name=run_name,
        allowed_content_for_run=allowed_content_for_run)
    logger.info("Use %s run", run_subfolder)
    return run_subfolder


def _verify_at_least_one_valid_single_config_dir(config_dirs):
    for each_config_dir in config_dirs:
        if os.path.exists(each_config_dir) and os.listdir(each_config_dir):
            return

    msg = "No configs were found inside of {}".format(config_dirs)
    raise FileNotFoundError(msg)


def _read_configs_from_dirs_with_ro(config_dirs: list,
                                    single_config_fnames,
                                    multiple_config_fnames,
                                    verify_unused_configs):
    config = {}
    for each_config_dir in config_dirs:
        if not each_config_dir or not os.path.isdir(each_config_dir):
            continue

        config_from_dir = _read_configs_from_directory(
            each_config_dir, single_config_fnames,
            multiple_config_fnames,
            verify_unused_configs=verify_unused_configs)
        config = project_utils.update_config_with_other_config(
            config, config_from_dir)
        project_global_config.add_global_project_config_from_project_dir(
            each_config_dir)
    return config


def _create_project_dirs_with_run_subfolders(project_dir, subdir,
                                             project_dirs_structure,
                                             run_name: Optional[str] = None,
                                             continue_last=False):
    logger = logging.getLogger(__name__)
    io_utils.maybe_mkdir(project_dir)
    lock = project_utils.ProjectLock(project_dir)
    lock.lock_or_wait()
    if continue_last:
        allowed_content_for_run = _get_allowed_content_for_run_to_continue(
            project_dirs_structure)
    else:
        allowed_content_for_run = ["configs"]
    run_subfolder = _select_run_subfolder(
        project_dir, subdir, run_name=run_name,
        allowed_content_for_run=allowed_content_for_run)
    logger.info("Use %s run", run_subfolder)
    project_structure = _update_project_structure_with_run_dir(
        run_subfolder, project_dirs_structure)
    kpi_dirs = _create_project_directories(project_dir, project_structure)
    dir_with_runs = os.path.join(project_dir, subdir)
    _add_symlink_for_last_run(os.path.join(dir_with_runs, run_subfolder))
    lock.release()
    return kpi_dirs


def _get_allowed_content_for_run_to_continue(project_dirs_structure):
    allowed_content_for_run = [
        each_dir.split(_NEW_RUN_FLAG)[-1].lstrip(os.path.sep)
        for each_dir in project_dirs_structure
        if _NEW_RUN_FLAG in each_dir
    ]
    return allowed_content_for_run


def _validate_training_project(project_dir, continue_training):
    project_meta_fname = os.path.join(project_dir, 'nucleus7_project.json')
    runtype = run_utils.get_run_type()
    if not os.path.isfile(project_meta_fname):
        used_runs = []
    else:
        used_runs = io_utils.load_json(project_meta_fname).get('runs', [])
    msg_project_exist = ("Project inside {} already exists! "
                         "Clean the folder, set --continue flag or "
                         "select new project folder!".format(project_dir))
    if (run_utils.is_chief()
            and runtype in used_runs
            and not continue_training):
        raise FileExistsError(msg_project_exist)
    if (run_utils.is_chief()
            and not continue_training
            and os.path.exists(project_dir)):
        training_dir = os.path.join(project_dir, _TRAINING_DIR)
        allowed_training_content = ["configs", 'global_config.json']
        if _get_not_allowed_content(training_dir, allowed_training_content):
            raise FileExistsError(msg_project_exist)


def _verify_unused_configs(config_dir: str, single_config_fnames: list,
                           multiple_config_fnames: list):
    dir_content = os.listdir(config_dir)
    for each_item in dir_content:
        path_full = os.path.join(config_dir, each_item)
        if os.path.isfile(path_full):
            fnames_should = (single_config_fnames +
                             ['project_global_config'])
            if (os.path.splitext(each_item)[0] not in fnames_should
                    and not each_item.startswith('_')
                    and not each_item == _CONFIG_MAIN_FILE_NAME):
                raise FileExistsError(
                    "Found unused configuration '{}'! "
                    "remove it or use '_' prefix to set it as a "
                    "temp file!".format(each_item))
        elif os.path.isdir(path_full):
            if (each_item not in multiple_config_fnames
                    and not each_item.startswith('_')):
                raise FileExistsError(
                    "Found unused configuration directory '{}'! "
                    "remove it or use '_' prefix to set it as a "
                    "temp directory!".format(each_item))


def _read_configs_from_directory(config_dir: str,
                                 single_config_fnames: list,
                                 multiple_config_fnames: list,
                                 verify_unused_configs: bool = False) -> dict:
    if verify_unused_configs:
        _verify_unused_configs(
            config_dir, single_config_fnames, multiple_config_fnames)

    config_main = _read_config_main(config_dir)
    configs_from_files = _get_configs_from_single_files(
        config_dir, single_config_fnames)
    configs_from_directories = _get_configs_from_directories(
        config_dir, multiple_config_fnames)

    config = project_utils.update_config_with_other_config(
        config_main, configs_from_files)
    config = project_utils.update_config_with_other_config(
        config, configs_from_directories)
    return config


def _read_config_main(config_dir: str) -> dict:
    config_main = {}
    config_main_path = os.path.join(config_dir, _CONFIG_MAIN_FILE_NAME)
    if os.path.exists(config_main_path):
        config_main = io_utils.load_json(config_main_path)
    return config_main


def _get_configs_from_directories(config_dir: str,
                                  multiple_config_fnames: list) -> dict:
    logger = logging.getLogger(__name__)
    configs = {}
    for dirname in multiple_config_fnames:
        full_dir_path = os.path.join(config_dir, dirname)
        if not os.path.isdir(full_dir_path):
            continue
        config_fnames = sorted(glob.glob(os.path.join(full_dir_path, '*.json')))
        if not config_fnames:
            continue
        config_name = dirname + '_config'
        logger.info('Found %s with configs %s', config_name, config_fnames)
        configs[config_name] = _read_components_configs(config_fnames)
    return configs


def _get_configs_from_single_files(config_dir: str,
                                   single_config_fnames: list) -> dict:
    logger = logging.getLogger(__name__)
    configs = {}
    for fname in single_config_fnames:
        config_fname = os.path.join(config_dir, fname + '.json')
        if os.path.isfile(config_fname):
            config_name = fname + '_config'
            logger.info('Found %s with config %s', config_name, config_fname)
            configs[config_name] = io_utils.maybe_load_json(config_fname)
    return configs


def _read_components_configs(config_fnames: list) -> list:
    """
    Read multiple config files

    Parameters
    ----------
    config_fnames
        list with file names with config json files

    Returns
    -------
    config_components
        list of read configurations from single files
    """
    config_components = []
    for config_fname in config_fnames:
        config_component = io_utils.maybe_load_json(config_fname)
        if isinstance(config_component, list):
            config_components.extend(config_component)
        else:
            config_components.append(config_component)
    return config_components


def _create_project_directories(
        project_dir: str,
        project_structure: NamedTuple) -> NamedTuple:
    project_dirs = {}
    for k, project_structure_subdir in project_structure._asdict().items():
        subdirecotry = _create_project_subdirectory(
            project_dir, project_structure_subdir)
        pathlib.Path(os.path.join(subdirecotry)).mkdir(
            parents=True, exist_ok=True)
        project_dirs[k] = subdirecotry

    return project_structure.__class__(**project_dirs)


def _create_project_subdirectory(
        project_dir: str, subdirectory: str) -> str:
    path_to_subdirectory = os.path.join(project_dir, subdirectory)
    return path_to_subdirectory


def _update_project_structure_with_run_dir(
        run_subfolder: str, project_structure: NamedTuple) -> NamedTuple:
    project_structure_as_dict = project_structure._asdict()
    for each_key, each_path in project_structure_as_dict.items():
        path_split = each_path.split(sep=os.sep)
        if _NEW_RUN_FLAG in path_split:
            run_flag_ind = path_split.index(_NEW_RUN_FLAG)
            new_run_path = os.path.join(
                *path_split[:run_flag_ind], run_subfolder,
                *path_split[run_flag_ind + 1:])
            project_structure = project_structure._replace(
                **{each_key: new_run_path})
    return project_structure


def _select_run_subfolder(
        project_dir: str, run_dir: str,
        run_name: Optional[str] = None,
        allowed_content_for_run: Optional[List[str]] = None) -> str:
    allowed_content_for_run = allowed_content_for_run or []
    path_to_runs = os.path.join(project_dir, run_dir)
    if run_name:
        run_folder_full_path = os.path.join(path_to_runs, run_name)
        if not os.path.exists(run_folder_full_path):
            return run_name

        not_allowed_subdirs = _get_not_allowed_content(run_folder_full_path,
                                                       allowed_content_for_run)
        if not_allowed_subdirs:
            msg = ("Folder {} has not allowed content {}. Please remove it "
                   "or select other run"
                   ).format(run_folder_full_path, not_allowed_subdirs)
            raise ValueError(msg)
        return run_name

    all_runs = file_utils.get_existing_fnames_with_index(
        os.path.join(path_to_runs, _RUN_DIR_IDENTIFIER), project_dir)
    if not all_runs:
        run_folder = '-'.join([_RUN_DIR_IDENTIFIER, "1"])
        return run_folder

    run_folder_full_path = all_runs[-1]
    if _get_not_allowed_content(run_folder_full_path, allowed_content_for_run):
        run_folder_full_path = file_utils.get_incremented_path(
            os.path.join(project_dir, run_folder_full_path))
    run_folder = os.path.relpath(run_folder_full_path, path_to_runs)
    return run_folder


def _get_not_allowed_content(
        path: str, allowed_content: Optional[List[str]] = None,
        exclude_prefix="_") -> Optional[Set[str]]:
    if allowed_content is not None:
        path_content = os.listdir(path)
        if exclude_prefix:
            path_content = [item for item in path_content
                            if not item.startswith(exclude_prefix)]

        not_allowed_content = set(path_content).difference(set(allowed_content))
        if not_allowed_content:
            return not_allowed_content
    return None


def _add_symlink_for_last_run(run_dir: str):
    path_to_runs, last_run_subdir = os.path.split(run_dir)
    link_path = os.path.join(path_to_runs, _LAST_RUN_NAME)
    try:
        if os.path.exists(link_path):
            os.remove(link_path)
    # pylint: disable=invalid-name
    # is common practice to call exceptions as e
    except (IsADirectoryError,) as e:
        msg = "Cannot create symlink to last run. Error traceback: %s" % e
        warnings.warn(msg)
        return
    try:
        os.symlink(last_run_subdir, link_path)
    # pylint: disable=invalid-name
    # is common practice to call exceptions as e
    except OSError as e:
        msg = "Cannot create symlink to last run. Error traceback: %s" % e
        warnings.warn(msg)


def _verify_data_extraction_run_name(project_dir, run_name, continue_last):
    run_directory = os.path.join(project_dir, _DATA_EXTRACT_DIR, run_name)
    if os.path.isdir(run_directory):
        if continue_last:
            not_allowed_subdirs = []
        else:
            not_allowed_subdirs = [
                os.path.split(ProjectDirs.DATA_EXTRACT.callbacks)[-1],
                os.path.split(ProjectDirs.DATA_EXTRACT.artifacts)[-1]]
        existing_subdirs = os.listdir(run_directory)
        not_allowed_subdirs = set.intersection(set(existing_subdirs),
                                               set(not_allowed_subdirs))
        if not_allowed_subdirs:
            msg = ("data extraction directory {} contains not allowed "
                   "directories {}!. Please remove them or use new run_name"
                   ).format(run_name, not_allowed_subdirs)
            raise ValueError(msg)
