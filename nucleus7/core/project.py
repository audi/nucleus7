# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
nucleus7 Project
"""

import logging
from typing import NamedTuple
# pylint: disable=unused-import
# is used for type declaration
from typing import Optional
from typing import Union

from nucleus7.core import project_dirs

# pylint: enable=unused-import
# pylint: disable=invalid-name
_ActiveProject = None  # type: Optional[Project]


# pylint: enable=invalid-name

class Project:
    """
    nucleus7 Project

    Parameters
    ----------
    project_type
        type of the project, train or inference
    project_dir
        project directory
    continue_last
        if the project must be continued, e.g. same directories should be reused
    """

    # pylint: disable=too-few-public-methods
    # project is not intended to used public
    def __init__(self, project_type: str, project_dir: str,
                 continue_last: bool = False,
                 run_name: Optional[str] = None):
        assert project_type in ["train", "infer", "kpi", "data_extraction"], (
            "project type must be one of [train, infer, kpi, data_extraction]!")
        if run_name:
            assert project_type != "train", (
                "run_name is not allowed for train project!")
        self.project_type = project_type
        self.project_dir = project_dir
        self.continue_last = continue_last
        self.run_name = run_name
        self.project_dirs = None  # type: Union[_TrainerDirs, _InferenceDirs]
        self._logger = logging.getLogger(__name__)
        self._validate_active_project()
        self._create_dirs()
        self._set_active_project()

    @property
    def entry_point(self):
        """
        Returns
        -------
        entry_point
            entry point for project
        """
        if self.project_type == "train":
            return "nc7-train"
        if self.project_type == "infer":
            return "nc7-infer"
        if self.project_type == "kpi":
            return "nc7-evaluate_kpi"
        return "nc7-extract_data"

    def _create_dirs(self):
        if (get_active_project() is None
                or _ActiveProject.project_type != self.project_type
                or _ActiveProject.project_dir != self.project_dir):
            self.project_dirs = self._create_dirs_for_type()

    def _validate_active_project(self):
        if get_active_project() is None:
            return
        if self.project_type not in ["train", "infer", "kpi",
                                     "data_extraction"]:
            raise NotImplementedError(
                "Project can be only train or inference type")

    def _set_active_project(self):
        if self.project_dirs is not None:
            # pylint: disable=global-statement
            # this is the control of the project for now
            # pylint: disable=invalid-name
            # this is not a constant
            global _ActiveProject
            _ActiveProject = self

    def _create_dirs_for_type(self) -> NamedTuple:
        """
        Create the directories structure according the the type of the project

        Returns
        -------
        project_structure
            named tuple holding the directories of the project
        """
        if self.project_type == "train":
            self._logger.info("Create trainer project inside of %s",
                              self.project_dirs)
            return project_dirs.create_trainer_project_dirs(
                project_dir=self.project_dir,
                continue_training=self.continue_last)
        if self.project_type == "infer":
            self._logger.info("Create inference project inside of %s",
                              self.project_dirs)
            return project_dirs.create_inference_project_dirs(
                project_dir=self.project_dir, run_name=self.run_name,
                continue_last=self.continue_last)
        if self.project_type == "kpi":
            self._logger.info("Create kpi evaluation project inside of %s",
                              self.project_dirs)
            return project_dirs.create_kpi_project_dirs(
                project_dir=self.project_dir, run_name=self.run_name,
                continue_last=self.continue_last)
        if self.project_type == "data_extraction":
            self._logger.info("Create data extraction project inside of %s",
                              self.project_dirs)
            return project_dirs.create_data_extraction_project_dirs(
                project_dir=self.project_dir, run_name=self.run_name,
                continue_last=self.continue_last)
        raise ValueError("Wrong project_type {}!".format(self.project_type))


def create_or_get_active_project(project_type: str, project_dir: str,
                                 continue_last: bool = False,
                                 run_name: Optional[str] = None) -> Project:
    """
    Create new project or get active one

    Parameters
    ----------
    project_type
        type of the project, train or inference
    project_dir
        project directory
    continue_last
        if the project must be continued, e.g. same directories should be reused
    run_name
        run name for the data_extraction project

    Returns
    -------
    ActiveProject
        active project
    """
    Project(project_type=project_type,
            project_dir=project_dir,
            continue_last=continue_last,
            run_name=run_name)
    return _ActiveProject


def get_active_project() -> Optional[Project]:
    """
    Get active project

    Returns
    -------
    active_project
        active project if exists
    """
    return _ActiveProject


def get_active_project_dirs() -> NamedTuple:
    """
    Get activation project directories

    Returns
    -------
    project_dirs
        project directories

    Raises
    ------
    ValueError
        if no project was created
    """
    _raise_if_no_active_project()
    return _ActiveProject.project_dirs


def get_active_artifacts_directory():
    """
    Get artifacts directory

    Returns
    -------
    artifacts_directory
        artifacts directory

    Raises
    ------
    ValueError
        if no project was created
    """
    _raise_if_no_active_project()
    return get_active_project_dirs().artifacts


def reset_project():
    """
    Reset project
    """
    # pylint: disable=global-statement
    # this is the control of the project for now
    # pylint: disable=invalid-name
    # this is not a constant
    global _ActiveProject
    _ActiveProject = None


def _raise_if_no_active_project():
    if _ActiveProject is None:
        raise ValueError("No active project!")
