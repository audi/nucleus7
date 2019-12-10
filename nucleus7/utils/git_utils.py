# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Utils to work with git, e.g. obtain the hash
"""

import importlib
import inspect
import os
import subprocess
from typing import Optional
from typing import Union


def get_git_revision_hash_from_dir(cwd: str = None) -> str:
    """
    Get the git hash revision from directory

    Parameters
    ----------
    cwd
        directory with git

    Returns
    -------
    git_hash
        hash of the git
    """
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=cwd).decode().strip()


def get_git_revision_short_hash(cwd: str = None) -> str:
    """
    Get the short git hash revision from directory

    Parameters
    ----------
    cwd
        directory with git

    Returns
    -------
    git_hash
        short hash of the git
    """
    return subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD'], cwd=cwd).decode().strip()


def get_git_revision_hash_from_obj(obj: object) -> Optional[str]:
    """
    Get git revision from the object module

    Parameters
    ----------
    obj
        some object, git of which module will be checked

    Returns
    -------
    git_hash
        hash of the git holding object module
    """
    try:
        obj_path = inspect.getfile(obj.__class__)
    except (AttributeError, TypeError):
        return None
    return get_git_revision_hash_from_module_path(module_path=obj_path)


def get_git_revision_hash_from_module_path(module_path,
                                           default_package='nucleus7'
                                           ) -> Union[str, None]:
    """

    Parameters
    ----------
    module_path
        path to imported module
    default_package
        name of default package of that module to use if the module has no
        __file__ attribute

    Returns
    -------
    git_hash
        ash of the git directory with module
    """
    if module_path is None:
        module_path = ''
    package = '.'.join(module_path.split('.')[:-1]) or default_package
    try:
        package_path = importlib.import_module(package).__file__
    except (AttributeError, ImportError):
        return None
    try:
        cwd = os.path.split(package_path)[0]
        git_hash = get_git_revision_hash_from_dir(cwd)
    except (subprocess.CalledProcessError, NotADirectoryError):
        git_hash = None
    return git_hash
