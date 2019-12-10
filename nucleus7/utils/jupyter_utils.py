# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
File containing utility functions for jupyter notebooks
If you just run this file, it will convert build_and_train.ipynb to a
.py-file which can be used for debugging
"""
import os
from subprocess import call


def convert_jupyter_to_python(path_in: str) -> str:
    """
    Convert jupyter notebook to python code

    Parameters
    ----------
    path_in
        path to jupyter notebook

    Returns
    -------
    path_to_python_code
        path to python code
    """
    command_base = [
        "jupyter-nbconvert",
        "--to",
        "script",
        "--log-level",
        "ERROR",
        path_in
    ]
    call(command_base)
    fn_wo_ext = os.path.splitext(path_in)[0]
    out_path = fn_wo_ext + '.py'
    return out_path


def read_file_and_execute(path_in, delete=False, prefix=None, exec_dir=None):
    """
    Read file and execute it

    Parameters
    ----------
    path_in
        path to file to execute
    delete
        flag if the file must be deleted after read
    prefix
        prefix to add to the file before execution
    exec_dir
        directory to execute
    """
    with open(path_in, 'r') as file:
        file_content = file.read()
    if delete:
        os.remove(path_in)
    if prefix is not None:
        file_content = prefix + '\n\n' + file_content
    cwd = os.getcwd()
    if exec_dir is not None:
        os.chdir(exec_dir)
    # TODO(johannes.dumler@audi.de) replace use of exec with more safe one
    # pylint: disable=exec-used
    # is legacy and needs to be changed - just not to block the pylint
    exec(file_content, globals(), globals())
    # pylint: enable=exec-used
    if exec_dir is not None:
        os.chdir(cwd)
