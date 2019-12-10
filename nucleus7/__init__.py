# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
nucleus7 package for deep learning
"""

import os

from nucleus7 import coordinator
from nucleus7 import data
from nucleus7 import kpi
from nucleus7 import model
from nucleus7.builders.runs_builder import build_infer
from nucleus7.builders.runs_builder import build_train
from nucleus7.core import project_global_config
from nucleus7.core.nucleotide import Nucleotide
from nucleus7.core.nucleotide import TfNucleotide
from nucleus7.core.project_global_config import query_global_parameter
from nucleus7.core.project_global_config import query_nucleotide_global_config
from nucleus7.core.register import *

# pylint: disable=invalid-name
project_root_dir = os.path.abspath(os.path.dirname(__file__))
version_file_name = os.path.join(project_root_dir, "VERSION")
__version__ = open(version_file_name, "r").read().strip()

del os
del project_root_dir
del version_file_name
