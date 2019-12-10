# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Base interfaces for data modules
"""
from nucleus7.core.nucleotide import Nucleotide
from nucleus7.utils import model_utils


class BaseDataPipeNucleotide(Nucleotide,
                             model_utils.CustomSessionHandlerMixin):
    """
    Base Data nucleotide to use with DataReader and DataProcessor
    """
    exclude_from_register = True
    is_tensorflow = False
