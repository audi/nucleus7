#!/bin/bash
# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

WD="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
BIN_DIR="${WD}/bin"

echo Activate nucleus7 local installation inside ${WD}

echo 1. Add ${WD} to PYTHONPATH
export PYTHONPATH=${WD}:${PYTHONPATH}

echo 2. Add ${BIN_DIR} to PATH
export PATH=${BIN_DIR}:${PATH}

echo 3. Make scripts from ${BIN_DIR} executable
find ${BIN_DIR} -type f -exec chmod 750 -- {} +

echo Done
