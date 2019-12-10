# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Coordinator interface
"""
import abc
import os
from typing import Dict
from typing import Optional
from typing import Union

# pylint: disable=cyclic-import
from nucleus7.coordinator.callbacks_handler import CallbacksHandler
# pylint: enable=cyclic-import
from nucleus7.core.base import BaseClass
from nucleus7.core.base import MetaLogAndRegister
from nucleus7.core.dna_helix import DNAHelix
from nucleus7.utils import object_utils

# pylint: disable=invalid-name
# these are type constants, not a class
_CALLBACKS_HANDLER_TYPE = Union[CallbacksHandler,
                                Dict[str, CallbacksHandler]]


# pylint: enable=invalid-name


class Coordinator(BaseClass, metaclass=MetaLogAndRegister):
    """
    Class for coordination of execution of models

    Parameters
    ----------
    project_dir
        project directory to store checkpoints, callback results, summaries etc.
    run_config
        run configuration for coordinator
    callbacks_handler
        callback handler holding the callbacks to be executed after each
        iteration
    session_config
        session configuration passed to `tf.Session(config=session_config)`
    """
    exclude_from_register = True

    def __init__(self, project_dir: str, run_config, *,
                 callbacks_handler: Optional[_CALLBACKS_HANDLER_TYPE] = None,
                 session_config: Union[dict, None] = None):
        super().__init__()
        self.project_dir = project_dir
        self.run_config = run_config
        self.callbacks_handler = callbacks_handler
        self.session_config = session_config
        self.project_dirs = None

    @property
    def dna_helices(self) -> Optional[Dict[str, DNAHelix]]:
        """
        Get the DNA helix of the project

        Returns
        -------
        dna_helices
            dict of dna helix for different coordinator modes
        """
        return None

    @property
    def defaults(self):
        session_config_default = {"log_device_placement": False,
                                  "allow_soft_placement": True,
                                  "gpu_options": {"allow_growth": True}}
        return {"session_config": session_config_default}

    def build(self):
        self._maybe_add_empty_callbacks_handler()
        return super().build()

    @object_utils.assert_is_built
    @abc.abstractmethod
    def run(self):
        """
        Main method to run the coordinator
        """

    def set_callback_properties(self, mode: str, log_dir: str,
                                number_iterations_per_epoch: int):
        """
        Set properties on all callbacks

        Parameters
        ----------
        mode
            mode
        log_dir
            log directory for callbacks
        number_iterations_per_epoch
            number of iterations per epoch
        """
        if isinstance(self.callbacks_handler, dict):
            callbacks_handler = self.callbacks_handler[mode]
        else:
            callbacks_handler = self.callbacks_handler
        callbacks_handler.mode = mode
        callbacks_handler.log_dir = log_dir
        callbacks_handler.number_iterations_per_epoch = (
            number_iterations_per_epoch)

    @object_utils.assert_is_built
    def visualize_project_dna(self, verbosity: int = 0,
                              save_as_artifact: bool = True):
        """
        Visualize the project dna helix for all modes and save it to current
        artifacts folder with name "dna_helix_{mode}.pdf".

        Parameters
        ----------
        verbosity
            how verbose should be visualization
        save_as_artifact
            if the file should be saved to artifacts folder

        Returns
        -------
        subplot
            interactive subplot with dna helix on it

        Raises
        ------
        figure_with_dna_helix
            subplot with dna helix

        See Also
        --------
        :obj:`DNAHelix.visualize`
        """
        dna_helices = self.dna_helices
        if dna_helices is None:
            raise ValueError(
                "Coordinator has no DNA Helix, so nothing to visualize!")
        for each_mode, each_dna_helix in dna_helices.items():
            self._visualize_project_dna_for_mode(
                each_dna_helix, each_mode, verbosity=verbosity,
                save_as_artifact=save_as_artifact)

    def _visualize_project_dna_for_mode(self, dna_helix: DNAHelix,
                                        mode: str, verbosity: int = 0,
                                        save_as_artifact: bool = True):
        if save_as_artifact:
            save_dir = self.project_dirs.artifacts
            save_file_name = "dna_helix_{}_(verbosity_{}).pdf".format(
                mode, verbosity)
            save_path = os.path.join(save_dir, save_file_name)
            if os.path.exists(save_path):
                return
        else:
            save_path = None
        project_name = os.path.basename(self.project_dir)
        title_prefix = "PROJECT: {}; MODE: {}\n".format(project_name, mode)
        dna_helix.visualize(
            save_path=save_path, verbosity=verbosity, title_prefix=title_prefix)

    def _maybe_add_empty_callbacks_handler(self):
        if self.callbacks_handler is None:
            self.callbacks_handler = CallbacksHandler(
                callbacks=[]).build()
        elif isinstance(self.callbacks_handler, dict):
            for each_mode in self.callbacks_handler:
                if self.callbacks_handler[each_mode] is None:
                    self.callbacks_handler[each_mode] = CallbacksHandler(
                        callbacks=[]).build()
