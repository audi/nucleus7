# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Interfaces to save KPI values
"""
import abc
import json
import logging
import os

import tensorflow as tf

from nucleus7.core.base import BaseClass
from nucleus7.utils import io_utils
from nucleus7.utils import kpi_utils
from nucleus7.utils import mlflow_utils
from nucleus7.utils import nest_utils
from nucleus7.utils import object_utils


class KPISaver(BaseClass):
    """
    Interface to save the KPIs

    Parameters
    ----------
    add_prefix_to_name
        if the prefix of KPI should be added to the saved name
    """

    def __init__(self, add_prefix_to_name=False):
        super(KPISaver, self).__init__()
        self.add_prefix_to_name = add_prefix_to_name
        self._save_target = None

    @property
    def save_target(self):
        """
        Save target like directory or server address etc.
        """
        return self._save_target

    @save_target.setter
    def save_target(self, save_target):
        if self._save_target is None or save_target is None:
            self._save_target = save_target
        else:
            raise ValueError("Cannot set cache_target since it is already set! "
                             "Set it to None before!")

    @abc.abstractmethod
    def save(self, name: str, values: dict):
        """
        Main method to save the values

        Parameters
        ----------
        name
            name of the KPI to save
        values
            kpi values to save
        """


class KPIJsonSaver(KPISaver):
    """
    Saves KPIs to json. First filters them if they are json serializable
    """

    @object_utils.assert_property_is_defined("save_target")
    @object_utils.assert_is_built
    def save(self, name: str, values):
        io_utils.maybe_mkdir(self.save_target)
        save_fname = os.path.join(self.save_target, name + ".json")
        values_filtered, values_filtered_out = kpi_utils.filter_kpi_values(
            values)
        if values_filtered_out:
            logging.info("Following KPI keys will not be stored to json: "
                         "%s", list(values_filtered_out.keys()))
        with open(save_fname, 'w') as file:
            json.dump(values_filtered, file, indent=2, sort_keys=True)


class MlflowKPILogger(KPISaver):
    """
    Logs KPI values to mlflow. First filters them if they are json serializable
    """

    @object_utils.assert_is_built
    def save(self, name: str, values: dict):
        values_filtered, values_filtered_out = kpi_utils.filter_kpi_values(
            values)
        if values_filtered_out:
            logging.info("Following KPI keys will not be stored to mlflow: "
                         "%s", list(values_filtered_out.keys()))
        values_filtered_flatten = nest_utils.flatten_nested_struct(
            values_filtered, '--')
        for each_kpi_name, each_kpi_value in sorted(
                values_filtered_flatten.items()):
            kpi_full_name = "_".join([name, each_kpi_name])
            mlflow_utils.log_metric_to_mlflow(kpi_full_name, each_kpi_value)


class TfSummaryKPISaver(KPISaver):
    """
    Save KPI values to tensorflow summary
    """

    def __init__(self):
        super().__init__()
        self._summary_writer = None
        self._summary_step = 0

    @property
    def summary_writer(self):
        """
        Summary writer to store kpi values
        """
        return self._summary_writer

    @summary_writer.setter
    def summary_writer(self, summary_writer: tf.summary.FileWriter):
        self._summary_writer = summary_writer

    @property
    def summary_step(self):
        """
        Summary step to store the kpi values
        """
        return self._summary_step

    @summary_step.setter
    def summary_step(self, summary_step: int):
        self._summary_step = summary_step

    def save(self, name: str, values: dict):
        kpi_filtered, _ = kpi_utils.filter_kpi_values(
            values, return_flattened=True)
        for kpi_name, kpi_value in kpi_filtered.items():
            tag = "/".join(["KPI", name, kpi_name])
            summary = tf.Summary(value=[tf.Summary.Value(
                tag=tag, simple_value=kpi_value)])
            self.summary_writer.add_summary(summary, self.summary_step)
