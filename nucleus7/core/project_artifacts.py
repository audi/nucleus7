# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Project artifacts
"""

from collections import namedtuple
from functools import wraps
import logging
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

from nucleus7.core import project
from nucleus7.core import project_serializer
from nucleus7.utils import object_utils

_ProjectArtifact = namedtuple(
    "ProjectArtifact",
    ["name", "mode_fn", "artifact", "serializers"])
# pylint: disable=invalid-name
# is not a constant
_ProjectArtifacts = None  # type: Optional[List[_ProjectArtifact]]
# these are type constants, not a class
_ARTIFACTS_SERIALIZER_TYPE = Union[
    List[project_serializer.ArtifactsSerializer],
    project_serializer.ArtifactsSerializer]


# pylint: enable=invalid-name

def add_project_artifact(
        artifact_name: str,
        artifact_attr_name: str,
        add_mode_to_name: bool = True,
        add_name_to_artifact_name: bool = True,
        serializers_to_use: _ARTIFACTS_SERIALIZER_TYPE =
        project_serializer.JSONArtifactsSerializer
) -> Callable:
    """
    Add artifacts to log

    Parameters
    ----------
    artifact_name
        name of the artifact
    artifact_attr_name
        artifact attribute name from self
    add_mode_to_name
        if the mode must be added to the attribute name before serialization
    serializers_to_use
        one or list of serializers to use
    add_name_to_artifact_name
        if the name of the object must be added to the artifact name as a prefix

    Returns
    -------
    wrapper
        wrapped method
    """

    def wrapper(function):
        @wraps(function)
        def wrapped(self, *args, **kwargs):
            # pylint: disable=global-statement
            # this is the control of the project for now
            # pylint: disable=invalid-name
            # this is not a constant
            global _ProjectArtifacts
            _ProjectArtifacts = _ProjectArtifacts or []
            result = function(self, *args, **kwargs)
            artifact = object_utils.recursive_getattr(self, artifact_attr_name)
            if add_mode_to_name:
                mode_fn = lambda: getattr(self, "mode", None)
            else:
                mode_fn = lambda: None

            artifact_name_ = artifact_name
            if add_name_to_artifact_name:
                instance_name = getattr(self, "name", self.__class__.__name__)
                artifact_name_ = "-".join([instance_name, artifact_name])

            artifact = _ProjectArtifact(
                artifact_name_, mode_fn, artifact, serializers_to_use)
            _ProjectArtifacts.append(artifact)
            return result

        return wrapped

    return wrapper


def reset_artifacts():
    """
    Reset artifacts
    """
    # pylint: disable=global-statement
    # this is the control of the project for now
    # pylint: disable=invalid-name
    # this is not a constant
    global _ProjectArtifacts
    _ProjectArtifacts = None


def get_artifacts() -> Optional[List[_ProjectArtifact]]:
    """
    Get all logged artifacts

    Returns
    -------
    ProjectArtifacts
        artifacts with fields [name, mode_fn, artifact, serializers] each
    """
    return _ProjectArtifacts


def serialize_project_artifacts(function: Callable):
    """
    Serialize project artifacts

    Parameters
    ----------
    function
        function to decorate

    Returns
    -------
    wrapper
        wrapped method
    """

    @wraps(function)
    def wrapped(*args, **kwargs):
        _serialize_project_artifacts()
        return function(*args, **kwargs)

    return wrapped


def _serialize_project_artifacts():
    """
    Serialize all the artifacts
    """
    logger = logging.getLogger(__name__)
    artifacts = get_artifacts()
    if not artifacts:
        return

    try:
        artifacts_dir = project.get_active_artifacts_directory()
    except ValueError:
        logger.warning("Artifacts will not be serialized")
        return

    for each_artifact in artifacts:
        artifact_name = each_artifact.name
        artifact_mode = each_artifact.mode_fn()
        if artifact_mode:
            artifact_name = "-".join([artifact_name, artifact_mode])
        project_serializer.serialize_all_serializers(
            artifact_name, artifacts_dir,
            each_artifact.serializers,
            artifact=each_artifact.artifact)
