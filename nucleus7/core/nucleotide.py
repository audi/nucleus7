# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Interface for main building block - Nucleotide
"""

from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import tensorflow as tf

from nucleus7.core.base import BaseClass
from nucleus7.core.base import MetaLogAndRegister
from nucleus7.utils import log_utils
from nucleus7.utils import model_utils
from nucleus7.utils import nucleotide_utils
from nucleus7.utils import object_utils
from nucleus7.utils import tf_collections_utils
from nucleus7.utils import tf_utils
from nucleus7.utils import tf_varscopes_utils


class Nucleotide(BaseClass, metaclass=MetaLogAndRegister):
    """
    Class for nucleus nodes

    Parameters
    ----------
    name
        name of the plugin; if not specified, will use class name
    inbound_nodes
        name of the input node
    incoming_keys_mapping
        list of mappings from inbound node output names to node inputs. They
        should ensure connections of corresponding nodes, e.g.
        if node has inputs `['input', 'input2']` and inbound node has outputs
        `['output', 'output2']` then to pass this check, mapping should
        be `{'inbound_node': {'output': 'input', 'output2': 'input2'}}`
        see corresponding implementation of node.
        The mappings should ensure connections of corresponding nodes
    data_format
        mapping between data type and data format

    Attributes
    ----------
    incoming_keys
        list of arguments for `self.__call__` (predict / process) method; used
        for graph connections checks; optional keys should have '_' as prefix
    generated_keys
        list of keys inside of return dict after `self.__call__` method; used
        for graph connections checks; optional keys should have '_' as prefix
    dynamic_incoming_keys
        if the nucleotide has dynamic incoming keys
    dynamic_generated_keys
        if the nucleotide has dynamic incoming keys
    _process_method_name
        name of main method call, like 'predict' for :obj:`ModelPlugin` and
        'process' for :obj:`ModelLoss`
    """
    exclude_from_register = True
    incoming_keys = []  # type: List[str]
    generated_keys = []  # type: List[str]
    dynamic_incoming_keys = False
    dynamic_generated_keys = False
    _process_method_name = 'process'

    def __init__(self,
                 inbound_nodes: Optional[Union[list, dict, str]] = None,
                 name: Union[str, None] = None,
                 incoming_keys_mapping: Union[dict, None] = None,
                 data_format: Union[dict, None] = None):
        super(Nucleotide, self).__init__()

        assert not name or '/' not in name, (
            'Slashes are not allowed in the name!!!')
        self._name = name
        if isinstance(inbound_nodes, str):
            inbound_nodes = [inbound_nodes]
        self.inbound_nodes = inbound_nodes or []
        self.incoming_keys_mapping = incoming_keys_mapping or {}
        if isinstance(inbound_nodes, dict):
            self.incoming_keys_mapping.update(inbound_nodes)
        self.data_format = data_format  # type: dict

        self._mode = None  # type: str
        self._check_input_mappings()
        self._check_input_keys()

    @property
    def defaults(self):
        data_format_default = {'image': 'NHWC',
                               'video': 'NTHWC',
                               'time_series': 'NTC'}
        return {"data_format": data_format_default}

    @property
    def mode(self) -> str:
        """
        Mode of nucleotide, e.g. train, eval, infer

        Returns
        -------
        mode
            mode
        """
        return self._mode

    @mode.setter
    def mode(self, mode: str):
        self._mode = mode

    @property
    def is_training(self):
        """
        If mode == train, e.g. training mode

        Returns
        -------
        is_training_flag
            True if mode == train, else otherwise
        """
        return self._mode == tf.estimator.ModeKeys.TRAIN

    @property
    def name(self):
        """
        Name of the nucleotide or the class name, if no name was provided to
        constructor

        Returns
        -------
        name
            name
        """
        return self._name or self.__class__.__name__

    # pylint: disable=no-self-argument
    # classproperty is class property and so cls must be used
    @object_utils.classproperty
    def incoming_keys_required(cls) -> List[str]:
        """
        Get required incoming keys

        Returns
        -------
        required_incoming_keys
            list of incoming keys without "_" in the name
        """
        return [k for k in cls.incoming_keys if not k.startswith('_')]

    # pylint: disable=no-self-argument
    # classproperty is class property and so cls must be used
    @object_utils.classproperty
    def incoming_keys_optional(cls) -> List[str]:
        """
        Get optional incoming keys without "_" prefix

        Returns
        -------
        optional_incoming_keys
            list of incoming keys with "_" in the name; names are provided
            without "_"
        """
        return [k[1:] for k in cls.incoming_keys if k.startswith('_')]

    # pylint: disable=no-self-argument
    # classproperty is class property and so cls must be used
    @object_utils.classproperty
    def incoming_keys_all(cls) -> List[str]:
        """
        Get all incoming keys; optional keys have no "_" prefix

        Returns
        -------
        incoming_keys
            list of all incoming keys
        """
        return cls.incoming_keys_optional + cls.incoming_keys_required

    # pylint: disable=no-self-argument
    # classproperty is class property and so cls must be used
    @object_utils.classproperty
    def generated_keys_required(cls) -> List[str]:
        """
        Get required generated keys

        Returns
        -------
        required_generated_keys
            list of generated keys without "_" in the name
        """
        return [k for k in cls.generated_keys if not k.startswith('_')]

    # pylint: disable=no-self-argument
    # classproperty is class property and so cls must be used
    @object_utils.classproperty
    def generated_keys_optional(cls) -> List[str]:
        """
        Get optional generated keys without "_" prefix

        Returns
        -------
        optional_generated_keys
            list of generated keys with "_" in the name; names are provided
            without "_"
        """
        return [k[1:] for k in cls.generated_keys if k.startswith('_')]

    # pylint: disable=no-self-argument
    # classproperty is class property and so cls must be used
    @object_utils.classproperty
    def generated_keys_all(cls) -> List[str]:
        """
        Get all generated keys; optional keys have no "_" prefix

        Returns
        -------
        generated_keys
            list of all generated keys
        """
        return cls.generated_keys_optional + cls.generated_keys_required

    @property
    def use_genes_as_inputs(self):
        """
        Returns
        -------
            Flag to identify if the nucleotide takes genes, e.g. dict with
            other nucleotides as keys and dict of particular values as values
            or on just dict with incoming_keys and corresponding values.
            Is used to simulate the GeneHandler behavior.
        """
        return False

    def _check_input_mappings(self):
        """Check if all keys from mapping are inside of inbound nodes"""
        for each_mapping_name in self.incoming_keys_mapping:
            if (each_mapping_name not in self.inbound_nodes
                    and each_mapping_name != nucleotide_utils.ALL_OTHERS_KEY):
                raise AttributeError(
                    "Inputs name mapping node name is not inside of "
                    "inbound nodes!!! "
                    "(Node: {}, mapping: {}, inbound nodes: {})".format(
                        self.name, each_mapping_name,
                        self.inbound_nodes))

    def filter_inputs(self, inputs: dict) -> Optional[dict]:
        """
        Filter all inputs to the nucleotide according to its inbound_nodes and
        incoming keys mapping

        Parameters
        ----------
        inputs

        Returns
        -------
        inputs_filtered
            filtered inputs
        """
        nucleotide_inputs = (
            nucleotide_utils.remap_and_collapse_nucleotide_inputs(
                inputs, self.inbound_nodes, self.incoming_keys_mapping))
        if nucleotide_inputs is None:
            return None

        if not self.dynamic_incoming_keys:
            nucleotide_inputs = {k: v for k, v in nucleotide_inputs.items()
                                 if k in self.incoming_keys_all}
        return nucleotide_inputs

    def __call__(self, *args, **kwargs):
        pass

    def __repr__(self):
        representation = (
            "Nucleotide {} with {} inbound nodes, "
            "{} incoming keys and "
            "{} generated keys"
        ).format(self.name, len(self.inbound_nodes),
                 len(self.incoming_keys_all), len(self.generated_keys_all))
        return representation

    def _check_input_keys(self):
        """Inspect signature of `predict` or `process` method and check
        if all items from `self.incoming_keys` are inside of function
        arguments
        """
        args = nucleotide_utils.get_nucleotide_signature(self)[0]
        for inp_key in self.incoming_keys_all:
            if inp_key not in args:
                raise AttributeError(
                    "Update signature incoming_nodes of node {} according "
                    "to incoming_keys!!! (signature args: {}, "
                    "incoming_keys: {})".format(
                        self.name, args, self.incoming_keys))


class TfNucleotide(Nucleotide,
                   model_utils.KerasLayersMixin,
                   model_utils.DefaultPlaceholderMixin,
                   model_utils.CustomSessionHandlerMixin):
    """
    Class for tensorflow nucleotides

    Parameters
    ----------
    trainable
        if the nucleotide is trainable; if set to False, all variables from this
        nucleotide will be removed from `tf.GraphKeys.TRAINABLE_VARIABLES`
        collection so will not be used for training / updating,
        but gradient will still flow;
    """
    exclude_from_register = True

    def __init__(self, trainable: bool = True, **kwargs):
        super(TfNucleotide, self).__init__(**kwargs)
        self.trainable = trainable
        self._variable_scope = None  # type: Optional[str]
        self._variables = []  # type: List[tf.Variable]

    @property
    def variable_scope(self) -> Optional[str]:
        """
        Returns
        -------
        variable_scope
            variable_scope used in nucleotide
        """
        return self._variable_scope

    @property
    def variables(self) -> List[tf.Variable]:
        """
        Returns
        -------
        variables
            tf variables of nucleotide
        """
        return self._variables

    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """
        All trainable variables in plugin

        Returns
        -------
        trainable_variables
            trainable variables inside of plugin
        """
        if not self.trainable:
            return []

        trainable_from_collection = set(tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, self.variable_scope))
        trainable_variables_plugin = set.intersection(
            trainable_from_collection, set(self.variables))
        non_trainable_keras_vars = {
            each_var for each_layer in self.keras_layers
            for each_var in each_layer.non_trainable_variables}
        trainable_vars = trainable_variables_plugin - non_trainable_keras_vars
        return list(trainable_vars)

    def reset_tf_graph(self):
        """
        Reset tf graph for nucleutide, e.g. remove all the variables and tensors
        from the graph
        """
        self.reset_keras_layers()
        self.remove_all_placeholders()

    def build(self):
        super().build()
        self.create_keras_layers()
        return self

    @object_utils.assert_is_built
    @object_utils.assert_property_is_defined('mode')
    @tf_varscopes_utils.with_named_variable_scope
    @tf_collections_utils.remove_not_trainable_from_collection
    @tf_varscopes_utils.add_var_scope_and_variables
    @tf_utils.replace_outputs_with_named_identity
    @log_utils.log_nucleotide_inputs_outputs()
    def __call__(self, **inputs) -> Dict[str, tf.Tensor]:
        """
        Wrapper on top of self.call with following checks:
            * if object is built
            * if mode property is defined
            * add variables to self._variables
            * use named var scope
            * add var_scope to self._var_scope
            * if nucleotide is not trainable, then remove from
            tf.trainable_variables collection
            * add tf.identity on result

        Parameters
        ----------
        inputs
            inputs to nucleotide

        Returns
        -------
        nucleotide_outputs
            nucleotide outputs
        """
        # pylint: disable=arguments-differ
        # parent __call__ method has more generic signature
        return self._call(**inputs)

    def _call(self, **inputs) -> Dict[str, tf.Tensor]:
        """
        Wrapper of predict or process method

        Parameters
        ----------
        inputs
            inputs to nucleotide

        Returns
        -------
        nucleotide_outputs
            nucleotide outputs

        """
        process_method = getattr(self, self._process_method_name)
        return process_method(**inputs)
