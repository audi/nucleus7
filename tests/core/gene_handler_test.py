# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from functools import partial
from unittest.mock import MagicMock

from absl.testing import parameterized

from nucleus7.core.dna_helix import DNAConnection
from nucleus7.core.gene_handler import GeneHandler
from nucleus7.core.nucleotide import Nucleotide


class _NucleotideWithProcess(Nucleotide):
    exclude_from_register = True

    def process(self_, **inputs):
        pass

    def __call__(self_, **inputs):
        return self_.process(**inputs)


class TestGeneHandler(parameterized.TestCase):

    def test_build(self):
        class GeneHandlerDummy(GeneHandler):
            gene_name_and_nucleotide_super_cls = {
                'gene1': Nucleotide, 'gene2': Nucleotide,
                'gene3': Nucleotide, 'gene4': Nucleotide}

        genes = {
            'gene1': [Nucleotide(name='gene1_nucleotide1',
                                 inbound_nodes=["data"]),
                      Nucleotide(name='gene1_nucleotide2',
                                 inbound_nodes=["data", "gene1_nucleotide1"]),
                      Nucleotide(name='gene1_nucleotide3',
                                 inbound_nodes=["data2", "gene1_nucleotide1"])],
            'gene2': [Nucleotide(name='gene2_nucleotide1',
                                 inbound_nodes=["data"]),
                      Nucleotide(name='gene2_nucleotide2',
                                 inbound_nodes=["gene2_nucleotide1",
                                                "data3", "gene1_nucleotide1"])]}
        gene3 = {'gene3_nucleotide1': Nucleotide(name='gene3_nucleotide1',
                                                 inbound_nodes=[]),
                 'gene3_nucleotide2': Nucleotide(name='gene3_nucleotide2',
                                                 inbound_nodes=[])}
        gene4 = Nucleotide(name='gene4_nucleotide1', inbound_nodes=[])
        gene_handler = GeneHandlerDummy(gene1=genes['gene1'],
                                        gene2=genes['gene2'],
                                        gene3=gene3,
                                        gene4=gene4).build()
        genes1_dict_representation = {n.name: n for n in genes['gene1']}
        genes2_dict_representation = {n.name: n for n in genes['gene2']}
        genes3_dict_representation = gene3
        genes4_dict_representation = {"gene4_nucleotide1": gene4}
        self.assertDictEqual(genes1_dict_representation, gene_handler.gene1)
        self.assertDictEqual(genes2_dict_representation, gene_handler.gene2)
        self.assertDictEqual(genes3_dict_representation, gene_handler.gene3)
        self.assertDictEqual(genes4_dict_representation, gene_handler.gene4)

        self.assertSetEqual({"data", "data2", "data3"},
                            set(gene_handler.inbound_nodes))
        self.assertEqual(3,
                         len(gene_handler.inbound_nodes))

        gene2_repeated = Nucleotide(name='gene1_nucleotide1',
                                    inbound_nodes=[])
        with self.assertRaises(AssertionError):
            _ = GeneHandlerDummy(gene1=genes['gene1'],
                                 gene2=gene2_repeated).build()

    def test_build_dna_and_properties(self):
        class _NucleotideType1(Nucleotide):
            pass

        class _NucleotideType2(Nucleotide):
            pass

        class _NucleotideType3(Nucleotide):
            pass

        class _InputNucleotide(Nucleotide):
            pass

        class _GeneHandlerDummy(GeneHandler):
            gene_name_and_nucleotide_super_cls = {
                'gene1': _NucleotideType1, 'gene2': _NucleotideType2,
                'gene3': _NucleotideType3
            }
            nucleotide_type_dependency_map = {
                _NucleotideType1: [_NucleotideType1],
                _NucleotideType2: [_NucleotideType1, _NucleotideType2],
                _NucleotideType3: [_NucleotideType2, _NucleotideType3]
            }

        incoming_nucleotides = [
            _InputNucleotide(name='input', inbound_nodes=[])]
        genes_right_type_dependency = {
            'gene1': [_NucleotideType1(name='gene1_nucleotide1',
                                       inbound_nodes=['input']),
                      _NucleotideType1(name='gene1_nucleotide2',
                                       inbound_nodes=[]),
                      _NucleotideType1(name='gene1_nucleotide3',
                                       inbound_nodes=[])],
            'gene2': [_NucleotideType2(name='gene2_nucleotide1',
                                       inbound_nodes=['input',
                                                      'gene1_nucleotide1',
                                                      'gene1_nucleotide2']),
                      _NucleotideType2(name='gene2_nucleotide2',
                                       inbound_nodes=['gene2_nucleotide1',
                                                      'gene1_nucleotide3'])],
            'gene3': [Nucleotide(name='gene3_nucleotide1',
                                 inbound_nodes=['gene2_nucleotide1']),
                      Nucleotide(name='gene3_nucleotide2',
                                 inbound_nodes=['gene2_nucleotide2',
                                                'gene3_nucleotide1'])]
        }

        genes_wrong_type_dependency = {
            'gene1': [_NucleotideType1(name='gene1_nucleotide1',
                                       inbound_nodes=['input']),
                      _NucleotideType1(name='gene1_nucleotide2',
                                       inbound_nodes=['input']),
                      _NucleotideType1(name='gene1_nucleotide3',
                                       inbound_nodes=['input'])],
            'gene2': [_NucleotideType2(name='gene2_nucleotide1',
                                       inbound_nodes=['gene1_nucleotide1',
                                                      'gene1_nucleotide2']),
                      _NucleotideType2(name='gene2_nucleotide2',
                                       inbound_nodes=['gene2_nucleotide1',
                                                      'gene1_nucleotide3'])],
            'gene3': [_NucleotideType3(name='gene3_nucleotide1',
                                       inbound_nodes=['gene2_nucleotide1']),
                      _NucleotideType3(name='gene3_nucleotide2',
                                       inbound_nodes=['gene1_nucleotide2',
                                                      # not allowed
                                                      'gene3_nucleotide1'])]
        }
        gene_handler_wrong_dependency = _GeneHandlerDummy(
            **genes_wrong_type_dependency).build()
        with self.assertRaises(ValueError):
            gene_handler_wrong_dependency.build_dna(incoming_nucleotides)

        gene_handler = _GeneHandlerDummy(**genes_right_type_dependency).build()
        gene_handler.build_dna(incoming_nucleotides)

        dna_must = {
            'gene1_nucleotide1': DNAConnection(incoming={'input'},
                                               outgoing={'gene2_nucleotide1'}),
            'gene1_nucleotide2': DNAConnection(incoming=set(),
                                               outgoing={'gene2_nucleotide1'}),
            'gene1_nucleotide3': DNAConnection(incoming=set(),
                                               outgoing={'gene2_nucleotide2'}),
            'gene2_nucleotide1': DNAConnection(incoming={'gene1_nucleotide1',
                                                         'gene1_nucleotide2',
                                                         'input'},
                                               outgoing={'gene2_nucleotide2',
                                                         'gene3_nucleotide1'}),
            'gene2_nucleotide2': DNAConnection(incoming={'gene1_nucleotide3',
                                                         'gene2_nucleotide1'},
                                               outgoing={'gene3_nucleotide2'}),
            'gene3_nucleotide1': DNAConnection(incoming={'gene2_nucleotide1'},
                                               outgoing={'gene3_nucleotide2'}),
            'gene3_nucleotide2': DNAConnection(incoming={'gene2_nucleotide2',
                                                         'gene3_nucleotide1'},
                                               outgoing=set())}
        sorted_nucleotide_names_must = {
            'gene1': ['gene1_nucleotide2', 'gene1_nucleotide3',
                      'gene1_nucleotide1'],
            'gene2': ['gene2_nucleotide1', 'gene2_nucleotide2'],
            'gene3': ['gene3_nucleotide1', 'gene3_nucleotide2']}
        self.assertDictEqual(dna_must, gene_handler.dna_helix.as_dict(False))
        self.assertDictEqual(sorted_nucleotide_names_must,
                             gene_handler.sorted_nucleotide_names)

        inbound_nodes_must = ['input']
        self.assertListEqual(inbound_nodes_must, gene_handler.inbound_nodes)

        inbound_nodes_all_nucleotides_must = {
            'gene1_nucleotide1': ['input'],
            'gene1_nucleotide2': [],
            'gene1_nucleotide3': [],
            'gene2_nucleotide1': ['input',
                                  'gene1_nucleotide1',
                                  'gene1_nucleotide2'],
            'gene2_nucleotide2': ['gene2_nucleotide1',
                                  'gene1_nucleotide3'],
            'gene3_nucleotide1': ['gene2_nucleotide1'],
            'gene3_nucleotide2': ['gene2_nucleotide2',
                                  'gene3_nucleotide1']}
        inbound_nodes_all_nucleotides = (
            gene_handler.get_property_from_all_genes('inbound_nodes'))
        self.assertDictEqual(inbound_nodes_all_nucleotides_must,
                             inbound_nodes_all_nucleotides)

    @parameterized.parameters({'use_additional_kwargs': False,
                               'use_nucleotide_inputs_preprocessing_fn': False},
                              {'use_additional_kwargs': True,
                               'use_nucleotide_inputs_preprocessing_fn': False},
                              {'use_additional_kwargs': True,
                               'use_nucleotide_inputs_preprocessing_fn': True},
                              {'use_additional_kwargs': False,
                               'use_nucleotide_inputs_preprocessing_fn': True})
    def test_process_gene(self, use_additional_kwargs=True,
                          use_nucleotide_inputs_preprocessing_fn=False):

        def side_effect_nucleotide_call(nucleotide: Nucleotide, **inputs):
            return {k: '_'.join([nucleotide.name, k])
                    for k in nucleotide.generated_keys_all}

        def _nucleotide_inputs_preprocessing_fn(nucleotide, inputs):
            inputs = {k: 'pp_' + v for k, v in inputs.items()}
            return inputs

        def _update_kwargs_must_with_preprocessing(kwargs_must):
            kwargs_must = {k: 'pp_' + v for k, v in kwargs_must.items()}
            return kwargs_must

        if use_additional_kwargs:
            additional_kwargs = {'is_training': True}
        else:
            additional_kwargs = None

        if use_nucleotide_inputs_preprocessing_fn:
            nucleotide_inputs_preprocessing_fn = MagicMock(
                side_effect=_nucleotide_inputs_preprocessing_fn)
        else:
            nucleotide_inputs_preprocessing_fn = None

        (gene_handler, gene_inputs, input_node1, input_node2,
         nucleotide1, nucleotide2, nucleotide3) = _get_gene_dummy_handler()

        input_node1.process = MagicMock(return_value=None)
        input_node2.process = MagicMock(return_value=None)

        nucleotide1.process = MagicMock(side_effect=partial(
            side_effect_nucleotide_call, nucleotide=nucleotide1))
        nucleotide2.process = MagicMock(side_effect=partial(
            side_effect_nucleotide_call, nucleotide=nucleotide2))
        nucleotide3.process = MagicMock(side_effect=partial(
            side_effect_nucleotide_call, nucleotide=nucleotide3))

        if use_additional_kwargs:
            result = gene_handler.process_gene(
                gene_name='gene', gene_inputs=gene_inputs,
                nucleotide_inputs_preprocessing_fn=
                nucleotide_inputs_preprocessing_fn,
                **additional_kwargs
            )
        else:
            result = gene_handler.process_gene(
                gene_name='gene', gene_inputs=gene_inputs,
                nucleotide_inputs_preprocessing_fn=
                nucleotide_inputs_preprocessing_fn,
            )

        result_must = {
            'nucleotide1': {'output11': 'nucleotide1_output11',
                            'output12': 'nucleotide1_output12'},
            'nucleotide2': {'output21': 'nucleotide2_output21',
                            'output22': 'nucleotide2_output22'},
            'nucleotide3': {'output31': 'nucleotide3_output31'}
        }
        self.assertDictEqual(result_must, result)

        input_node1.process.assert_not_called()
        input_node2.process.assert_not_called()

        addictional_kwargs = additional_kwargs or {}
        nucleotide1_process_arguments_must = {'input11': 'value11'}
        if use_nucleotide_inputs_preprocessing_fn:
            nucleotide1_process_arguments_must = (
                _update_kwargs_must_with_preprocessing(
                    nucleotide1_process_arguments_must))
        nucleotide1.process.assert_called_once_with(
            **nucleotide1_process_arguments_must, **addictional_kwargs)

        nucleotide2_process_arguments_must = {'input21': 'value21',
                                              'input22': 'nucleotide1_output11'}
        if use_nucleotide_inputs_preprocessing_fn:
            nucleotide2_process_arguments_must = (
                _update_kwargs_must_with_preprocessing(
                    nucleotide2_process_arguments_must))
        nucleotide2.process.assert_called_once_with(
            **nucleotide2_process_arguments_must, **addictional_kwargs)

        nucleotide3_process_arguments_must = {'input31': 'nucleotide1_output11',
                                              'input32': 'nucleotide2_output22'}
        if use_nucleotide_inputs_preprocessing_fn:
            nucleotide3_process_arguments_must = (
                _update_kwargs_must_with_preprocessing(
                    nucleotide3_process_arguments_must))
        nucleotide3.process.assert_called_once_with(
            **nucleotide3_process_arguments_must, **addictional_kwargs)

        if use_nucleotide_inputs_preprocessing_fn:
            self.assertEqual(3, nucleotide_inputs_preprocessing_fn.call_count)

    def test_process_gene_with_empty_results(self):
        def side_effect_nucleotide_call(nucleotide: Nucleotide, **inputs):
            return {k: '_'.join([nucleotide.name, k])
                    for k in nucleotide.generated_keys_all}

        (gene_handler, gene_inputs, input_node1, input_node2,
         nucleotide1, nucleotide2, nucleotide3) = _get_gene_dummy_handler()

        input_node1.process = MagicMock(return_value=None)
        input_node2.process = MagicMock(return_value=None)

        nucleotide1.process = MagicMock(side_effect=partial(
            side_effect_nucleotide_call, nucleotide=nucleotide1))
        nucleotide2.process = MagicMock(return_value=None)
        nucleotide3.process = MagicMock(side_effect=partial(
            side_effect_nucleotide_call, nucleotide=nucleotide3))

        result = gene_handler.process_gene(
            gene_name='gene', gene_inputs=gene_inputs,
        )
        result_must = {
            'nucleotide1': {'output11': 'nucleotide1_output11',
                            'output12': 'nucleotide1_output12'},
            'nucleotide2': None,
            'nucleotide3': None
        }
        self.assertDictEqual(result_must, result)

        input_node1.process.assert_not_called()
        input_node2.process.assert_not_called()

        nucleotide1_process_arguments_must = {'input11': 'value11'}
        nucleotide1.process.assert_called_once_with(
            **nucleotide1_process_arguments_must)

        nucleotide2_process_arguments_must = {'input21': 'value21',
                                              'input22': 'nucleotide1_output11'}
        nucleotide2.process.assert_called_once_with(
            **nucleotide2_process_arguments_must)
        nucleotide3.process.assert_not_called()

    def test_mode_setter(self):
        mode = 'NEW_MODE'

        class GeneHandlerDummy(GeneHandler):
            gene_name_and_nucleotide_super_cls = {
                'gene1': Nucleotide, 'gene2': Nucleotide,
                'gene3': Nucleotide, 'gene4': Nucleotide}

        genes = {
            'gene1': [Nucleotide(name='gene1_nucleotide1', inbound_nodes=[]),
                      Nucleotide(name='gene1_nucleotide2', inbound_nodes=[]),
                      Nucleotide(name='gene1_nucleotide3', inbound_nodes=[])],
            'gene2': [Nucleotide(name='gene2_nucleotide1', inbound_nodes=[]),
                      Nucleotide(name='gene2_nucleotide2', inbound_nodes=[])]}
        gene_handler = GeneHandlerDummy(gene1=genes['gene1'],
                                        gene2=genes['gene2']).build()
        gene_handler.mode = mode
        for gene in genes.values():
            for nucleotide in gene:
                self.assertEqual(nucleotide.mode, mode)

    def test_assert_all_nucleotides_have_same_mode(self):
        mode = 'NEW_MODE'

        class GeneHandlerDummy(GeneHandler):
            gene_name_and_nucleotide_super_cls = {
                'gene1': Nucleotide, 'gene2': Nucleotide,
                'gene3': Nucleotide, 'gene4': Nucleotide}

        genes = {
            'gene1': [Nucleotide(name='gene1_nucleotide1', inbound_nodes=[]),
                      Nucleotide(name='gene1_nucleotide2', inbound_nodes=[]),
                      Nucleotide(name='gene1_nucleotide3', inbound_nodes=[])],
            'gene2': [Nucleotide(name='gene2_nucleotide1', inbound_nodes=[]),
                      Nucleotide(name='gene2_nucleotide2', inbound_nodes=[])]}
        gene_handler = GeneHandlerDummy(gene1=genes['gene1'],
                                        gene2=genes['gene2']).build()
        gene_handler.mode = mode
        gene_handler.assert_all_nucleotides_have_same_mode()
        gene_handler.gene1['gene1_nucleotide1'].mode = 'WrongMode'
        with self.assertRaises(ValueError):
            gene_handler.assert_all_nucleotides_have_same_mode()

    @parameterized.parameters({"with_keys_mapping": True},
                              {"with_keys_mapping": False})
    def test_flatten_results(self, with_keys_mapping):
        gene_handler, *_ = _get_gene_dummy_handler()
        results = {"nucleotide1": {'output11': 11, 'output12': 12},
                   "nucleotide2": {'output21': 21, 'output22': 22},
                   "nucleotide3": {'output31': 31}}

        if with_keys_mapping:
            keys_mapping = {"nucleotide1": {"output11": "output11_rm"},
                            "nucleotide3": {"output31": "output31_rm"}}
        else:
            keys_mapping = None

        results_flat = gene_handler.flatten_results(results, keys_mapping)
        if with_keys_mapping:
            results_flat_must = {'output11_rm': 11, 'output12': 12,
                                 'output21': 21, 'output22': 22,
                                 'output31_rm': 31}
        else:
            results_flat_must = {'output11': 11, 'output12': 12,
                                 'output21': 21, 'output22': 22,
                                 'output31': 31}
        self.assertDictEqual(results_flat_must,
                             results_flat)

    @parameterized.parameters({"with_keys_mapping": True},
                              {"with_keys_mapping": False})
    def test_get_flat_generated_keys(self, with_keys_mapping):
        gene_handler, *_ = _get_gene_dummy_handler()
        gene_handler.gene["nucleotide1"].generated_keys += [
            "_optional_key11", "_optional_key12"]
        gene_handler.gene["nucleotide2"].generated_keys += ["_optional_key2"]
        if with_keys_mapping:
            keys_mapping = {
                "nucleotide1": {"output11": "output11_rm",
                                "optional_key11": "optional_key11_rm"},
                "nucleotide3": {"output31": "output31_rm"}}
        else:
            keys_mapping = None

        flat_keys_required = gene_handler.get_flat_generated_keys_required(
            keys_mapping)
        flat_keys_optional = gene_handler.get_flat_generated_keys_optional(
            keys_mapping)

        if with_keys_mapping:
            flat_keys_required_must = ["output11_rm", "output12", "output21",
                                       "output22", "output31_rm"]
            flat_keys_optional_must = ["optional_key11_rm", "optional_key12",
                                       "optional_key2"]
        else:
            flat_keys_required_must = ["output11", "output12", "output21",
                                       "output22", "output31"]
            flat_keys_optional_must = ["optional_key11", "optional_key12",
                                       "optional_key2"]
        self.assertSetEqual(set(flat_keys_required_must),
                            set(flat_keys_required))
        self.assertSetEqual(set(flat_keys_optional_must),
                            set(flat_keys_optional))


def _get_gene_dummy_handler():
    nucleotide1 = _NucleotideWithProcess(
        name='nucleotide1',
        inbound_nodes=['input_node1'],
        incoming_keys_mapping={
            'input_node1': {
                'output1': 'input11'
            }
        })
    nucleotide1.incoming_keys = ['input11']
    nucleotide1.generated_keys = ['output11', 'output12']
    nucleotide2 = _NucleotideWithProcess(
        name='nucleotide2',
        inbound_nodes=['nucleotide1', 'input_node2'],
        incoming_keys_mapping={
            'nucleotide1': {
                'output11': 'input22',
            },
            'input_node2': {
                'output1': 'input21',
            }
        })
    nucleotide2.incoming_keys = ['input21', '_input22']
    nucleotide2.generated_keys = ['output21', 'output22']
    nucleotide3 = _NucleotideWithProcess(
        name='nucleotide3',
        inbound_nodes=['nucleotide1', 'nucleotide2'],
        incoming_keys_mapping={
            'nucleotide1': {
                'output11': 'input31',
            },
            'nucleotide2': {
                'output22': 'input32',
            }
        })
    nucleotide3.incoming_keys = ['input31', 'input32']
    nucleotide3.generated_keys = ['output31']
    input_node1 = Nucleotide(name='input_node1')
    input_node1.generated_keys = ['output1', 'output2']
    input_node2 = Nucleotide(name='input_node2')
    input_node2.generated_keys = ['output1']
    incoming_nucleotides = [input_node1, input_node2]
    gene_inputs = {'input_node1': {'output1': 'value11',
                                   'output2': 'value12'},
                   'input_node2': {'output1': 'value21'}}
    gene = [nucleotide3, nucleotide1, nucleotide2]
    gene_handler = GeneHandler(gene=gene)
    gene_handler.build()
    gene_handler.build_dna(incoming_nucleotides)
    return (gene_handler, gene_inputs, input_node1, input_node2,
            nucleotide1, nucleotide2, nucleotide3)
