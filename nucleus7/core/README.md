Nucleotide and co.
==================

- [Nucleotide](#nucleotide)
- [DNA helix](#dna-helix)
- [GeneHandler](#gene-handler)
- [Buffer intermediate results](#buffer-intermediate-results)

[nucleotide]: ../../docs_source/images/nucleotide.png
[nucleus_structure]: ../../docs_source/images/nucleus_structure.png
[Model]: ../model/README.md
[Coordinator]: ../coordinator/README.md
[data]: ../data/README.md
[kpi]: ../kpi/README.md

## Nucleotide <a name="nucleotide"></a>

Nucleotide can be represented as follows:

![Nucleotide][nucleotide]

and has following interface (simplified):

```python
class Nucleotide:
    incoming_keys = ['input1', '_input2']
    generated_keys = ['output1', '_output2']

    def process(*, input1, input2=None):
        ...
        result = {'output1': ..., 'output2': ...}
```

So each nucleotide has following mandatory fields:

- `incoming_keys` and `generated_keys`:
    * this are basically the signature of the nucleotide - `ìncoming_keys`
    describe the names of input data parameters and `generated_keys` describe
    the output keys and can be of 2 types - required and optional.
    * if the key is optional, add the '_' in the beginning, e.g.
    `_optional_key`, but use it without '_' prefix inside of the process method 
- `process` method:
     - perform calculations on inputs and return outputs; inputs are always
     passed as kwargs and its keys are same as `ìncoming_keys` and outputs is
     a dict (also nested), which keys are equal to `generated_keys`

It is not mandatory that `incoming_keys` or `generated_keys` are defined,
but if `incoming_keys` is not defined, it assumes that nucleotide has no inputs
and if `generated_keys` is not defined, it assumes that nucleotide does not
produce any outputs, but still may perform some calculations.

Also `incoming_keys` and `generated_keys` are class attributes. It means
that they are fixed for one nucleotide implementation.

There are different **Nucleotide** types:

* [Model][Model]
    - ModelPlugin
    - ModelLoss
    - ModelPostProcessor
    - ModelMetric
    - ModelSummary
* [Callback][Coordinator]
    - CoordinatorCallback
* [KPIEvaluator][KPI]
    - KPIPlugin
    - KPIAccumulator
* [Data][data]
    - Dataset
    - DataFeeder

In short, all of nucleotides inside of nucleus7 project are only interfaces and
need to be implemented for particular tasks.

## DNA helix <a name="dna-helix"></a>

![Gene structure][nucleus_structure]

`DNA helix` called the graph constructed of all model nucleotides and also
sorted in topological order. This means that nucleotides cannot be cycled.

Each **Nucleotide** has a list of inbound nodes (`nucleotide.inbound_nodes`)
and a list of key mapping from inbound node generated keys to own incoming keys
(`nucleotide.incoming_keys_mapping`), e.g. Nucleotide1 from image has
following mapping:

```python
Nucleotide1.inbound_nodes = ['Nucleotide2', 'Nucleotide3']
Nucleotide1.incoming_keys_mapping = {
    'Nucleotide2': {'out1': 'inp1', 'out2': 'inp2'},
    'Nucleotide3': {'out2': 'inp2', 'out1': 'inp3'}}
```

These mappings and inbound node list can vary for each instance of same
nucleotide class and can be controlled through its configuration. Mapping has
following rules:

- if no mapping was provided, it assumes that generated key name of inbound node
correspond to incoming key name.

- if you map some key to `_`, then it will be omitted and so not used as input
to nucleotide node process. 

- you can remap all other keys for different nested levels using `*`, e.g.
`{'*': '_', 'a': 'b'}` will redirect all inputs except 'a' to '\_' (omit them)
and `{'a:b:*': '_', 'a:b:c': 'e'}` will redirect all nested levels from `a:b` to
`_` except `a:b:c`, which will be forwarded to `e`. 

- if after the mapping, there are inputs from different inbound nodes but
mapped key, then they will be appended to one list in the order of
inbound nodes.

- it is possible to map multiple output keys (also from different inbound nodes)
to one incoming key and also to pass the output as a key of the inputs dict in
key; following will result in `out2 = [Nucleotide1:out1:key1, Nucleotide2:out1]`
and `out1 = {"key1": Nucleotide1:out2, "key2": Nucleotide2:out2}`:
    ```python
    Nucleotide1.incoming_keys_mapping = {
      "Nucleotide1": {"out1:key1":  "out2:0",
                      "out2":  "out1:key1"},
      "Nucleotide2": {"out1":  "out2:1",
                      "out2":  "out1:key2"}
    }
    ```

Let´s go a bit deeper and see what is happening under the hood in **nucleus7**
during the **dna_helix** building. Pseudo code is following:

```python
def build_dna():
    nucleotides_in_sorted_order = topological_sort(nucleotides)
    results = {}
    for nucleotide in nucleotides_in_sorted_order:
        nucleotide_inputs = select_inputs_from_inbound_nodes(
            results, inbound_nodes)
        nucleotide_inputs_mapped = map_generated_keys_to_incoming_keys(
            nucleotide_inputs, nucleotide.incoming_keys_mapping)
        nucleotide_output = nucleotide(is_training, **nucleotide_inputs_mapped)
        results[nucleotide.name] = nucleotide_output
    return results
```

Each `process` block of nucleotide takes inputs without knowing where are
they coming from (from which nucleotide) and returns a dictionary with
it´s `generated_keys` as keys. That result will be put to main dictionary
holding all results of all nucleotides.

## GeneHandler <a name="gene-handler"></a>

`GeneHandler` takes multiple genes (gene is a set os nucleotides of same kind)
and is responsible for the data flow across them:
- it creates a DNA helix for all the genes
- checks if the dependency of each gene on other gene is allowed
(e.g. ModelLoss cannot depend on ModelMetric)
- sort each gene in topological order if it is possible inside of the gene and
also across all the genes

## Buffer intermediate results <a name="buffer-intermediate-results"></a>

In some cases we want to buffer the sample intermediate results and then use
them only when everything was collected. This may be useful for data
summarization, tracking, KPI accumulation and evaluation etc.
To deal with it in general fashion, class `SamplesBuffer` and `BufferProcessor`
are introduced. Former is the buffer, which only stores the data and cleans it.
Later one is the processor to deal, e.g. to accumulate the data from batches,
split it and then execute some processor function on the buffer inputs.
Refer to particular implementation of Nucleotide for further info on how to
use it.
