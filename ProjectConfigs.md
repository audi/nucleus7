Project configs
===============

- [General case](#general-case)
- [Combining multiple configs of same gene](#combining-multiple-configs-of-same-gene)
- [Datasets config](#datasets-config)
    - [DataPipe datasets](#data-pipe-datasets)
    - [DatasetMix config](#dataset-mix-config)
- [DataPipe data feeders](#data-pipe-data-feeders)
- [FileList config](#file-list-config)
- [DataFilter config](#data-filter-config)
- [Specify model parameters during inference](#specify-model-parameters-during-inference)
- [Structure of config_main.json](#structure-of-config-main)
- [Project global config](#project-global-config)
- [Configs update rule](#configs-update-rule)

[ProjectStructure]: ./ProjectStructure.md
[data]: ./nucleus7/data/README.md
[Development]: ./Development.md

## General case <a name="general-case"></a>

To control the nucleus7 objects, json config files are used.

Every nucleus7 object can be controlled through the corresponding `config.json`
file, which values will be passed to its constructor. If it must be a
named tuple config, you need to specify it as a dict with corresponding
key-value mapping.

Every nucleotide object can (and some MUST) have following fields:

```json
{
  "class_name": "name.of.the.class.to.import",
  "name": "arbitrary but unique name for constructed object",
  "inbound_nodes": ["list of names of inbount nodes"],
  "incoming_keys_mapping": {
    "Inbound node 1": {"output_key1": "input_key2"},
    "Inbound node 2": {"output_key1": "input_key3"}
  }
}
```

For classes like `Model`, `Trainer` and `Inferer` it can be omitted, you can
specify the "class_name" if you want to use your own implementation and for
`Inferer` you can specify the `model_incoming_keys_mapping` instead of
`incoming_keys_mapping`, which will map the keys from data feeder to the loaded
model.

This configs must be placed in order described inside of
[project structure][ProjectStructure] for particular run and have specified names for
single configs, e.g. there can be only 1 trainer, so its config must be saved
inside of 'trainer.json', but we can have multiple plugins or callbacks, so
they must be saved in specified folder but have arbitrary file names.

## Combining multiple configs of same gene <a name="combining-multiple-configs-of-same-gene"></a>

If you want to combine multiple configs of same object to build the gene, you
can do it as follows:

1. Create folder of the gene name in plural, e.g. for plugin gene create the
folder plugins and for callback gene create callbacks folder
2. Place the configs of singe nucleotides or of combinations there (you can mix
it in difference files):
    * if it is a singe nucleotide, just place it to the "some_name.json" file
    * if it is a list of nucleotides, use the list of their configs inside of
    "some_name.json" file

## Datasets config <a name="datasets-config"></a>

Dataset is unique part of the training and must be specified for both train and
eval. So we use one file - `datasets.json` and place their configurations in
following format:

```json
{
  "train": {
    "class_name": "...",
    "...": "..."
  },
  "eval": {
    "class_name": "...",
    "...": "..."
  }
}
```

### DataPipe datasets <a name="data-pipe-datasets"></a>

To use the `DataPipe` pipeline inside of dataset, you need to specify it as
follows:

```json
{
  "train": {
    "data_pipe": [
      {"class_name":  "reader1",
       "reader_kwarg1": "...",
       "reader_kwarg2": "..."},
      "..."
    ],
    "output_keys_mapping": {"..."},
    "...": "..."
  },
  "eval": {
    "data_pipe": [
      {"class_name":  "reader1",
       "reader_kwarg1":  "...",
       "reader_kwarg2":  "..."},
      "..."
    ],
    "output_keys_mapping": {"..."},
    "...": "..."
  }
}
```

`class_name` for dataset may be omitted - it will select the appropriate
class from `DataPipe` itself.

### DatasetMix config <a name="dataset-mix-config"></a>

To know what it is, please refer to [data module][data]

To use the `DatasetMix` in your configs, you just need to combine the dataset 
configs in a list and optionally add the "sampling_weight" key to the single
dataset configs (if you do not set it, it will be sampling_weight = 1):

```json
{
  "train": 
   [
    {
      "class_name": "...",
      "other_parameters": "...",
      "sample_weight": 0.1
    },
    {
      "class_name": "...",
      "other_parameters": "...",
      "sample_weight": 0.9
    }
  ],
  "eval": 
  [
    {
      "class_name": "...",
      "other_parameters": "..."
    },
    {
      "class_name": "...",
      "other_parameters": "..."
    }
  ]
}
```

Caveat: sample_weight for eval datasets will no have any effect since on the
evaluation stage the uniform sampling is used.

## DataPipe data feeders <a name="data-pipe-data-feeders"></a>

To use the `DataPipe` pipeline inside of data feeder, you need to specify it as
follows:

```json
{
  "data_pipe": [
    {
      "class_name":  "reader1",
      "reader_kwarg1": "...",
      "reader_kwarg2": "..."
    },
    "..."
  ],
  "output_keys_mapping": {
    "...":  "..."
  },
  "...": "..."
}
```

`class_name` for dataset may be omitted - it will select the appropriate
class from `DataPipe` itself.

## FileList config <a name="file-list-config"></a>

To know what it is, please refer to [data module][data]

`FileList` has no standalone config.json, but must be included inside of
config.json of the object, where it should be placed:

```json
{
  "class_name": "...",
  "file_list": {
    "class_name": "...",
    "file_names": {
      "key1":  "glob/pattern/*/to/search/key1",
      "key2":  "glob/pattern/*/to/search/key2"
    }
  }
}
```
All of the parameters inside of `file_list` section correspond to the
[ProjectConfigs][ProjectConfigs] policy.

Or if you have multiple `FileList` objects and want to use them you can combine
them to a list of file_lists:

```json
{
  "class_name": "...",
  "file_list": [
    {
      "class_name": "...",
      "file_names": {
        "key1":  "glob/pattern/*/to/search/key1",
        "key2":  "glob/pattern/*/to/search/key2"
      }
    },
    {
      "class_name": "...",
      "file_names": {
        "key1":  "other/glob/pattern/*/to/search/key1",
        "key3":  "other/glob/pattern/*/to/search/key2"
      }
    },
    "..."
  ]
}
```

In list case, it will create all the ["key1", "key2", "key3"], but using empty
values for the not presenting keys:

```python
file_names_combined = {
  "key1": ["value1_1", "value1_2", "value2_1", "value2_2", "value2_3"],
  "key2": ["value1_1", "value1_2",         "",         "",         ""],
  "key3": [        "",         "", "value2_1", "value2_2", "value2_3"]
}

``` 

## DataFilter config <a name="data-filter-config"></a>

DataFilter doesn't have the standalone config file too, but it can be
included inside of configs of all data objects. 
To include `DataFilter` inside of config files, just add following to
the field of object with filter:

```json
{
  "data_filter": [
    {
      "class_name": "...",
      "parameter1": "..."
    }
  ]
}
```

## Specify model parameters during inference <a name="specify-model-parameters-during-inference"></a>

If plugins or postprocessors of your model have default placeholders inside
(see [here][Development]), you can specify it using the "model_parameters" key
inside of "inferer.json":

```json
{
  "model_parameters": {
    "nucleotide_name_with_parameter": {
      "parameter_name1": "parameter_value1",
      "parameter_name2": "parameter_value2"
    }
  }
}
```

## Structure of config_main.json <a name="structure-of-config-main"></a>

You can also specify the `config_main.json` file which can include all the
configs for all nucleotides inside. You need to use the same style as before
but you need to add put the individual configurations under the corresponding
keys:

```json
{
  "datasets_config": {
    "train": {
      "...": "..."
    },
    "eval": {
      "...": "..."
    }
  },
  "augmenter_config": {
    "...": "..."
  },
  "plugins_config": [
    {"...": "..."},
    "..."
  ],
  "...": "..."
}
```

So there are following rules:

* if you have "config.json", e.g."trainer.json", you need to place its content
to the key "trainer_config" or the `config_main.json` (same for datasets,
data_feeder etc)
* for all the gene configs, e.g. from the folders like "plugins", you need to
add "_config" to the folder name and place all the configs from that folder as
a list of configs

## Project global config <a name="project-global-config"></a>

If you want to set the config values globally, you can put them inside of the
`training/configs/project_global_config.json`.
It allows to specify some global configuration with keys as arguments and
pass it to all of nucleotides if this parameter exist in its constructor
and it was not explicitly specified in nucleotide config.

It is also possible to specify the parameters type-wise and class-wise and
also as complete global, e.g.

```json
{
    "param1": "value1",
        "ModelPlugin": {
            "activation": "elu",
            "param2": "value2"
        },
        "Plugin1": {
            "param1": "value12",
            "activation": "relu"
        }
}
```
In that case, ModelPlugin will be initialized with activation = elu and
param1 = value1 and Plugin1 (for this example Plugin1 is a child of ModelPlugin)
class will be initialized with activation = relu and
param1 = value12 and param2 = value2. E.g. it will create inheritance map
for each nucleotide and will resolve it in hierarchy order, e.g. lower
hierarchy (Nucleotide) will be overridden by higher
(Nucleotide -> ModelPlugin -> Plugin1 -> Plugin1Child -> ...)
Parameters can be set to all of nucleus7 interfaces.

## Configs update rule <a name="configs-update-rule"></a>

Since there may be used a nested structure of the configs, we need to resolve
them. Resolve means update one config with other, e.g.
`config1` &rightarrow; `config2` means update `config1` with `config2`.
Each config has following format `{key: subconfig}` and subconfigs can be of
2 types: dict (e.g. model, dataset) and list (e.g. model_plugin, callbacks etc.) 
Following rules apply for every subconfig:

* dict subconfigs:
    - if key of `subconfig2` does not exist in `config1`, it will add it 
    - if `config2` has `__UPDATE_CONFIG__` in it and it is set to `True`,
    then it will perform `subconfig1.update(subconfig2)`
    - else `subconfig2` will replace `subconfig1` for the same key 
* list subconfigs:
    - if `subconfig2` has `__BASE__` key inside, then this key will be replaced
    with the `subconfig1`, e.g.
    ```python
    subconfig1 = [2, 3]
    subconfig2 = [1, "__BASE__", 4, 5]
    subconfig_updated = [1, 2, 3, 4, 5]
    ```
    - otherwise replace `subconfig1` withg `subconfig2`

Configs are read and resolved / updated in following order:

* training:
`config_main.json` &rightarrow; `component configs`
* inference:
`inference/configs` &rightarrow; `additional_configs`, where each of them
is resolved in `config_main.json` &rightarrow; `component configs` order
