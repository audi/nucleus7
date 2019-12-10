Project execution
=================

- [Training](#training)
    - [Continue training](#continue-training)
    - [Distributed run](#distributed-run)
- [Inference](#inference)
    - [Additional configs for inference](#additiona-configs-infer)
    - [Additional parameters for inference](#additional-parameters-infer)
- [KPI Evaluation](#kpi-evaluation)
    - [Additional configs for KPI evaluation](#additiona-configs-kpi)
    - [Additional parameters for KPI evaluation](#additional-parameters-kpi)
- [Data extraction](#data-extraction)
    - [Additional configs for data extraction](#additiona-configs-data)
    - [Additional parameters for data extraction](#additional-parameters-data)
- [Get nucleotide signature](#get-nucleotide)
- [Create nucleotide sample config](#create-nucleotide-sample-config)
- [Create dataset file list](#create-dataset-file-list)
- [Visualize project DNA](#visualize-project-dna)


[ProjectStructure]: ./ProjectStructure.md
[ProjectConfigs]: ./ProjectConfigs.md

[Bin]: ./bin
[train]: ./bin/nc7-train
[infer]: ./bin/nc7-infer
[extract_data]: ./bin/nc7-extract_data
[evaluate_kpi]: ./bin/nc7-evaluate_kpi
[get_nucleotide_info]: ./bin/nc7-get_nucleotide_info
[create_nucleotide_sample_config]: ./bin/nc7-create_nucleotide_sample_config
[create_dataset_file_list]: ./bin/nc7-create_dataset_file_list
[visualize_project_dna]: ./bin/nc7-visualize_project_dna

[TF_CONFIG]: https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate

You can find the scripts for nucleus7 projects inside of [nucleus7/bin][Bin]
folder. They are automatically added to your PATH after installation and so
can be used directly by the name of the script.

nucleus7 has following scripts:


| Shortcut | Executable script | Description |
| -------- | ----------------- | ----------- |
| `nc7-train` | [`bin/nc7-train`][train] | training and evaluation |
| `nc7-infer` | [`bin/nc7-infer`][infer] | inference |
| `nc7-extract_data` | [`bin/nc7-extract_data`][extract_data] | extract data using `DataExtractor` |
| `nc7-evaluate_kpi` | [`bin/nc7-evaluate_kpi`][evaluate_kpi] | evaluate KPI on predictions against labels using `KPIEvaluator` |
| `nc7-get_nucleotide_info` | [`bin/nc7-get_nucleotide_info`][get_nucleotide_info] | print the nucleotide signature |
| `nc7-create_nucleotide_sample_config` | [`bin/nc7-create_nucleotide_sample_config`][create_nucleotide_sample_config] | create a config sample for nucleotide |
| `nc7-create_dataset_file_list` | [`bin/nc7-create_dataset_file_list`][create_dataset_file_list] | match the file names and save them to the folder |
| `nc7-visualize_project_dna` | [`bin/nc7-visualize_project_dna`][visualize_project_dna] | create project DNA and save it to the project folder |


## Training <a name="training"></a>

To start the training, use:

```bash
nc7-train path/to/project_dir
```

This will start the training inside of the project_dir folder. Be sure that you
have at least 1 root folder if you do not set the `MLFLOW_TRACKING_URI`,
since the mlruns will be created under the `path/to/` folder.

The folder must contain the training configs. How to setup them, refer to
[project configs][ProjectConfigs] and about the project directory structure
refer to [project structure][ProjectStructure].

If the training already exist inside of the project_dir, it will raise an Error.

### Continue training <a name="continue-training"></a>

If you want to continue the training, set the continue flag:

```bash
nc7-train path/to/project_dir --continue
```

If no training was found there, it will run same as without the flag, but if
you have performed the training there, it will start from the last checkpoint.

### Distributed run <a name="distributed-run"></a>

Since **nucleus7** uses `tf.estimator` API and `tf.estimator.train_and_evaluate`
method to run the training and evaluation, it is possible to run the distributed
training and evaluation without any modifications of source code.

Let's show an example how start distributed training. You need to have the project_dir, that
is mounted in the same manner in all your processes, when you want to start the
training / evaluation. You need also to setup the `TF_CONFIG` for
every process you start ([see more here][TF_CONFIG]).

Following example has (but is only an example and you can use whatever
configuration you want to use, but evaluator is all the time in a separate
node / process, also ports and host IPs are free to choose):

- chief worker
- 1 additional worker
- 2 parameter servers - one on chief node and one on worker node 
- evaluator

```bash
# from CHIEF node:

# 1 start the chief worker node
TF_CONFIG='{
    "cluster": {
        "chief": ["chief_host:2222"],
        "worker": ["host_worker: 2222"],
        "ps": ["chief_host:3333", "host_worker: 3333"]
    },
    "task": {"type": "chief", "index": 0}
}' nc7-train path/to/project/dir > /dev/null 2>&1

# 2 start the parameter server node
TF_CONFIG='{
    "cluster": {
        "chief": ["chief_host:2222"],
        "worker": ["host_worker: 2222"],
        "ps": ["chief_host:3333", "host_worker: 3333"]
    },
    "task": {"type": "ps", "index": 0}
}' nc7-train path/to/project/dir > /dev/null 2>&1


# from WOKRER node:
# 1 start the chief worker node
TF_CONFIG='{
    "cluster": {
        "chief": ["chief_host:2222"],
        "worker": ["host_worker: 2222"],
        "ps": ["chief_host:3333", "host_worker: 3333"]
    },
    "task": {"type": "worker", "index": 0}
}' nc7-train path/to/project/dir > /dev/null 2>&1

# 2 start the parameter server node
TF_CONFIG='{
    "cluster": {
        "chief": ["chief_host:2222"],
        "worker": ["host_worker: 2222"],
        "ps": ["chief_host:3333", "host_worker: 3333"]
    },
    "task": {"type": "ps", "index": 1}
}' nc7-train path/to/project/dir > /dev/null 2>&1


# from EVALUATOR node:
TF_CONFIG='{
    "cluster": {
        "chief": ["chief_host:2222"],
        "worker": ["host_worker: 2222"],
        "ps": ["chief_host:3333", "host_worker: 3333"]
    },
    "task": {"task": "evaluator", "index": 0}
}' nc7-train path/to/project/dir > /dev/null 2>&1

```

All the communication and triggering is done by tensorflow / nucleus7 :)

Also if you want to set up the `MLFLOW_TRACKING_URI`, you need to do it only
on the chief and evaluator.

## Inference <a name="inference"></a>

To start the inference, please use:

```bash
nc7-infer path/to/project_dir
```

This will:
- take the last saved_model from `project_dir/saved_models`
- create new run folder under the `project_dir/inference`
- use configs from `project_dir/inference/configs`
- start inference in the not existing folder with name
`project_dir/inference/run-{N}`, where N is incremented if a folder with name
{N-1} was found and it had other as configs subfolder.

It is also possible to provide the run name explicit and not use a incremented
run-{} structure. For it use (if folder has not allowed content, e.g. other
folders as configs, then ValueError will be raised):

```bash
nc7-infer path/to/project_dir --run_name run-name
```

How to setup the configs, refer to [project configs][ProjectConfigs] and about
the project directory structure refer to [project structure][ProjectStructure].

Basically you do not need to do anything more to start the inference. But if
you want more control, you can use additional configs and also control some
parameters using CLI arguments. 

### Additional configs for inference <a name="additiona-configs-infer"></a>

To update the default configs with other ones, use additional_configs_path
argument:

```bash
nc7-infer path/to/project_dir --additional_configs_path other/path/to/configs
```

This will search for the configs inside of `other/path/to/configs` path
(relative to project_dir or absolute) and update the main configs with it.

### Additional parameters for inference <a name="additional-parameters-infer"></a>

You can also set some attributes on CLI to override the config parameters:

| CLI argument | overridden config parameter | meaning | default value |
| ------------ | --------------------------- | ------- | ------------- |
| `--batch_size` | inferer.json: run_config: batch_size | batch size to use | no default value; if not specified inside of the config and as argument, will raise an error |
| `--saved_model` | inferer.json: load_config: saved_model_path | tag of the saved model to use relative to `project_dir/saved_models` | last existing tag inside of `project_dir/saved_models`|
| `--prefetch_buffer_size` | inferer.json: run_config: prefetch_buffer_size | how many batches to prefetch | 10 |
| `--use_tensorrt` | inferer.json: tensorrt_config: use_tensorrt | if tensorrt should be used for inference | False or defined in the config |
| `--use_single_process` | inferer.json: run_config: use_multiprocessing | if the inference must run in single process | False |
| `--checkpoint` | inferer.json: load_config: checkpoint_path | checkpoint file name relative to `project_dir/checkpoints` | no default, since default is saved_models used |
| `--number_of_shards` | data_feeder.json: file_list: number_of_shards | will shard the data if file list is presented inside of data_feeder; used for the parallel inference, where you want to run multiple processes of inference | 1 |
| `--shard_index` | data_feeder.json: file_list: shard_index | will use this shard index to select out of number_of_shards of file list, if file list is presented in data_feeder | 0 |

## KPI Evaluation <a name="kpi-evaluation"></a>

To calculate KPI (Key Performance Index) values, you can use following command:

```bash
nc7-evaluate_kpi path/to/project_dir
```

This will:
- use configs from `project_dir/kpi_evaluation/configs`
- start kpi evaluation in the not existing folder with name
`project_dir/kpi_evaluation/run-{N}`, where N is incremented if a folder with
name {N-1} was found and it had other as configs subfolder.

It is also possible to provide the run name explicit and not use a incremented
run-{} structure. For it use (if folder has not allowed content, e.g. other
folders as configs, then ValueError will be raised):

```bash
nc7-evaluate_kpi path/to/project_dir --run_name run-name
```

How to setup the configs, refer to [project configs][ProjectConfigs] and about
the project directory structure refer to [project structure][ProjectStructure].

Basically you do not need to do anything more to start the inference. But if
you want more control, you can use additional configs and also control some
parameters using CLI arguments. 

### Additional configs for KPI evaluation <a name="additiona-configs-kpi"></a>

To update the default configs with other ones, use additional_configs_path
argument:

```bash
nc7-evaluate_kpi path/to/project_dir --additional_configs_path other/path/to/configs
```

This will search for the configs inside of `other/path/to/configs` path
(relative to project_dir or absolute) and update the main configs with it.

### Additional parameters for KPI evaluation <a name="additional-parameters-kpi"></a>

You can also set some attributes on CLI to override the config parameters:

| CLI argument | overridden config parameter | meaning | default value |
| ------------ | --------------------------- | ------- | ------------- |
| `--batch_size` | inferer.json: run_config: batch_size | batch size to use | 1 |
| `--prefetch_buffer_size` | inferer.json: run_config: prefetch_buffer_size | how many batches to prefetch | 10 |
| `--use_single_process` | inferer.json: run_config: use_multiprocessing | if the inference must run in single process | False |
| `--number_of_shards` | data_feeder.json: file_list: number_of_shards | will shard the data if file list is presented inside of data_feeder; used for the parallel inference, where you want to run multiple processes of inference | 1 |
| `--shard_index` | data_feeder.json: file_list: shard_index | will use this shard index to select out of number_of_shards of file list, if file list is presented in data_feeder | 0 |


## Data extraction <a name="data-extraction"></a>

To perform data extraction, you can use following command:

```bash
nc7-extract_data path/to/project_dir --run_name run-name
```

This will:
- use configs from `project_dir/data_extraction/configs`,
`project_dir/data_extraction/run_name/configs`
- start data extraction for given run name

How to setup the configs, refer to [project configs][ProjectConfigs] and about
the project directory structure refer to [project structure][ProjectStructure].

Basically you do not need to do anything more to start the inference. But if
you want more control, you can use additional configs and also control some
parameters using CLI arguments. 

### Additional configs for data extraction <a name="additiona-configs-data"></a>

To update the default configs with other ones, use additional_configs_path
argument:

```bash
nc7-evaluate_kpi path/to/project_dir --run_name run-name --additional_configs_path other/path/to/configs
```

This will search for the configs inside of `other/path/to/configs` path
(relative to project_dir or absolute) and update the main configs with it.

### Additional parameters for data extraction <a name="additional-parameters-data"></a>

You can also set some attributes on CLI to override the config parameters:

| CLI argument | overridden config parameter | meaning | default value |
| ------------ | --------------------------- | ------- | ------------- |
| `--batch_size` | inferer.json: run_config: batch_size | batch size to use | 1 |
| `--prefetch_buffer_size` | inferer.json: run_config: prefetch_buffer_size | how many batches to prefetch | 10 |
| `--use_single_process` | inferer.json: run_config: use_multiprocessing | if the inference must run in single process | False |
| `--number_of_shards` | data_feeder.json: file_list: number_of_shards | will shard the data if file list is presented inside of data_feeder; used for the parallel inference, where you want to run multiple processes of inference | 1 |
| `--shard_index` | data_feeder.json: file_list: shard_index | will use this shard index to select out of number_of_shards of file list, if file list is presented in data_feeder | 0 |

## Get nucleotide signature <a name="get-nucleotide"></a>

If you want to get the nucleotide signature (incoming and generated keys,
constructor parameters etc.) you can run:

```bash
nc7-get_nucleotide_info class.name.of.nucleotide
```

### Create nucleotide sample config <a name="create-nucleotide-sample-config"></a>

It is hard sometimes to create the configs for nucleotides, so you can use
the sample config generator to create the config with default values and then
adjust it and use it in your project:

```bash
nc7-create_nucleotide_sample_config class.name.of.nucleotide \
    --output_file_path /path/to/output/json # otherwise will be printed to stdout
```

Caveat: do not forget to change every section of the generated config with
"TODO" before using! 

## Create dataset file list <a name="create-dataset-file-list"></a>

Sometimes it is good idea to generate the file names out of the glob expressions
after matching, sharding etc. For it you can use:

```bash
nc7-create_dataset_file_list --save_dir path/to/save/dir \
    -f path/to/main/file_list_config.json
```

### Visualize project DNA <a name="visualize-project-dna"></a>

If you want to generate the graphical representation of the project, e.g.
data flow over nucleotides, e.g. `DNA helix`, you can use:

```bash
nc7-visualize_project_dna.py path/to/project/dir --type train
```

It will display the project dna for type you want and the plot is interactive :)
So you can select subraphs for every nucleotide and see it's information.
Just check it out :)
