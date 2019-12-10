Project structure
=================

- [Training and evaluation](#training-and-evaluation-project)
- [Inference](#inference-project)
- [KPI evaluation](#kpi-evaluation-project)
- [Data extraction](#data-extraction-project)

## Training and evaluation <a name="training-and-evaluation-project"></a>

To start the training, project directory must have following structure:

|                    Directory name               |                                Content                                   |  Fixed name | Mandatory  |
| ------------------------------------------------|--------------------------------------------------------------------------|:---:|:----------:|
| `training/configs/`                             | configs for training                                                     | yes | yes |
| `training/configs/project_global_config.json`   | global parameters for configs                                            | yes | no  |
| `training/configs/config_main.json`             | full configuration for the whole project including all the components    | yes | no  |
| `training/configs/datasets.json`                | configuration of the datasets with under `train` and `eval` keys         | yes | yes |
| `training/configs/trainer.json`                 | configuration of the trainer                                             | yes | yes  |
| `training/configs/model.json`                   | configuration of the model                                               | yes | no  |
| `training/configs/callbacks/*.json`             | configs for callbacks that will be used for both `train` and `eval`, \*   | no  | no  |
| `training/configs/callbacks_train/*.json`       | configs for callbacks that will be used for `train` only, \*              | no  | no  |
| `training/configs/callbacks_eval/*.json`        | configs for callbacks that will be used for `eval` only, \*               | no  | no  |
| `training/configs/kpi/*.json`                   | configs for kpi plugins and accumulators, \*                              | no  | no  |
| `training/configs/losses/*.json`                | configs for losses, \*                                                    | no  | yes |
| `training/configs/metrics/*.json`               | configs for metrics, \*                                                   | no  | no  |
| `training/configs/plugins/*.json`               | configs for plugins, \*                                                   | no  | yes |
| `training/configs/postprocessors/*.json`        | configs for postprocessors, \*                                            | no  | no  |
| `training/configs/summaries/*.json`             | configs for summaries, \*                                                 | no  | no  |

where \* means that it can be one config with a list of configs or a single one or
multiple configs with single configs and arbitrary names

Training run will generate following folders and files:

|            Directory name                   |                       Content                       |
| --------------------------------------------|-----------------------------------------------------|
| `nucleus7_project.json`                     | meta information about the project                  |
| `saved_models/{tag}`                        | saved models in SavedModels format after each epoch |
| `saved_models/{tag}/model.info`             |information about the model, e.g. global_step        |
| `saved_models/{tag}/eval_result.json`       | loss value for this model                           |
| `saved_models/{tag}/kpi_{}.json`            | optional KPI values for this model if KPIEvaluators were provided |
| `checkpoints/`                              | checkpoints after every epoch                       |
| `checkpoints/input_output_names.json`       | input and output tensors of the model               |
| `checkpoints/graph_inference.meta`          | meta graph for the inference                        |
| `training/artifacts/`                       | training artifacts, e.g. file lists used            |
| `training/artifacts/_config_train_run.json` | configuration used to start the run including the default values used for every nucleus7 object |
| `training/callbacks/train`                  | train callbacks folder, e.g. if some of the train callbacks write |
| `training/callbacks/eval`                   | evaluation callbacks folder, e.g. if some of the eval callbacks write |
| `training/summaries/train`                  | train summaries |
| `training/summaries/eval`                   | evaluation summaries |


## Inference <a name="inference-project"></a>

Inference project relies on the configs and also on the load configuration, e.g.
you need to load the model from SavedModels or from the checkpoints.
So following directory structure should be in the beginning:

|            Directory name             |                     Content                       |
| --------------------------------------|---------------------------------------------------|
| `saved_models/{tag}`                  | if you want to load from the saved model with tag |
| `checkpoints/{checkpoint_name}.index` | if you want to load from checkpoint_name          | 
| `checkpoints/{checkpoint_name}.chpt`  | if you want to load from checkpoint_name          |
| `checkpoints/graph_inference.meta`    | if you want to load from checkpoint_name          |
| `inference/configs`                   | configs to load for inference (default folder to search for configs, but is optional) |

You can provide the configs inside of the `inference/configs` folder, which is
default one to search for configs, or to use the additional configs to load
from other place (if you provide both, then first default configs will be
loaded and then they will be updated with additional one). But configs folders
must have following structure:

|        Directory name                  |                     Content                       |  Fixed name | Mandatory  |
| ---------------------------------------|---------------------------------------------------|:---:|:----------:|
| `{configs}/project_global_config.json` | global parameters for configs                                            | yes | no  |
| `{configs}/config_main.json`           | full configuration for the whole project including all the components | yes | no |
| `{configs}/data_feeder.json`           | configuration of the data feeder                  | yes | yes |
| `{configs}/inferer.json`               | configuration of the inferer                      | yes | no |
| `{configs}/callbacks/*.json`           | configs for callbacks that will be used, \*        | yes | no |
| `{configs}/kpi/*.json`                 | configs for kpi plugins and accumulators, \*       | no  | no |

Following directories will be generated after inference run:

|                   Directory name                     |                     Content                       |
| -----------------------------------------------------|---------------------------------------------------|
| `inference/run-{N}`                                  | folder for the run |
| `inference/run-{N}/artifacts`                        | artifacts of the run, e.g. file lists used          |
| `inference/run-{N}/artifacts/_config_infer_run.json` | configuration used to start the run including the default values used for every nucleus7 object |
| `inference/run-{N}/results`                          | results of callbacks          |
| `inference/last_run`                                 | symbolic link to the last run folder |


## KPI evaluation <a name="kpi-evaluation-project"></a>

KPI evaluation project relies on the configs with following structure
So following directory structure should be in the beginning:

|            Directory name           |                     Content                       |
| ------------------------------------|---------------------------------------------------|
| `kpi_evaluation/configs`            | configs to load for kpi evaluation (default folder to search for configs, but is optional) |

You can provide the configs inside of the `kpi_evaluation/configs` folder, which is
default one to search for configs, or to use the additional configs to load
from other place (if you provide both, then first default configs will be
loaded and then they will be updated with additional one). But configs folders
must have following structure:

|        Directory name                  |                     Content                       |  Fixed name | Mandatory  |
| ---------------------------------------|---------------------------------------------------|:---:|:----------:|
| `{configs}/project_global_config.json` | global parameters for configs                                            | yes | no  |
| `{configs}/config_main.json`           | full configuration for the whole project including all the components | yes | no |
| `{configs}/data_feeder.json`           | configuration of the data feeder                  | yes | yes |
| `{configs}/callbacks/*.json`           | configs for callbacks that will be used, \*        | yes | no |
| `{configs}/kpi/*.json`                 | configs for kpi plugins and accumulators, \*       | no  | no |

Following directories will be generated after inference run:

|                   Directory name                          |                     Content                       |
| ----------------------------------------------------------|---------------------------------------------------|
| `kpi_evaluation/run-{N}`                                  | folder for the run |
| `kpi_evaluation/run-{N}/artifacts`                        | artifacts of the run, e.g. file lists used          |
| `kpi_evaluation/run-{N}/artifacts/_config_infer_run.json` | configuration used to start the run including the default values used for every nucleus7 object |
| `kpi_evaluation/run-{N}/results`                          | results of callbacks and KPI evaluator          |
| `kpi_evaluation/last_run`                                 | symbolic link to the last run folder |

## Data extraction <a name="data-extraction-project"></a>

Data extraction project relies on the configs with following structure
So following directory structure should be in the beginning:

|            Directory name           |                     Content                       |
| ------------------------------------|---------------------------------------------------|
| `data_extraction/configs`           | configs to load for data extraction (default folder to search for configs, but is optional) |

You can provide the configs inside of the `data_extraction/configs` folder, which is
default one to search for configs, or to use the additional configs to load
from other place (if you provide both, then first default configs will be
loaded and then they will be updated with additional one). Also it is possible
to use the same configs structure for different runs, e.g.
`data_extraction/train/cofnigs` to use additional configs for train run.
But configs folders must have following structure:

|        Directory name                 `|                     Content                       |  Fixed name |` andatory  |
| ---------------------------------------|---------------------------------------------------|:---:|:----------:|
| `{configs}/project_global_config.json` | global parameters for configs                                            | yes | no  |
| `{configs}/config_main.json`           | full configuration for the whole project including all the components | yes | no |
| `{configs}/data_feeder.json`           | configuration of the data feeder                  | yes | yes |
| `{configs}/callbacks/*.json`           | configs for callbacks that will be used, \*        | yes | no |

Following directories will be generated after inference run:

|                   Directory name                           |                     Content                       |
| -----------------------------------------------------------|---------------------------------------------------|
| `kpi_evaluation/run_name`                                  | folder for the run |
| `kpi_evaluation/run_name/artifacts`                        | artifacts of the run, e.g. file lists used          |
| `kpi_evaluation/run_name/artifacts/_config_infer_run.json` | configuration used to start the run including the default values used for every nucleus7 object |
| `kpi_evaluation/run_name/extracted`                        | results of callbacks          |
