Model
=====

- [Main concepts](#concepts)
    - [Variable scopes](#varscopes)
    - [Collections](#collections)
    - [Computational graphs](#graphs)
- [ModelPlugin](#plugin)
- [ModelLoss](#loss)
- [ModelPostProcessor](#postprocessor)
- [ModelSummary](#summary)
- [ModelMetric](#metric)
- [Model](#model)
    - [Mixed precision](#mixed-precision)
- [ModelHandler](#model-handler)
- [Restoring of nucleotide variables and meta graphs](#restore)

[Development]: ../../Development.md
[Optimization]: ../optimization/README.md
[MixedPrecisionLink]: https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html

Model module describes all of nucleotide types and other classes used to build
tensorflow model (computational graph)

## Main concepts <a name="concepts"></a>

Tensorflow graph is constructed by connecting operations from all of
nucleotides in topological order.
Variables / operations for all nucleotides lie in their own variable scope.
All nucleotide outputs will be added to corresponding graph collections.

### Variable scopes <a name="varscopes"></a>

There are following [variable scopes](fields.py) inside of *nucleus7* models:

|             Nucleotide                       |                 Variable scope name                 |           Graph          |
| -------------------------------------------- |-----------------------------------------------------|--------------------------|
| Dataset                                      | `ScopeNames.DATASET`                               | train + eval + inference |
| preprocessing augmentation                   | `ScopeNames.DATASET`                               | train + eval             |
| random augmentation                          | `ScopeNames.DATASET`                               | train                    |
| ModelPlugin                                  | `ScopeNames.MODEL` + / plugin name                 | train + eval + inference |
| ModelLoss                                    | `ScopeNames.LOSSES` + / loss name                  | train + eval             |
| ModelMetric                                  | `ScopeNames.METRIC` + / metric name                | train + eval             |
| ModelPostProcessor                           | `ScopeNames.POSTPROCESSING` + / postprocessor name | train + eval + inference |
| ModelSummary                                 | `ScopeNames.SUMMARY` + / summary name              | train + eval             |
| Training operation                           | `ScopeNames.TRAIN_OP`                              | train                    |

### Collections <a name="collections"></a>

Following [collections](fields.py) are created inside of graph:

|             Collection name                 |                 Description                                                         |         Graph        |
| --------------------------------------------|-------------------------------------------------------------------------------------|----------------------|
| `CollectionNames.INPUTS`                    | Inputs to model, for inference graph - placeholders to feed the data to             | train + eval + inference |
| `CollectionNames.INPUTS_PREPROCESSED`       | Inputs after constant augmentation                                                  | train + eval + inference |
| `CollectionNames.PREDICTIONS`               | Predictions after applying postprocessors, combined outputs of ModelPostProcessors  | train + eval + inference |
| `CollectionNames.LOSSES`                    | All losses, combined outputs of ModelLosses                                         | train + eval             |
| `CollectionNames.SUMMARY`                   | All summaries, combined outputs of ModelSummaries                                   | train + eval             |
| `CollectionNames.METRIC`                    | All metrics, combined outputs of ModelMetrics                                       | train + eval             |

As collections cannot handle structures other than list of tensors or tensors,
so inside you need to use `nc7.utils.tf_collections_utils.nested2collection`
and `nc7.utils.tf_collections_utils.collection2nested` to
convert nested dictionary structures to collections and retrieve nested
structured tensors from that flatten collections using the collection names
provided above.

It works in that way

```python
from nucleus7.utils import tf_collections_utils
from nucleus7.model.fields import CollectionNames

data = {'input1': tensor1,
        'input2': tensor2,
        'input3': [tensor3, tensor4],
        'input4': {'inp41': tensor5, 'inp42': tensor6}}
# add data to main collection CollectionNames.INPUTS
tf_collections_utils.nested2collection(CollectionNames.INPUTS, data)

# Following collections are generated:
#       inputs:input1 :       [tensor1]
#       inputs:input2 :       [tensor2]
#       inputs:input3/0 :     [tensor3]
#       inputs:input3/1 :     [tensor4]
#       inputs:input4/inp41 : [tensor5]
#       inputs:input4/inp42 : [tensor6]

# and now retrieve nested dict structure from that collections
data = tf_collections_utils.collection2nested(CollectionNames.INPUTS)

# data = {'input1': tensor1,
#         'input2': tensor2,
#         'input3': [tensor3, tensor4],
#         'input4': {'inp41': tensor5, 'inp42': tensor6}}
```

Then they can be retrieved from collections when needed, also after restoring
the graph from `.meta` file.

### Computational graphs <a name="graphs"></a>

There are 3 tensorflow computational graph types for each model: training, evaluation and inference graph:

|                                 |         Train graph           |       Evaluation graph        |      Inference graph    |
| ------------------------------- |:-----------------------------:| :----------------------------:| :----------------------:|
| devices                         | replicated to all defined     | replicated to all defined     | only first device       |
| collections                     | inputs, inputs_preprocessed,  | inputs, inputs_preprocessed,  | inputs, predictions     |
|                                 | predictions,                  | predictions,                  |                         |
|                                 | losses, summary, metrics,     | losses, summary, metrics,     |                         |
|                                 | summary_op, train_op          | summary_op                    |                         |
| operations                      | forward pass, loss, summary   | forward pass, loss, summary   | only forward pass       |
|                                 | summary_op, train_op          |                               |                         |
| modes                           | 'train'                       | 'eval'                        | 'infer'                 |


## ModelPlugin <a name="plugin"></a>

For more information and more perks see [Development][Development] section.

This nucleotide describes the (sub)architecture of neural network.
Main work is done inside of `ModelPlugin.predict` method.
You need to override this method.

```python

import nucleus7 as nc7

class NewPlugin(nc7.model.ModelPlugin)
    incoming_keys = ["incoming keys"]
    generated_keys = ["generated keys"]

    def create_keras_layers(self):
        self.layer1 = self.add_keras_layer(tf.keras.layers.Dense(10))

    def predict(self, input1, input2):
        ...
        output = {'output1': output1}
        return output
```

All of variables, which are generated inside of nucleotide are stored to its
`ModelPlugin.variables` field and the actual variable scope of them is stored
to `ModelPlugin.variable_scope` field.

Also you can (see API to it):
- define if that variables are trainable
- if you want to restore them from other checkpoint
- do not restore them even if they are inside of model checkpoint
- stop gradient propagation to its inputs etc.

You can also set the individual optimization config for each plugin. To read
more go to [optimization][Optimization] module.

## ModelLoss <a name="loss"></a>

This nucleotide represents the single loss function. You can have multiple
`ModelLoss` nucleotides inside of one model.
Main method called `ModelLoss.process`

```python

import nucleus7 as nc7

class NewLoss(nc7.model.ModelLoss):
    incoming_keys = ["incoming keys"]
    generated_keys = ["generated keys"]

    def process(self, predictions, labels):
        ...
        loss = {'loss': loss}
        return loss
```

If you have multiple loss components inside of one function, you can provide
`loss_scale_factors` dictionary as input to constructor, and then corresponding
losses will be multiplied with that weights. Also 'total_loss' key will be
generated after the call as scaled sum of all loss components.
This will be further accumulated to build complete loss.

Losses can also be masked, if you provide `sample_mask` incoming key to
`ModelLoss`-

You can add l1 and l2 regularizations by providing `regularization_l1` and
`regularization_l2` parameters to the `Model` constructor.

## ModelPostProcessor <a name="postprocessor"></a>

This one uses to make a postprocessing on it's inputs, e.g. obtaining class id
from probabilities etc. These operations are do not generate any gradients.
Main method called `ModelPostProcessor.process`

```python

import nucleus7 as nc7

class NewPostprocessor(nc7.model.ModelPostProcessor)
    incoming_keys = ["incoming keys"]
    generated_keys = ["generated keys"]

    def process(self, predictions1, predictions2):
        ...
        postproessed = {'predictions1_pp': predictions1_pp,
                        'predictions2_pp': predictions2_pp}
        return postproessed
```

It can be used also for postprocessing the model predictions to further use
inside of summaries

## ModelSummary <a name="postprocessor"></a>

It is used to store interested tensors inside of `.event` files
(using `tf.Summary.FileWriter`) for further browsing inside of tensorboard.
It is possible to store different data formats (images, scalars, text etc).
To identify the type you want, use following prefixes inside of return keys
after `ModelSummary.process` method:

* `scalar_*` - scalar data
* `image_*` - images
* `histogram_*` - histogram summaries
* `text_*` - text data
* `audio_*` - audio signals

Keys inside of output dictionary which do not have that prefixes, will be
ignored and not stored inside of `.event` files
(this will cause logger warning with description).
These prefixes are stripped from value names inside of `.event` files.

```python

import nucleus7 as nc7

class NewSummary(nc7.model.ModelSummary)
    incoming_keys = ["incoming keys"]
    generated_keys = ["generated keys"]

    def process(self, predictions1, predictions2):
        ...
        summary = {'scalar_value': value1,                  # will store value1 as scalar
                   'image_awesome_image': awesome_image,    # will awesome_image as image
                   'value2': value2}                        # value2 will not be stored
        return summary
```

## ModelMetric <a name="metric"></a>

Is same as **ModelSummary**. There are only 2 differences:

1. Variables generated inside of `ModelMetric.process` will be added to
collection `CollectionNames.metric_variables`
2. Combination of images across multiple gpus is done by averaging, where in
summaries is only first one selected.

## Model <a name="model"></a>

Most magic happens inside of `Model` class, which inherits from `GeneHandler`:

* Build `DNA helix` and make checks if the provided `nucleotide` configuration
is valid and logs helpful information, if some of checks failed:
    - topological sort according only to inbound nodes of each nucleotide
    - checks topological dependency, like `ModelPlugin` depends only on
    `Dataset` and other `ModelPlugin` etc. (for more precise definition of
    valid topological dependencies see `Model.nucleotide_type_dependency_map`)
    - checks *nucleotides* incoming / generated keys according to provided
    incoming key mappings
* Process all of *nucleotides* in topological order
(with feeding corresponding inputs to them) and define build methods
for plugins, losses, summaries etc.

### Mixed precision <a name="mixed-precision"></a>

You can also enable the [MixedPrecision][MixedPrecisionLink] by providing the
`MixedPrecisionConfig` or by adding following to the `model.json` file:

```json
{
  "mixed_precision_config":  {
    "use": true,
    "loss_scale_factor": 128
  },
  "...": "..."
}
```

## ModelHandler <a name="model-handler"></a>

`ModelHandler` is here to replicate the model, create the inference graph and
provide the model_fn to `Trainer`.

## Restoring of nucleotide variables and meta graphs <a name="restore"></a>

You can restore variables in 2 different ways:

* All of variables inside of model
* Variables of particular `ModelPlugin?`

For that you need to provide `load_config` to either `Model` or
`load_fname` to `ModelPlugin`, or both of them. If both load configurations are
provided, then variables from  `ModelPlugin` will be initialized from
`ModelPlugin.load_fname` and all other variables from `ModelHandler.load_config`

You can define variable scope mapping inside of `ModelPlugin.load_var_scope`
to load variables from checkpoints generated in different models.

To restore only trainable parameters (e.g. without optimizers), set
`only_trainable_parameters=True` inside of `Model.load_config`.
