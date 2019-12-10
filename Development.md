Development
===========

- [Create new nucleotide](#create-new-nucleotide)
- [Variables sharing for multi-device setups](#variables-sharing-for-multi-device-setups)
- [Use of keras or other shared state objects](#use-of-keras)
- [Add model parameters that can be modified during inference](#model-parameters-during-inference)
- [Special cases](#special-cases)
    - [Meta flags](#meta-flags)
    - [Add artifacts to log](#add-artifacts-to-log)
- [Data format](#data-format)

[ProjectConfigs]: ./ProjectConfigs.md

## Create new nucleotide <a name="create-new-nucleotide"></a>

Adding new nucleotide is pretty easy. You just need to define which type you would
like to inherit from and override it's process method.

```python
import nucleus7 as nc7

class VerySophisticatedArchitecture(nc7.model.ModelPlugin):
    # this names must match kwargs from predict method:
    incoming_keys = ['image1', 'image2', '_optional_input']
    # this names must match keys of the returned dict out of process method
    generated_keys = ['very_novel_output']

    def __init__(self, param1=None, **config_kwargs):
        super(VerySophisticatedArchitecture, self).__init__(...)
        self.param1 = param1

    @property
    def defaults(self):
        defaults = super(VerySophisticatedArchitecture, self).defaults
        param1_default = {"k1": "v1", "k2": "v2"}
        defaults["param1"] = param1_default
        return defaults

    # in most of the cases you do not need this method
    def build(self):
        super(VerySophisticatedArchitecture, self).build()
        # do a build of your attributes
        return self

    # process method is 'predict' for ModelPlugin and 'process' for other nucleotides
    def predict(self, image1, image2, optional_input=None):
        is_training = self.is_training
        mode = self.mode
        very_novel_output = ...
        return {"very_novel_output": very_novel_output}
```

And that is all!

Let's take a look at main building blocks of the implementation:

- `incoming_keys` and `generated_keys`:
    * this are basically the signature of the nucleotide - `ìncoming_keys`
    describe the names of input data parameters and `generated_keys` describe
    the output keys.
    * if the key is optional, add the '\_' in the beginning, e.g.
    `_optional_key`, but use it without '\_' prefix inside of the process method
    * it is also possible to specify dynamic keys e.g. the keys that
    can differ and depend on inbound nucleotides (setting
    `dynamic_incoming_keys=True` and (or) `dynamic_generated_keys=True`),
    and to combine static and dynamic keys. If nucleotide has dynamic incoming
    keys, then that keys are equal to remapped generated keys from all
    inbound nodes; if it has dynamic generated keys, then you are free to
    return whatever you want. Be sure, that you use **kwargs** signature
    inside of process method if you specify dynamic_incoming_keys!

    ```python
    import nucleus7 as nc7

    class NucleotideWithDynamicKeys(nc7.Nucleotide):
        incoming_keys = ["key1"]
        generated_keys = ["key2"]
        dynamic_incoming_keys = True
        dynamic_generated_keys = True

        def process(self, key1, **dynamic_keys):
            return {"key2": ..., "dynamic_key1": ...,}
    ```
    
- `__init__` method:
    * this is common constructor. Everything what is passed using
    **config_kwargs** can be configured over corresponding `config.json` and
    will be logged when the object is constructed
    * **DO NOT** perform any computations here, only assignments and sanity
    checks
    * **DO NOT** create any tensorflow objects with variables here,
    since graph is empty at this point and if you create some, it will be not
    used further
- `defaults` method:
    this method is used when you need to set the default values for arguments
    that were not provided to constructor. This is much cleaner to use this
    standalone method instead of providing the defaults inside of the
    constructor (in case of mutable objects) and it is automatically
    called inside of the `build` method.
- `build` method:
    * if you need any calculations or building of attributes, this is for you
    * **DO NOT** create any tensorflow objects with variables here,
    since graph is empty at this point and if you create some, it will be not
    used further
- `process` or `predict` method:
     - take a look on the parent, which method it has (only ModelPlugin uses
     `predict`, other ones use `process`)
     - create tensorflow variables here
     - kwargs names of the method must match the `Class.incoming_keys` and
     optional ones must be set to `None`
     - method **MUST** return dict with keys equal to `Class.generated_keys`
     (optional keys can be omitted) 
- if you need to access the run mode of nucleotide or to know if it is training,
e.g. for the dropout layers, you can use `self.mode` and `self.is_training`
respectively

**DO NOT** override the `__call__` method, since it is used internally!

Side note: It is better to do the implementation as flexible as possible and
do not hardcore attributes if they can alter.

## Variables sharing for multi-device setups <a name="variables-sharing-for-multi-device-setups"></a>

Parameters can be shared over multiple devices, not over multiple nucleotides.

Currently (as of tensorflow <= 2.0), there are 2 ways to create variables
(both ways are supported by **nucleus7**) and need to be handled differently:
- using `tf.get_variable` - used by tensorflow
    * to handle the sharing, `tf.get_variable_scope().reuse` can be used even
    if new object is generated
    * does not need any special behavior from user side
- using `tf.Variable` - used by keras
    * can be handled ONLY by creating the object (e.g. keras layer) once and
    then reusing it over all the instances
    * need to be reset if new graph is used
    * implementations using keras layers, need to add the layers / objects
    using predefined interface

## Use of keras or other shared state objects <a name="use-of-keras"></a>

Since we need to create them only once and reset when it is needed, e.g.
for evaluation or inference graph export, following interface must be used:

```python
import tensorflow as tf

import nucleus7 as nc7

class PluginWithKeras(nc7.ModelPlugin):

    def __init__(self, **config_kwargs):
        ...
        # following lines are a good style, but not mandatory for functionality
        self.layer1 = None # type: tf.keras.layers.Layer
        self.layer2 = None # type: tf.keras.layers.Layer
    
    def create_keras_layers(self):
        self.layer1 = self.add_keras_layer(tf.keras.layers.Dense(10))
        self.layer2 = self.add_keras_layer(tf.keras.layers.Dense(20))

    def predict(self, x, **inputs):
        x = self.layer1(x)
        x = self.layer2(x)
        one_more_layer = self.add_keras_layer(
            tf.keras.layers.Dense(100), name="new_layer")
        x = one_more_layer(x)
        ...

```

Let's take a look on the steps:

* attributes that will hold the layers should be defined in the constructor
(just a good style)
* create the keras layers using `self.add_keras_layer(...)` inside
`create_keras_layers()` - this will track the layers in the nucleotide and
allow them to be reset when needed.
**CAUTION**: make sure that this method does not generate any tensorflow
variables!
Otherwise call `add_keras_layer(...)` as described further in
`plugin.predict(...)`
* `create_keras_layers()` will be called automatically inside `build()`
* use attribute layers (aka `self.layer1`) inside of the `predict()` /
`process()`
* if you want to create keras layers with constructors which require tensorflow
tensors (usually not the best choice, but sometimes unavoidable),
then you need to add the `name` argument to
`add_keras_layer(..., name="unique_name")`,
as shown above for `one_more_layer`
This allows the layer to be created once and reused for multi-device setup.
Use unique names for unique layers
* it is possible to provide a lambda expression to `add_keras_layer(...)`
to build the layer. This is an alternative to building the heavy weight layers
every time inside `predict(...)`

**DO NOT** call `keras.Layer.build(...)` method before
`nucleotide.predict(...)` method!

## Add model parameters that can be modified during inference <a name="model-parameters-during-inference"></a>

It is possible to change a parameter of tensorflow nucleotide during inference.
This can be useful e.g. for object detection models like Faster RCNN, where
you can specify different number of predictions and your trained model
does not depend on it. So the idea is to add the placeholder with default
value to the tensorflow graph and then to modify it if needed during inference:

```python
import nucleus7 as nc7

class PluginWithParameter(nc7.model.ModelPlugin):

    def predict(self, **inputs):
        parameter1 = self.add_default_placeholder(10, "parameter1")
        ...
``` 

Now you have a placeholder with name
`predictions_raw/PluginWithParameter/PluginWithParameter//parameter1:0` and 
value 10 inside of your model and it can be feeded during inference
(see [here][ProjectConfigs] for details). Also this placeholder is written
to the input receiver function inside of saved model interface and also
to 'checkpoints/input_output_names.json' file.

## Special cases <a name="special-cases"></a>

Following functionality already used in its default configuration, so use it
only if you really need it :)

### Meta flags <a name="meta-flags"></a>

Following functionality must be used with caution and only if you know what
you want to do.

There are some special class attributes by setting them you can control how
it will be handled internally:

```python
import nucleus7 as nc7


# you are free to select the class to inherit
class NewNucleotide(nc7.core.Nucleotide):
    register_name_scope = 'e.g. plugin'
    register_name = 'class.name'
    log_name_scope = 'e.g. plugin'
    exclude_from_register = False
    exclude_from_log = False
    exclude_args_from_log = ['inherited']
```

Let's describe them:
* register_name_scope - name scope for register; if not set, original
  class name will be used (will be searched also inside of bases)
* register_name - name of the class inside of register; if not set,
  class name will be used (will be searched only in the class)
* log_name_scope - name scope for config logger; defaults to
  register_name_scope (will be searched also inside of bases)
* exclude_args_from_log - list of the constructor arguments that must be
  not logged (will be searched also inside of bases)
* exclude_from_register - flag if this class must be excluded from
  register, e.g. for interfaces and base classes
  (will be searched only in the class)
* exclude_from_log - flag if this class must be excluded from
  constructor log, e.g. for intermediate classes, defaults to
  exclude_from_register (will be searched only in the class)

### Add artifacts to log <a name="add-artifacts-to-log"></a>

If you want that some of the objects, that will be generated during the build
of your nucleotide, are logged, e.g. file lists of datasets and data feeders
(is already there by default), you can use following decorator on the top
of the property to store:

```python
import nucleus7 as nc7
from nucleus7.core import project_artifacts


# you are free to select the class to inherit
class NewNucleotide(nc7.core.Nucleotide):

    def __init__(self, **config_kwargs):
        ...
        self.parameter_to_log = None

    @project_artifacts.add_project_artifact(
        "name_of_the_artifact_to_save", "parameter_to_log")
    def build(self):
        ...
        self.parameter_to_log = {"key": "value to log"}
        ...

```

This will add the `parameter_to_log` to the artifacts.
If `NewNucleotide` have `mode` property (as most of them) and this property
is set on the time of run start (even after the build method was called),
it will add mode value to the artifact name. This is also possible to use
different custom serializer to serialize the artifact
(default the json serializer is used).

Parameters can have also a nested structure, e.g. if you want to log the
`self.param1.param2.param3` value, use
`add_project_artifact("name_of_the_artifact_to_save", "param1.param2.param3")`.

Caveat: artifacts will be saved to memory till they are saved to project

### Tracking of the execution time

It is also possible to log the execution time of methods of nucleotides. But
this is method execution time, so if you add it to the predict method of
model plugin, this method is called once for the training for construction
of tensorflow graph, so it will give you only this name. But it makes sence
for the objects, which are not inside of the tensorflow graph, like
`CoordinatorCallback` (also for the training), `DataFeeder` etc.

This performance measurement will be added to the mlflow metrics tracking

To add it, use following decorator (will log the time to
`performance-{ClassName}-method_name`):

```python
import nucleus7 as nc7
from nucleus7.utils import mlflow_utils


class NewNucleotide(nc7.coordinator.CoordinatorCallback):

    @mlflow_utils.log_nucleotide_exec_time_to_mlflow(method_name="some_name")
    def some_method(self):
        ...
```

You can also specify for which mode you want to log the performance by
providing `log_on_train`, `log_on_eval` or `log_on_infer` arguments
to the decorator. Default performance is logged only for inference.

## Data format <a name="data-format"></a>

It is also possible to set the data_format for all the nucleotides
globally or locally by using following format:

```json
{
  "data_format": {
    "image": "NHWC",
    "video": "NTHWC",
    "time_series": "NTC",
    "...": "..."
  }
}
```

And you can access it by `self.data_format` in nucleotide itself.

**Caveat**: to make this setting work, you need to have be sure that
the nucleotides you use, rely on it.
