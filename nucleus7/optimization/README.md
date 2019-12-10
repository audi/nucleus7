Optimization
============

- [Optimization control levels](#optim_control_levels)
- [Learning rate manipulation](#lr_manipulation)
- [Supported optimizers](#supported-optimizers)

[decouple_regularization_paper]: https://arxiv.org/abs/1711.05101
[gradient_noise_paper]: http://arxiv.org/abs/1511.06807

## Optimization control levels <a name="optim_control_levels"></a>

nucleus7 has a hierarchy of optimization settings, where more specific
optimizations settings will overwrite or extend the more general optimization
settings. In total, there are the following three levels sorted by increasing
specificy:

- [Global level](#global_level)
- [Plugin level](#plugin_level)

The first two levels allow full control over the used optimizer (e.g. you could
set a GradientDescentOptimizer as global optimizer and overwrite it for a
specific plugin to use AdamOptimizer). The only exception to this is the
learning rate which can only be set at the global level, where the plugin level
only allows to set a learning rate multiplier
(`local_learning_rate = global_learning_rate * learning_rate_multiplier`).


### Global level <a name="global_level"></a>

In general the optimization parameters at the global level can have following
attributes:

```json
{
    "learning_rate": 0.1,
    "optimizer_name": "RMSPropOptimizer",
    "decouple_regularization": true,
    "optimizer_parameters": {
      "RMSPropOptimizer_arg1": "value1",
      "RMSPropOptimizer_arg2": "value2"
    },
    "learning_rate_manipulator": {
        "class_name": "nucleus7.optimization.TFLearningRateDecay",
        "TFLearningRateDecay_arg1": "value1",
        "TFLearningRateDecay_arg2": "value2"
    },
    "gradient_noise_std": 0.01,
    "gradient_clip": 0.1
}
```

The `learning_rate` is the initial global learning rate. It should be a float.

The `optimizer_name` is the name of the used optimizer. You can find a list of
supported optimizers [here](#supported-optimizers)

The `decouple_regularization` parameter specifies whether L1 and L2
regularization are optimized using a separate optimizer. This is can improve
performance of optimizers like Adam. For details look up the [paper on
arXiv][decouple_regularization_paper].

The `optimizer_parameters` are the additional constructor arguments except
of learning_rate, that are passed to the tensorflow optimizer constructor. It's
a dict containing keyword-argument pairs.

The `learning_rate_manipulator` dict will control the learning rate
manipulation. It needs a full class name including packages, e.g.
`class_name = nucleus7.optimization.ConstantLearningRate`.
The other parameters will be given to the constructor of the chosen learning
rate manipulator. If class_name is "tf", it will wrap the standard tensorflow
decay implementation in tf.train (tf.train.*_decay).

`gradient_noise_std` allows to add a uniform random noise to gradients before
update ([arXiv][gradient_noise_paper]).

`gradient_clip` allows to set the global norm and gradients will be clipped to
it.

### Plugin level <a name="plugin_level"></a>
At the plugin level you can set all the parameters, except `learning_rate`
and `learning_rate_manipulator``:

```json
{
    "learning_rate_multiplier": 0.1,
    "optimizer_name": "RMSPropOptimizer",
    "decouple_regularization": true,
    "optimizer_parameters": {
      "RMSPropOptimizer_arg1": "value1",
      "RMSPropOptimizer_arg2": "value2"
    },
    "gradient_noise_std": 0.01,
    "gradient_clip": 0.1
}
```

This dict will be combined together with the global one using following rules:
* if the local config doesnt have `optimizer_name` provided then the global
optimizer class will be used with either local `optimizer_parameters` (if they
were provided) or with global ones otherwise
* if local config has `optimizer_name` even if it matches the global one,
global `optimizer_parameters` will not be used
* `optimizer_parameters`, `gradient_noise_std` and `gradient_clip` will be
taken from global config, if they were not provided inside of local one.
* `learning_rate_multiplier` will be multiplied with the global learning rate to
get a plugin-level learning rate which will change the same way as the global
one.

The dict needs to be specified in the plugin parameters. The parameter name the
dict should be passed as __"optimization_parameters"__.

It is also possible to set the optimization config only for particular variables
by providing it as a mapping of {variable_pattern: optimization_config}:

```json
{
  "batch_norm": {
    "learning_rate_multiplier": 0.01
  },
  "*": {
    "learning_rate_multiplier": 0.1,
    "optimizer_name": "RMSPropOptimizer",
    "decouple_regularization": true,
    "optimizer_parameters": {
      "RMSPropOptimizer_arg1": "value1",
      "RMSPropOptimizer_arg2": "value2"
    }
  }
}
```
This one will use learning_rate_multiplier for all plugin variables that have
**batch_norm** in their names and the `*` config will be used for the rest of
variables (but this is not mandatory).

## Learning rate manipulation <a name="lr_manipulation"></a>

To be able to manipulate the learning rate nucleus7 offers the ability to write
LearningRateManipulator. To create a new one you need to inherit from the
`optimization.learning_rate_manipulator.LearningRateManipulator`:

```python
import tensorflow as tf

import nucleus7 as nc7

class NewLearningRateManipulator(nc7.optimization.LearningRateManipulator):

    def __init__(self, **kwargs):
        ...

    def get_current_learning_rate(
            self,
            initial_learning_rate: float,
            global_step: tf.Tensor) -> tf.Tensor:
        return new_learning_rate
```

Please note that this function takes a tensorflow tensor in (global_step) and
is expected to return a tensorflow tensor as well. The return tensor must have
the dtype tf.float32. If you need numpy computation or more complex control
flow you can use tf.py_func to wrap normal python code. This could for example
also be used for dynamic setting of the learning rate in a cluster using MQTT.

The learning rate manipulator class must be set in the global optimization
config as

```json
{
    "learning_rate_decay": {
        "class_name": "enter.here.full.package.path.of.lr.manipulator",
        "...": "..."
    },
    "...": "..."
}
```

See the [global optimization level](#global_level) for more details.

A tip for writing new learning_rate_manipulators: 
global_step might become big, so use `tf.float64` for the internal computation
of learning rate

## Supported optimizers <a name="supported-optimizers"></a>

All optimizers from `tf.train` namespace are supported and so every parameter
from its constructor can be set using "optimizer_parameters" key. So if you want
to use e.g. `tf.train.RMSPropOptimizer` , use:

```json
{
  "optimizer_name": "RMSPropOptimizer",
  "optimizer_parameters": {
      "decay": "...",
      "momentum": "..."
  }
}
```
