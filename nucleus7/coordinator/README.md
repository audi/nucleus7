Coordinator
===========

- [CoordinatorCallback](#callback)
    - [Interruption of data flow](#interruption-of-data-flow)
    - [Saver callback](#saver-callback)
    - [Buffer callback](#buffer-callback)
    - [Callbacks as SessionRunHook](#callback_training)
    - [KPIEvaluator as callback](#kpi-callback)
- [CallbacksHandler](#callbacks-handler)
- [Trainer](#trainer)
    - [Using of estimator API](#using-of-estimator-API)
    - [Export of the model](#model-export)
    - [Profiling](#profiling)
- [Inferer](#inferer)
    - [Tensorrt support](#tensorrt-support)

[saved_model]: https://www.tensorflow.org/guide/saved_model#using_savedmodel_with_estimators
[profiler_hook]: https://www.tensorflow.org/api_docs/python/tf/train/ProfilerHook
[Install]: ../../INSTALL.md
[tf-tensorrt-link]: https://github.com/tensorflow/tensorflow/blob/16d7642c6481b703ab433596af27c2ef5141eb51/tensorflow/python/compiler/tensorrt/trt_convert.py

Main task is to coordinate the process of training / inference.

## CoordinatorCallback <a name="callback"></a>

`CoordinatorCallback` is a particular *Nucleotide* with main task to execute
some functions after each coordinator iteration (can be training, evaluation,
inference), e.g. logging the training status, storing of predictions on hard
drive, postprocessing of predictions etc.

This class has same interface for all the modes.

Let's look if you want to use it:

```python
import nucleus7 as nc7


class NewCoordinatorCallback(nc7.coordinator.CoordinatorCallback):
    # these names must match kwargs from predict method:
    incoming_keys = ['image1', 'image2', '_optional_input']
    # these names must match keys of the returned dict out of process method
    generated_keys = ['very_novel_output']

    def begin(self):
        ...

    def end(self):
        ...
    
    def on_iteration_end(self, **data):
        log_dir = self.log_dir
        summary_writer = self.summary_writer
        summary_step = self.summary_step
        iteration_info = self.iteration_info
        mode = self.mode
        is_training = self.is_training
        number_iterations_per_epoch = self.number_iterations_per_epoch
        ...

    def on_iteration_start(self):
        ...
```

Following differences to main nucleotide class:
- process method is called `on_iteration_end`, but has same signature
- it has following executive methods:
    * `begin` - is called without any arguments on the beginning of the run;
    can be used to initialize some variables
    * `end` - is called on run end, so can be used to finalize
    * `on_iteration_start` - is called before every iteration
    * `on_iteration_end` - is called after every iteration and has the outputs
    of iteration as input
- following internal attributes:
    - log_dir - directory to save the results
    - summary_writer - to use if you want to save the results to summaries
    (directory is different from log_dir, so use only this writer)
    - summary_step - step of the summaries - differ from global step, e.g.
    has offset for evaluation mode
    - iteration_info - iteration info with following fields:
        * epoch_number
        * iteration_number
        * execution_time - time of network execution
        * is_last_iteration - flag if this iteration is last one in the
        inference (is all the time False for training)
    - number_iterations_per_epoch - number of iterations per epoch

### Interruption of data flow <a name="interruption-of-data-flow"></a>

In some cases, callbacks may collect some data from multiple iterations and
pass only that collected data further, e.g. processing of sequences etc.
In all other cases, when data was not collected still, you don't want to pass
data to next callbacks. To implement it, you can just return `None` inside of
your `on_iteration_end` method:

```python
import nucleus7 as nc7

class CollectorCallback(nc7.coordinator.CoordinatorCallback):

    def process(self, **data):
        collect_data()
        if not_collected:
            return None

        return result
```

This will pass the results only when data is collected and in other cases
all dependent callbacks will not be executed.

### Saver callback <a name="saver-callback"></a>

Saving mostly should be done on single samples and not on the batches. It is
introduced inside of the `SaverCallback` interface:

```python
import nucleus7 as nc7

class NewSaverCallback(nc7.coordinator.SaverCallback):

    def save_sample(self, **sample_data):
        save_name = self.save_name
        # save the data to save_name
```

It uses method `split_batch_inputs` to split the inputs of batches, by first
flattering the result and then split it to samples and unflatten back. If some
of the inputs are not in batch-wise fashion, it is possible to set the flag
`not_batch_keys` to point to that keys.

One more important is that the save_name is abstracted away from the developer.
It is set on every sample as following:

- if save_names was provided as a key, it will use it and add to `self.log_dir`
together with save_prefix and suffix. 
- if it was not provided, then it will be inferred out of the epoch, iteration
and sample number.
- to add the extension, you need to add it by your own.

Also there is a class for saving the data to tfrecords format, e.g.
`TfRecordsSaverCallback`. It is a standalone and self-contained class, which
will save the data to tfrecords files, where features, e.g. shapes and dtypes
are inferred from input data. It can be used directly. Also it will split the
data to files up to max number of files or when save_name was changed (useful
for sequential data).  

### Buffer callback <a name="buffer-callback"></a>

Sometimes it is a good idea to accumulate the results and then perform some
calculations on them. Therefore there is a `BufferCallback` interface:

```python
import nucleus7 as nc7

class MyBufferCallback(nc7.coordinator.BufferCallback):

    def process_buffer(self, **buffer_data):
        # process the buffer
```

This interface splits the batch data to samples as described before and then
adds single sample to the buffer. When you pass `evaluate` flag to the callback,
it will execute `process_buffer` method on the buffer and free it afterwards.
So buffer processing can be controlled from outside. It is possible to use
different buffer interfaces for it.

It can be useful to deal with sequence data or to collect the data statistics.

### Callbacks as SessionRunHook <a name="callback_training"></a>

Even if same callbacks can be used for all training, evaluation and inference,
there is a difference *under the hood*, how they are executed inside of
different coordinators: *Inferer* calls each callback explicitly after
each sess.run,
where *Trainer* hides these calls by providing converting callbacks to
`tf.train.SessionRunHook` and providing them to
`tf.train.MonitoredTrainingSession`.
This conversion is done by using of method `convert_callback_to_session_hook`.

There are also default `tf.train.SessionRunHook` defined for *Trainer*:

* SummarySaverHook - saves summaries to *train* or to *eval* logging directories
depending on the mode and aligned to number of proceeded samples
* MetricUpdateHook - evaluates metrics ops (e.g. update_op that are not a part
of train / eval graphs per default, but are used inside of metrics)

From user or developer point of view, you do not need to adapt callback
implementation for training and can use same on for both - training and
inference. Only difference is that inside of training you can use also
`self.summary_writer` (`tf.train.FileWriter`) to write your values to .event
files.

### KPIEvaluator as callback <a name="kpi-callback"></a>

You can also include the `KPIEvaluator` configs as a callback configs.
**nucleus7** will convert them to `KPIEvaluatorCallback` and so it will run
the evaluation as intended on every iteration. Even if you can include it for
both training and evaluation, it is encouraged to use it only as a evaluation
callback, since it may take lot of time and so will slow down the training.

Calculated KPIs will be automatically added to tensorfboard summaries and
will also be saved together with the exported model as evaluation results.

## CallbacksHandler <a name="callbacks-handler"></a>

`CallbacksHandler` takes care of execution of callbacks and first creates
a graph (DNA helix) of the callbacks and sorts them in topological order and
the calls it's methods in the right order:

`begin` &rightarrow;
{*repeat* `on_iteration_start` &rightarrow; `iteration` &rightarrow;
`on_iteration_end`} &rightarrow; `end`

## Trainer <a name="trainer"></a>

As you can see from the name, **Trainer** is used to launch and coordinate
training / evaluation process and has following tasks:

- connect dataset to model and callbacks
- create tensorflow graph for training, evaluation and inference
- create checkpoints after every evaluation run
- export model after every run

### Using of estimator API <a name="using-of-estimator-API"></a>

**Trainer** is using `tf.estimator.Estimator` API for training and evaluation.
It builds model_fn inside of `self.get_model_fn `using *ModelHandler*.
That allows to use directly same code for distributed and local
training / evaluation.

Evaluation is triggered once checkpoint is written. In local mode, session will
be closed and new one created with restored checkpoint on mode switch
(train -> eval -> train...). Summaries are written according to summary step
(global step with offset). Steps for training and evaluation are synchronized
in that way, that last step for evaluation and training in one epoch is the same
number and so evaluation step has initial offset of difference of iterations
for evaluation and training.

### Export of the model <a name="model-export"></a>

After every evaluation, the model is exported as following:
- using [saved_model][saved_model] model format under the
`project_dir/saved_models` directory together with:
    - current losses
    - current KPI values
    - current global step
- using meta graph and checkpoints, which are saved to
`project_dir/checkpoints` directory (inference meta graph is saved only once)
together with names of input and output nodes

### Profiling <a name="profiling"></a>

Sometimes it is needed to profile training. You can activate the
[ProfilerHook][profiler_hook] inside of `trainer.json` by adding its
constructor attributes to following section (do not provide
output_dir!):

```json
{
  "run_config": {
    "profile_hook_config": {
      "save_steps": "...",
      "show_memory": "...",
      "...": "..."
    }
  }
}
```

It will create `timeline-.json` file inside of project directory and you
can load it using Chrome.

## Inferer <a name="inferer"></a>

It is also pretty obvious, **Inferer** coordinates inference
process. And it uses `tf.contrib.predictor` API to make predictions.

There are some differences to *Trainer*:

* **Inferer** does not build any tensorflow graph, but restores it from saved
model or meta graph and checkpoints
* it uses only one GPU if available as for now
* uses *DataFeeder* instead of *Dataset* to feed the data to placeholders
* *Callbacks* are executed explicitly
* Can handle list of feed_dicts and perform one iteration for each feed_dict and
then collapse outputs before passing them to callbacks
* if you want to store the results, you need to provide corresponding Callback

By default, data prefetching, network itself and callbacks are running
in separate processes and communicate using queues. But if you want to
use single process, you can enable it by adding following in your
`inferer.json` config file:

```json
{
  "run_config": {
    "use_multiprocessing": false
  }
}
```

### Tensorrt support <a name="tensorrt-support"></a>

If you have installed tensorrt ([see how][Install]), you can enable it
during inference. In that case, it will create tensorrt graph out of
saved model and use it. To enable it, you can provide the following
config inside of `inferer.json`:

```json
{
  "tensorrt_config": {
    "use_tensorrt": true,
    "max_batch_size": "...",
    "...": "..."
  }
}
```

or you can pass the `nc7.coordinator.configs.TensorrtConfig` to the
`Inferer` constructor.

To refer to parameters and what they mean, see tensorflow tensorrt
[code][tf-tensorrt-link]
