Data handling
=============

- [FileList](#file-list)
- [Dataset](#dataset)
    - [DatasetFileList](#dataset-file-list)
    - [DatasetTfRecords](#dataset-tfrecords)
    - [DatasetMix](#dataset-mix)
- [DataFeeder](#datafeeder)
    - [DataFeederFileList](#datafeeder-file-list)
- [Alternative data pipeline](#alternative-data-pipeline)
    - [DataReader](#data-reader)
    - [TfRecordsDataReader](#tfrecords-data-reader)
    - [DataProcessor](#data-processor)
    - [RandomAugmentationTf](#random-augmentation)
    - [DataPipe](#data-pipe)
    - [Dataset as DataPipe](#dataset-as-data-pipe)
    - [DataFeeder as DataPipe](#datafeeder-as-data-pipe)
- [DataFilter](#data-filter)

Inside of *nucleus7*, data is handled in different ways for training and
inference. So *Trainer* receives inputs from *Dataset*, where samples are
prefetched in separate thread using `tf.data.Dataset` and then feeded through
`tf.data.Iterator` to model, where *Inferer* uses classical way - it has
placeholder nodes and data is feeded using seed_dict itself
(so no prefetching / buffering is performed).

## FileList <a name="file-list"></a>

`FileList` is an object to handle the file names:
* match them
* perform initial shuffle
* downsample them, e.g. select every 10th
* shuffle it
* sort them
* etc.

Provided file lists, e.g. `nc7.data.FileList` and
`nc7.data.FileListExtendedMatch` are already enough for most of the situations,
but if you want to have the different file names structure, then you can
override following methods (if you do not override `sort_fn` method, it's sort
pattern will be the same as match pattern):

```python
import nucleus7 as nc7

class NewFileList(nc7.data.FileList):

    def match_fn(self, path: str, key: str):
        ...
        return match_pattern

    def sort_fn(self, path: str, key: str):
        ...
        return sort_pattern
        
```

As a bonus: the generated file names are saved to project artifacts
automatically :)

## Dataset <a name="dataset"></a>

[**Dataset**](dataset.py) class is used to get data and feed it to training model.
It does use `tf.train.Dataset` under the hood.


It is also a *Nucleotide*, but you need to define generated_keys only,
as it does not have any successor nodes:

```python
import tensorflow as tf

import nucleus7 as nc7

class NewDataset(nc7.data.Dataset):
    generated_keys = ["keys to generate"]

    def create_initial_data(self) -> tf.data.Dataset:
        ...
        return data
        
```

The whole magic of the `Dataset` is done under the hood - you only need to
override the `create_initial_data` method (you need to return the
`tf.data.Dataset` object, where you have dict of data),
where you define how to create one sample of the dataset!

Magic is following:
* generate data using the `create_initial_data` method
* shuffle the dataset if needed
* repeat it
* prefetch it to memory
* cache to the disk if needed
* combine to batch

This generic class is there to allow to feed different data
(like generate data on the fly or take if from some RL environment),
but it is not so interesting for common tasks, where you need to read the data
from files. For it see [next](#dataset-file-list).

### DatasetFileList <a name="dataset-file-list"></a>

If you want to read the data from file pairs (e.g. image file and label file),
you need 2 things:

* `FileList` object
* implement logic how to read that data

And this is what the **DatasetFileList** handles:

```python
import nucleus7 as nc7

class NewDatasetFileList(nc7.data.DatasetFileList):
    file_list_keys = ["list of file list keys that this dataset can handle"]
    generated_keys = ["keys to generate"]

    def read_raw_data_from_file(self, **sample_fnames) -> dict:
        ...
        return data
```

So you need to overwrite the `read_raw_data_from_file` method,
which takes dict of file name pairs for one sample as kwargs with keys
from `file_list_keys` and returns the dict of the tensorflow tensors out of it.
You even don't need to create a `tf.data.Dataset` object - it will be created
for you from the data you return.

### DatasetTfRecords <a name="dataset-tfrecords"></a>

This is the extension to handle the tfrecords data format. tfrecords is
preferred over reading the single raw files.

```python
from typing import Optional

import nucleus7 as nc7


class NewDatasetTfRecords(nc7.data.DatasetTfRecords):
    file_list_keys = ["list of file list keys that this dataset can handle"]
    generated_keys = ["keys to generate"]

    def get_tfrecords_features(self) -> dict:
        ...
        return features

    def get_tfrecords_output_types(self) -> Optional[dict]:

        return None

    def postprocess_tfrecords(self, **data) -> dict:
        return data
``` 

Basically you need to override the `get_tfrecords_features` method to decode
tfrecords features from the tfrecord files and if the output types are not
clear afterwards, override `get_tfrecords_output_types` method.
You can also perform some posptocessing, e.g. renaming the keys etc.
inside of the `postprocess_tfrecords` method.

### DatasetMix <a name="dataset-mix"></a>

`DatasetMix` adds the functionality to combine different datasets also with
different features. You do not need to inherit it, just use the datasets you
want ald provide them to constructor of `DatasetMix` with also possible
sampling weights. It will sample with that probability samples from all the
datasets and then combine then to a batch. For inference, it will sample the
samples equally from all datasets.

if you want to use different Datasets on the same bunch of data,
you need to provide the same `FileList`s to that datasets and so automatically
it will try to merge the data for all the samples. See `DatasetMix` docs for
more information. 

## DataFeeder <a name="datafeeder"></a>

As it stays in the name, it 'feeds' the data directly to the model placeholders.

As *Dataset*, it is also a *Nucleotide* and you need to define generated_keys
only and it has slightly different inheritance structure, too:

```python
import nucleus7 as nc7

class NewDataFeeder(nc7.data.DataFeeder):
    generated_keys = ["keys to generate"]

    def build_generator(self):
        ...
        return generator

```

Same as *Dataset*, batching should be identical for all data types.
So you need to implement only function to generate the data - `build_generator`
, which should create a data generator, as it may be bad idea to store all of
the inputs in the memory :)

Also as `Dataset`, this generic class is there mostly as an interface.

### DataFeederFileList <a name="datafeeder-file-list"></a>

This feeder allows to read the data from file name and feed it to the model
placeholders:

```python
import nucleus7 as nc7

class NewDataFeeder(nc7.data.DataFeederFileList):
    file_list_keys = ["list of file list keys that this dataset can handle"]
    generated_keys = ["keys to generate"]

    def read_element_from_file_names(self, **fnames) -> dict:
        ...
        return read_data
```

So you need to override just one method is how to read the file names and create
the dict of sample

## Alternative data pipeline <a name="alternative-data-pipeline"></a>

### DataReader <a name="data-reader"></a>

`DataReader` is an interface to read a sample from file names or to generate it.
Files to read will be provided and so it allows to have multiple readers
attached to the same file pairs. It has following interface:

```python
import nucleus7 as nc7

class NewDataReader(nc7.data.DataReader):
    is_tensorflow = True # if it should be used for the training or False otherwise

    def read(self, **fnames):
        return data_from_file_names

```

`is_training` is a class attribute, which specified if this particular
implementation is tensorflow implementation, e.g. can be used to apply on
`tf.data.Dataset` or can be used only for inference like projects.

Also `DataReaders` are not allowed to have any inbound nodes - this should
generate data from file names or from the seed.

### TfRecordsDataReader <a name="tfrecords-data-reader"></a>

For the training, it is more straightforward to extract all the available data
to tfrecord files and then read only what is needed. For it, there is a
`DataReaderTfRecords` interface (is similar to the `DatasetTfRecords`):

```python
import nucleus7 as nc7

class NewDataReaderTfRecords(nc7.data.TfRecordsDataReader):

    def get_tfrecords_features(self) -> dict:
        ...
        return features

    def get_tfrecords_output_types(self) -> Optional[dict]:

        return None

    def postprocess_tfrecords(self, **data) -> dict:
        return data
```

Before, the `tf.data.TFRecordDataset` will be generated and so `self.read`
method will be mapped on the tfrecords dataset, which allows to have multiple
readers from the same tfrecord file.

### DataProcessor <a name="data-processor"></a>

`DataProcessor` is an interface to process the data. It is different to
`DataReader` in that, that it takes results from `DataReaders` and modifies
them. It works also sample-wise, e.g. **not** on the batched data!.

```python
import nucleus7 as nc7

class NewDataProcessor(nc7.data.DataProcessor):
    is_tensorflow = True # if it should be used for the training or False otherwise

    def process(self, **data):
        return result
```

### RandomAugmentationTf <a name="random-augmentatio"></a>

This `DataProcessor` is used to apply random augmentations and stack them
together. In addition it can be used to generate random data during
training / evaluation.

```python
import nucleus7 as nc7

class NewRandomAugmentation(nc7.data.RandomAugmentationTf):
    random_variables_keys = ["noise", "other_random_var"]

    def __init__(self, *,
                 augmentation_probability: float = 0.5,
                 **kwargs):
        pass

    def augment(self, **data):
        noise = self.random_variables["noise"]
        other_random_var = self.random_variables["other_random_var"]
        augmented_result = f(data)
        return augmented_result

    def not_augment(self, **inputs) -> Dict[str, tf.Tensor]:
        return inputs

    def create_random_variables(self) -> Dict[str, tf.Tensor]:
        return {"noise": tf_random,
                "other_random_var": tf_random}
```

The augmentation flag controls whether the augmentation is applied.
If the augmentation flag evaluates to `True` then `augment(...)` is called,
otherwise `not_augment(...)` is called.
The augmentation flag can be set in 2 ways:

1. Pass `augmentation_probability` to the constructor.
The augmentation flag is set to `True` with probability
`augmentation_probability`. `augmentation_probability` is overridden by the
`augment` key if provided (see 2).
2. Provide the tf.bool `augment` key. If provided, this overrides the
`augmentation_probability` parameter.
This allows chained augmentations, i.e. use the same augmentation flag value
for multiple augmentations.
Concretely, a `RandomAugmentationTf` outputs the augmentation flag value in the
`augment` key. This key is then passed to the next `RandomAugmentationTf`
as the `augment` incoming key, overriding any passed `augmentation_probability`.

Random variables can also be chained. To do so, define their keys inside the
`random_variables_keys` class attribute and define how to
generate them inside `create_random_variables()`. These variables will be used,
if provided on instantiation, otherwise they will be generated
(as defined in `create_random_variables()`) and passed further as
generated keys. To access the variables inside `augment(...)`,
use `self.random_variables[random_varialble_key]`.

### DataPipe <a name="data-pipe"></a>

`DataPipe` is a container to hold all the `DataReaders` and `DataProcessors`
and to create the whole pipeline. It is built automatically and has no other
constructor parameters. It also checks if provided components can be combined:

- All components should be either tensorflow or not tensorflow components.
- All `DataReaders` should be tfrecords readers or not tfrecord readers.

### Dataset as DataPipe <a name="dataset-as-data-pipe"></a>

If the `DataPipe` is a tensorflow pipeline, it is possible to create `Dataset`
out of it directly. Difference is that if dataset has a `FileList` inside, first
`tf.data.Dataset` from file list will be generated (or tfrecords dataset) and
then the `DataPipe` will be mapped onto it. Afterwards the dataset will be
batched etc. as in original `Dataset`.

Mapping of `DataPipe` component results in form
{component_name: component_results} can be remapped to `Dataset`
generated keys using `output_keys_mapping`.

### DataFeeder as DataPipe <a name="datafeeder-as-data-pipe"></a>

If the `DataPipe` is a non-tensorflow pipeline, it is possible to create
`DataFeeder` out of it directly.
Difference is that if data feeder has a `FileList` inside, first
generator from file list will be generated and
then the `DataPipe` will be applied to it. Afterwards the data will be
batched etc. as in original `DataFeeder`.

Mapping of `DataPipe` component results in form
{component_name: component_results} can be remapped to `DataFeeder`
generated keys using `output_keys_mapping`.

## DataFilter <a name="data-filter"></a>

If you want to filter the data inside of any data module object, like 
`FileList`, `DataFeeder`, `Dataset`, `DataExtractor`, you can use
`DataFilter` for it. For it you need just to override one method:

```python
import nucleus7 as nc7

class OddIndexSelector(nc7.data.DataFilter):
    def predicate(self, index, **other_inputs) -> bool:
        return index % 2


dataset = ...

dataset.add_data_filter(OddIndexSelector())
```

If you include this filter to your Dataset, it will filter only samples,
that have odd index key. Please note, that it is possible to use ths same
`DataFilter` in different classes, but you need to make sure, that
implementation works on tf tensors if you want to use it inside of
`Dataset`.

If you want to remap inputs from object that uses DataFilter, e.g. from Dataset
or DataFeeder to filter predicate method, use `predicate_keys_mapping` argument.

It is also possible to have multiple filters, which will act as AND
filters. 
