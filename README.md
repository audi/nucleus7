<a>
    <img src="icon.png" alt="nucleus7 logo" title="nucleus7" align="right" height="120" />
</a>


nucleus7
========

Welcome to **nucleus7** - library for exchangeable and reproducible development
of Deep Learning models built on top of tensorflow!

- [Why and when to use it?](#why-and-when)
- [Installation](./INSTALL.md)
- [Glossary](#glossary)
- [Project structure](./ProjectStructure.md)
- [Nucleotide and co.](./nucleus7/core/README.md)
- [Data flow](./nucleus7/data/README.md)
- [Model components](./nucleus7/model/README.md)
- [Metrics and KPIs](./nucleus7/kpi/README.md)
- Training and inference Coordination:
    * [Coordinators like Trainer and Inferer](./nucleus7/coordinator/README.md)
    * [Optimization control](./nucleus7/optimization/README.md)
- nucleus7 project
    * [Configs](./ProjectConfigs.md)
    * [Structure](./ProjectStructure.md)
    * [Execution](./ProjectExecution.md)
- [Nucleotide development](./Development.md)
- [Tutorials](#tutorials)
- [Mlflow integration](#Mlflow-integration)
- [Known bugs](#known-bugs)
- [Documentation](https://aev.github.io/nucleus7/)
- [Contribution](./CONTRIBUTING.md)


## Why and when to use it? <a name="why-and-when"></a>

* If you want to spend your time on important stuff (architecture design, paper
  implementation) and not on how to launch tf.Session on multi gpus :)

* If you want your code to be able to be used by other developers without any
  problems and without spending hours and hours of their time to understand where
  the training begins :)

* If you want to use modules (like architectures, losses, metrics etc.) developed
  by yourself and by others in *plug-and-play* mode

* If you want to load classification models from your neighbour and use those
  weights for your dog-cat object detection (of course, it is better that your
  neighbour also uses **nucleus7**)

## Glossary <a name="glossary"></a>

Names of **nucleus7** components are based on nucleus structure: nucleotide,
gene, dna helix etc.

**Nucleotide** is a building block of **nucleus7**. It has a modular structure
and also has a data flow interfaces, e.g. which data does it take and which
data does it output. There are different kinds of nucleotides, like ModelPlugin
and CoordinatorCallback, which serve for different tasks, like neural network
architecture and callbacks to execute after each iteration.

**Gene** is a combination of same type nucleotides, like plugins gene has all the
ModelPlugin nucleotides inside etc. There may be many different genes in one
model. This abstraction allows also to restrict the gene-to-gene connections,
e.g. data can flow only from one gene to other and other connection is not
allowed, e.g. ModelPlugin cat take inputs from the Dataset, but not from
ModelLoss and ModelLoss can take inputs from ModelPlugin but not from Callbacks.

**DNA helix** called the graph constructed of all model nucleotides.
DNA will sort the nucleotides in each gene 

## Tutorials  <a name="tutorials"></a>

You can find some totorials how to use **nucleus7** for your projects
inside of [tutorials folder](./tutorials) in the root directory.

**DO NOT FORGET**- you need to have **nucleus7** inside of your `PYTHONPATH`,
e.g. (see [Install section](./INSTALL.md))

To run notobooks:

```bash
jupyter notebook
```

## Mlflow_integration <a name="Mlflow-integration"></a>

By default, when you start the nucleus7 project, e.g. training or inference,
it will create a folder mlruns inside of the root of project_dir, e.g.
for `/path/to/root/project_dir` it will create `path/to/root/project_dir` and
add the experiment there with name of the `project_dir` directory or
it will search for "project_name" inside of "nucleus7_project.json" file
under `PROJECT_NAME` key. This
will make sure, that you track everything to mlflow and so you can start mlflow
from `path/to/root` folder:

```bash
cd /path/to/root
mlflow ui
```

But you also can set `MLFLOW_TRACKING_URI` environment variable to point to the
URI with main mlflow tracker (see mlflow help for more details) and it will
create the experiment there (if experiment with that name exists, than it will
add the run to it):

```bash
export MLFLOW_TRACKING_URI='path/to/uri'
nc7-train /path/to/root/project_dir
```

## Known bugs <a name="known-bugs"></a>

- If you have tensorflow(-gpu) > 1.11, then you can issue the pylint 
no-member warnings issues for tensorflow estimator due to the fact, that
estimator API is still accessible as in tensorflow 1.11, but is officially
legacy there and is moved to tensorflow_estimator. Since we maintain code for
nucleus7 for tf >= 1.11, this bug cannot be solved easily and is more cosmetic
issue. When tensorflow 2.0 will come out, tensorflow 1.11 support will be 
dropped and this issue will be removed. Inside of the testing, pylint is used
only for tensorflow 1.11, so it does not raise an issue.

- Since by starting the inference project (nc7-infer), the symlinks are
generated, it causes the errors with the directory cleaning inside of
tf.TestCase. But since the temporary folders are used (/tmp/...), this is also
only a cosmetic issue, since this folders are cleaned automatically by the OS
