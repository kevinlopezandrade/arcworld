# [WIP]
## Dependencies
You need pytorch, torchmetrics, chardet, hydra, wandb to be able to run the script.

If have you installed the package using the README.md at the root directory.
Then simply run.
```bash
$ conda activate arcworld
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
$ pip install torchmetrics
$ pip install chardet
$ pip install hydra-core
$ pip install wandb
```

## Configurations
Hydra config files can be found in the conf/ directory.

## Training
You can either run in distributed mode or in one-gpu mode.
### One GPU
For running the training in one gpu simple run
```bash
python train.py
```
and it will use the configuration specified in config.yaml.
Check the hydra documentation to see how to override and compose config files
from the command line.

### One-Node Multiple-GPU
In order to use DDP you need to spawn independent Python processes running the
distributed.py script. Each process will run a separate copy of distributed.py,
but they will communicate and coordinate through the PyTorch DDP framework.
This is essential for distributing the work across multiple devices or nodes
and for achieving parallelism.

The **rank** argument is an integer that uniquely identifies each process in a
distributed setup. It ranges from 0 to **world_size** - 1. This allows each
process to know its own identity and determine its role in the training
process.  Only the process with rank 0 (a.k.a master) will logging the results
to wandb, and save the weights.

The **world_size** argument specifies the total number of processes (a.k.a worker)
participating in the distributed training. It should be the same for all
processes. It is typically equal to the number of GPUs or nodes available in
your cluster.

For example, in a setup with four GPUs, you might run your script with the
following command for each GPU:
```bash
$ python distributed.py rank=0 world_size=4
$ python distributed.py rank=1 world_size=4
$ python distributed.py rank=2 world_size=4
$ python distributed.py rank=3 world_size=4
```

The enviroment variable **CUDA_VISIBLE_DEVICES** controls which GPUS
are avalaible to the process (a.k.a worker). If not specified then
all the GPUS will be avalaible to worker. For example in a setting
where there is one node and 8 gpus, assuming that GPU0 and GPU1 have
all their memory occupied, you could launch two workers for training
in 2 gpus, in GPU4 and GPU7 as follows:

```bash
$ CUDA_VISIBLE_DEVICES=4,7 python distributed.py rank=0 world_size=2
$ CUDA_VISIBLE_DEVICES=4,7 python distributed.py rank=1 world_size=2
```
Then in pytorch gpus are 0-indexed, so "cuda:0" will refer to GPU4 and
"cuda:1" will refer to GPU7.

## GPU Usage
There is a small script **display-gpu-info.sh** that displays information about
the processes that are using the GPUs in the node. You could for instance run
```bash
$ watch -n 0.5 "./display-gpu-info.sh | grep kevin"
```
to display information about all the processes launched by user *kevin* that
are using a GPU in the node. And update that information every 0.5 seconds.
