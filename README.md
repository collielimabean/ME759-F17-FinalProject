ME759 Fall 2017 Final Project - Distributed Task Library
========================================================

## Default Project 1 Problem Statement Summary:
Suppose we have a mainJobA that is computing a time integration of 
a deforming elastic body. This integration requires a Jacobian J, which is
computationally expensive to generate. A helperJobB is tasked to update that
Jacobian J. HelperJobB may not be fast enough to update the entire Jacobian
before each timestep. 

We should have flexibility on where mainJobA and helperJobB are run.

After talking with Professor Negrut, we have refined this to be something
similar to TBB or OpenMP. In this project, however, the programmer can create a parent task that can spawn child subtasks on either the host, GPU, or a
different node entirely. These subtasks must complete before the parent task can move on.

## Building
Run `make`. It will generate two binaries (`mpi_task` and `cpu_task`) in the top level directory,
which are the programs in the examples folder. 

The `cpu_task` binary can be run on any machine, as it spins off threads only.

The `mpi_task` must be run with sbatch or srun, as it requires a node with a GPU. An example shell script is given below:

```bash
#!/bin/bash
#SBATCH --partition=slurm_shortgpu
#SBATCH --time=0-00:15:00 # run time in days-hh:mm:ss
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --error=sbatch.err
#SBATCH --output=sbatch.out
#SBATCH --gres=gpu:1

./mpi_task
```
