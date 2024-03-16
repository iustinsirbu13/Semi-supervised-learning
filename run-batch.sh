#!/bin/bash

sbatch -p ucsx --gres gpu:1 --time=360 --cpus-per-task=5 --mem-per-cpu=16G ./apptainer-exec.sh $1

