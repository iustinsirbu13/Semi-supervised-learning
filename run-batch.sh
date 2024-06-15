#!/bin/bash

sbatch -p ucsx --gres gpu:1 --time=360 --cpus-per-task=1 --mem=32G ./apptainer-exec.sh $1

