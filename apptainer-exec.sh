#!/bin/bash

apptainer exec --nv ssl.sif python train.py --c $1
