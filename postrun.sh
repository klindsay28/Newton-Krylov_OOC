#!/bin/bash

cfg_fname=$1

source activate iter_methods
./nk_driver.py --cfg_fname $cfg_fname --resume
