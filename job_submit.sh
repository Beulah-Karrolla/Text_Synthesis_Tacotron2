#!/bin/bash

#SBATCH --time=04:00:00 
#SBATCH --job-name=TTS_1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100-32g:2
#SBATCH --cpus-per-gpu=2
#SBATCH --account=PAS2400
#SBATCH --array=0-1:1
#SBATCH --mail-type=BEGIN,END,FAIL


python tts_main.py
