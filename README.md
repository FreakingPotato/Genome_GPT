# minGPT 
- code adopted from https://github.com/karpathy/minGPT

## data:
- subset of human genome sequence for quick proototyping 
- training dataset is about 1/300 (500m/~3G, 100m/500m, 10m/100m)
- using split command to reduce size of raw data, for example:

        
         split -b10m hg38.500m.100m.aa hg38.500m.100m.10m.

## setup environment:

            conda env create -f genomegpt_env.yml -n genomegpt

            conda activate genomegpt

## train:

            python3 genome_gpt.py

## TODO:

> DAY 1: 
1. checking out the latest minGPT as well as nanoGPT repo
2. adding ArgumentParser 
3. adding profiling code: Nsight??
4. adding multi-node code: Horovod, NCCL??
