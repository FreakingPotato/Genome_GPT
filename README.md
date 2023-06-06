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
2. adding ArgumentParser [DONE]
3. adding profiling code: Nsight [DONE]

    -- keep the block length short to get the result quicker, however, the profile info maybe invalid for block with different length

    -- try profiling with less iterations (currently, profiling 5 epochs will generate 1G file)

    -- nsys profiling is very memory intensive

        nsys profile python3 genome_gpt.py -n_blocks 10 -b 10240

4. adding multi-node code: Horovod, NCCL?? Huggingface Accelerate/Microsoft DeepSpeed/ Ignite/ 

    https://opus.nci.org.au/display/DAE/Pytorch+using+Horovod
5. optimizer: Sophia [DONE]

    adopted the code from: https://github.com/karpathy/minGPT

6. get baseline of current model: training speed, perplexity (loss) 

7. exon sequencing dataset


8. (optional) Model Parallelism: nvidia transformer-lm
