# ANL DQ Documentation
======================

# Set Up
Setup to use this code on ANLs LCRC.

## Environment setup
Set up the common anl-dq python enviroment. If not yet done, create a working directory and clone this repository:
```
mkdir anl-hep-ai-ml
git clone git@github.com:UniHD-CEG/DeepHYDRA.git
``` 

Now, create the common conda environment, run:
```
module load anaconda3
conda env create -f DeepHYDRA/envs/anl-dq.yml
```

## Session Setup
Setup to run on ANL's LCRC. 

Source the 
```
setup_anl.sg
``` 
script. 
