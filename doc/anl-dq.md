# ANL DQ Documentation
======================

# Set Up
Setup to use this code on ANLs LCRC.

## Environment setup
Set up the common anl-dq python enviroment. 
This only needs to be done one time.
If not yet done, create a working directory and clone this repository:
```
mkdir anl-hep-ai-ml
cd anl-hep-ai-ml
git clone git@github.com:UniHD-CEG/DeepHYDRA.git
``` 

Now, create the common conda environment, run:
```
module load anaconda3
conda env create -f DeepHYDRA/envs/anl-dq.yml
conda activate anl-dq 
```

## Session Setup
In your working directory, source
```
. DeepHYDRA/setup_anl.sh
``` 
script. 

# Notes, Tips & Tricks
## Start Jupyter Lab on your favorit PORT (for example the default of 8888)
```
jupyter lab --no-browser --port PORT
```

