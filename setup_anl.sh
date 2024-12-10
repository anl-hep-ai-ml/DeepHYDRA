module load anaconda3
conda activate anl-dq

# add the repository base address to PYTHONPATH so we can use import STUFF without the need of sys.path.append
export PYTHONPATH=$PYTHONPATH:$(readlink -f $(dirname "${BASH_SOURCE[0]}"))

