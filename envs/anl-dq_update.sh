#/bin/bash
conda env export --no-builds | grep -v "^prefix: " > anl-dq.yml
