# selecting how much memory, time and computation resources are need

#!/bin/bash
#PBS -l select=1:ncpus=64:mem=50gb:scratch_local=32gb
#PBS -l walltime=120:00:00
#PBS -N name_of_the_run
#PBS -m abe
# initialize the required application (e.g. Python, version 3.4.1, compiled by gcc)

# paths to the data and files in the meta directories
trap 'clean_scratch' TERM EXIT
DATADIR=/storage/projects/CVUT_Fsv_AO/Matyas_BP/results/train
DATADIR_arrays=/storage/projects/CVUT_Fsv_AO/Matyas_BP/data/pix3d/imgs_256.npz
DATADIR_voxels=/storage/projects/CVUT_Fsv_AO/Matyas_BP/data/pix3d/voxels_32.npz

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

# name of the script that is going to be run
cp $DATADIR/model_1_train.py $SCRATCHDIR
cp $DATADIR_arrays $SCRATCHDIR
cp $DATADIR_voxels $SCRATCHDIR

cd $SCRATCHDIR

# creating directory for saving results
mkdir output

module add python/3.6.2-gcc

# running the script
python3 model_1_train.py

cp -r $SCRATCHDIR/output $DATADIR
clean_scratch
exit