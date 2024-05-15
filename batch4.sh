#!/bin/bash 

#$ -N pac_4 
#$ -cwd
##$ -pe smp 1
#$ -l h_vmem=40G


trap "rsync -av $TMPDIR/results/ $SGE_O_WORKDIR/processed4/;exit" SIGUSR1


cd $TMPDIR 
mkdir input 
mkdir results 
rsync -av $SGE_O_WORKDIR/reference4/ input/ 
rsync -av $SGE_O_WORKDIR/pacdat/ input/

cd /ddn/crichard/pipeline 
python tensorpac_hpc1.py 

rsync -av $TMPDIR/results/ $SGE_O_WORKDIR/processed4/



