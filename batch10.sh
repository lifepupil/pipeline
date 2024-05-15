#!/bin/bash 

#$ -N pac_10 
#$ -cwd
##$ -pe smp 1
#$ -l h_vmem=40G


trap "rsync -av $TMPDIR/results/ $SGE_O_WORKDIR/processed10/;exit" SIGUSR1


cd $TMPDIR 
mkdir input 
mkdir results 
rsync -av $SGE_O_WORKDIR/reference10/ input/ 
rsync -av $SGE_O_WORKDIR/pacdat/ input/

cd /ddn/crichard/pipeline 
python tensorpac_hpc1.py 

rsync -av $TMPDIR/results/ $SGE_O_WORKDIR/processed10/



