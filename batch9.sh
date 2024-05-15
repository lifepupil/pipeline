#!/bin/bash 

#$ -N pac_9 
#$ -cwd
##$ -pe smp 1
#$ -l h_vmem=40G


trap "rsync -av $TMPDIR/results/ $SGE_O_WORKDIR/processed9/;exit" SIGUSR1


cd $TMPDIR 
mkdir input 
mkdir results 
rsync -av $SGE_O_WORKDIR/reference9/ input/ 
rsync -av $SGE_O_WORKDIR/pacdat/ input/

cd /ddn/crichard/pipeline 
python tensorpac_hpc1.py 

rsync -av $TMPDIR/results/ $SGE_O_WORKDIR/processed9/



