#!/bin/bash 

#$ -N fc2_pac_1 
#$ -cwd
##$ -pe smp 1
#$ -l h_vmem=40G


trap "rsync -av $TMPDIR/results/ $SGE_O_WORKDIR/proc_fc2_1/;exit" SIGUSR1


cd $TMPDIR 
mkdir input 
mkdir results 
rsync -av $SGE_O_WORKDIR/ref_fc2_1/ input/ 
rsync -av $SGE_O_WORKDIR/pacdat/ input/

cd /ddn/crichard/pipeline 
python tensorpac_HPC_FC2.py 

rsync -av $TMPDIR/results/ $SGE_O_WORKDIR/proc_fc2_1/



