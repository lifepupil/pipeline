#!/bin/bash 

#$ -N o2_pac_2 
#$ -cwd
##$ -pe smp 1
#$ -l h_vmem=40G


trap "rsync -av $TMPDIR/results/ $SGE_O_WORKDIR/proc_o2_2/;exit" SIGUSR1


cd $TMPDIR 
mkdir input 
mkdir results 
rsync -av $SGE_O_WORKDIR/ref_o2_2/ input/ 
rsync -av $SGE_O_WORKDIR/pacdat/ input/

cd /ddn/crichard/pipeline 
python tensorpac_HPC_O2.py 

rsync -av $TMPDIR/results/ $SGE_O_WORKDIR/proc_o2_2/



