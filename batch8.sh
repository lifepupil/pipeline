#!/bin/bash 

#$ -N pac_fc2o2_3 
#$ -cwd
##$ -pe smp 1
#$ -l h_vmem=40G


trap "rsync -av $TMPDIR/results/ $SGE_O_WORKDIR/proc_fc2o2_3/;exit" SIGUSR1


cd $TMPDIR 
mkdir input 
mkdir results 
cd results
mkdir FC2O2
mkdir O2FC2
cd $TMPDIR
rsync -av $SGE_O_WORKDIR/ref_fc2o2_3/ input/ 
rsync -av $SGE_O_WORKDIR/pacdat/ input/

cd /ddn/crichard/pipeline/
python tensorpac_HPC_chan_pair.py 

rsync -av $TMPDIR/results/ $SGE_O_WORKDIR/proc_fc2o2_3/



