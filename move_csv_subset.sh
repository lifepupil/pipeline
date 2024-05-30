#!/bin/bash

cd /ddn/crichard/pipeline

# TO MAKE NEW SOFT LINK IN pipeline/
# ln -s /ddn/crichard/eeg_csv/FC2/ ./ref_fc2


ls /ddn/crichard/pipeline/ref_o2/ | wc -l
ls /ddn/crichard/pipeline/ref_o2_1/ | wc -l
ls /ddn/crichard/pipeline/ref_o2_2/ | wc -l
ls /ddn/crichard/pipeline/ref_o2_3/ | wc -l

mv $(ls -d /ddn/crichard/pipeline/ref_o2/* | head -4920) /ddn/crichard/pipeline/ref_o2_1/
mv $(ls -d /ddn/crichard/pipeline/ref_o2/* | head -4920) /ddn/crichard/pipeline/ref_o2_2/
mv $(ls -d /ddn/crichard/pipeline/ref_o2/* | head -4920) /ddn/crichard/pipeline/ref_o2_3/

ls /ddn/crichard/pipeline/ref_o2/ | wc -l
ls /ddn/crichard/pipeline/ref_o2_1/ | wc -l
ls /ddn/crichard/pipeline/ref_o2_2/ | wc -l
ls /ddn/crichard/pipeline/ref_o2_3/ | wc -l
