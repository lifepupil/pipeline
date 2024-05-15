#!/bin/bash

cd /ddn/crichard/pipeline
ls /ddn/crichard/pipeline/references/cleaned_data/ | wc -l
ls /ddn/crichard/pipeline/reference1/ | wc -l
ls /ddn/crichard/pipeline/reference2/ | wc -l
ls /ddn/crichard/pipeline/reference3/ | wc -l
ls /ddn/crichard/pipeline/reference4/ | wc -l
ls /ddn/crichard/pipeline/reference5/ | wc -l
ls /ddn/crichard/pipeline/reference6/ | wc -l
ls /ddn/crichard/pipeline/reference7/ | wc -l
ls /ddn/crichard/pipeline/reference8/ | wc -l
ls /ddn/crichard/pipeline/reference9/ | wc -l
ls /ddn/crichard/pipeline/reference10/ | wc -l

python update_folder_files.py

ls /ddn/crichard/pipeline/references/cleaned_data/ | wc -l
ls /ddn/crichard/pipeline/reference1/ | wc -l
ls /ddn/crichard/pipeline/reference2/ | wc -l
ls /ddn/crichard/pipeline/reference3/ | wc -l
ls /ddn/crichard/pipeline/reference4/ | wc -l
ls /ddn/crichard/pipeline/reference5/ | wc -l
ls /ddn/crichard/pipeline/reference6/ | wc -l
ls /ddn/crichard/pipeline/reference7/ | wc -l
ls /ddn/crichard/pipeline/reference8/ | wc -l
ls /ddn/crichard/pipeline/reference9/ | wc -l
ls /ddn/crichard/pipeline/reference10/ | wc -l

