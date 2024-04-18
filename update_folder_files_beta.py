# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 14:45:43 2024

@author: lifep
"""

# do_remove_done_from_remaining_files

import coga_support_defs as csd
import os

read_folder, write_folder = 'reference','processed'
SourceEEGfileExtention = 'csv'
TargetEEGfileExtention = 'jpg'

is_os_linux = False
remove_ref_files = False
remove_proc_files = False
copy_ref_files = True



if is_os_linux:
    base_dir = "/ddn/crichard/pipeline/"
    #read_dir = os.environ['TMPDIR'] + '/input/'
    slash = '/'
else:    
    base_dir = 'D:\\COGA_eec\\update_folder_testing\\' #  BIOWIZARD
    slash = '\\'

# # GET MASTER TABLE OUT 
# pacdat = pd.read_pickle(base_dir + which_pacdat)

# # GET FILE LIST WITH GIVEN EXTENSION FROM SOURCE FOLDER 
# existingFiles = csd.get_file_list(base_dir + read_folder, SourceEEGfileExtention)
# cf = [f[1] for f in existingFiles]
human_format_output = []
for i in range(1,11):
    
    all_processed_trg = csd.get_file_list(base_dir + write_folder + str(i), TargetEEGfileExtention)
    fnames_trg = [f[1] for f in all_processed_trg]
    fnames_split = [str.split(fn,'_') for fn in fnames_trg]
    fnames_src = set(['_'.join(fn[0:len(fn)-1]) + '.' + SourceEEGfileExtention for fn in fnames_split])
    rcount = 0 
    pcount = 0
    
    ref_files_src = [base_dir + read_folder + str(i) + slash + f for f in fnames_src]
    ref_files_trg = [base_dir + read_folder + slash + f for f in fnames_src]
    processed_files_src = [base_dir + write_folder + str(i) + slash + f for f in fnames_trg]
    processed_files_trg = [base_dir + write_folder + slash + f for f in fnames_trg]
    
    # MOVE REFERENCE FILES (E.G., EEG .CSVs) BACK TO MAIN REFERENCE FOLDER
    if remove_ref_files:
        for f in range(0,len(ref_files_src)):
            if os.path.isfile(ref_files_src[f]):
                # print(ref_files_src[f] + ' EXISTS')
                os.rename(ref_files_src[f], ref_files_trg[f])
                rcount+=1
            else:
                msg = ('Missing reference' + str(i) + ' file ' + ref_files_src[f])
                print(msg)
                human_format_output.append(msg)
        msg = 'DONE reference' + str(i)  + ' \n'
        human_format_output.append(msg)
        rmsg = 'Moved ' + str(rcount) + ' image files derived from EEG files from ' + write_folder + str(i) + ' to ' + write_folder

    # MOVE PROCESSED FILES (E.G., PAC .JPGs) TO MAIN PROCESSED FOLDER
    if remove_proc_files:
        for f in range(0,len(processed_files_src)):
            if os.path.isfile(processed_files_src[f]):
                # print(processed_files_src[f]+ ' EXISTS')
                os.rename(processed_files_src[f], processed_files_trg[f])
                pcount+=1
            else:
                msg = ('Missing processed' + str(i) + ' file ' + ref_files_src[f])
    
                human_format_output.append(msg)
        msg = 'DONE processed' + str(i)  + ' \n'
        human_format_output.append(msg)
        pmsg = 'Moved ' + str(pcount) + ' EEG channel .CSV files from ' + read_folder + str(i) + ' back to ' + read_folder


    # COPY PROCESSED FILES (E.G., PAC .JPGs) TO MAIN PROCESSED FOLDER
    if copy_ref_files:
        for f in range(0,len(processed_files_src)):
            if os.path.isfile(processed_files_src[f]):
                # print(processed_files_src[f]+ ' EXISTS')
                os.copy(processed_files_src[f], processed_files_trg[f])
                pcount+=1
            else:
                msg = ('Missing processed' + str(i) + ' file ' + ref_files_src[f])
    
                human_format_output.append(msg)
        msg = 'DONE processed' + str(i)  + ' \n'
        human_format_output.append(msg)
        pmsg = 'Moved ' + str(pcount) + ' EEG channel .CSV files from ' + read_folder + str(i) + ' back to ' + read_folder


    human_format_output.append(rmsg)
    human_format_output.append(pmsg +'\n')

# for r in range(len(human_format_output)-1,0,1):
print('\n')
for r in range(0,len(human_format_output)-1):
    print(human_format_output[r])