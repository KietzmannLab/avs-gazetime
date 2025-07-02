# this script will go through the AVS et results and input and create a data package that can be used for the GUI.
# 1) we will make package that contains the 4080 scenes
# 2) Another part will be the AVS explogs with the transcribed scene captions
# 3) Another part will be the fixation data (events dataframe)

import os 
import sys
import json
import pandas as pd
from avs_analysis.tools.avs_directory_tools import get_data_dirs, get_input_dirs, get_sub_sess_id
from avs_analysis.eye_tracking import avs_prep
import numpy as np

subjects = [1,2,3,4,5]
sessions = np.arange(1,10+1)
output_prefix = "as"
recompute_data_package = True

#get scene input dir 
input_dir = get_input_dirs(server="uos")
normal_image_dir = os.path.join(input_dir, "NSD_scenes_MEG_size_adjusted_925")


output_dir = os.path.join(input_dir, "fix2cap")
if recompute_data_package:
    # wipe the output dir
    if os.path.exists(output_dir):
        print(f"removing {output_dir}")
        os.system(f"rm -rf {output_dir}")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_scene_dir = os.path.join(output_dir, "scenes")
if not os.path.exists(output_scene_dir):
    os.makedirs(output_scene_dir)



#get the et results dir
_, results_dir = get_data_dirs(server="uos")
data_path = results_dir

#get the explogs dir
# make a huge dataframe based on a subset of the events dataframe that contains subject, scene_id, trial, block, duration, start_time, time_in_trial, mean_gx, mean_gy and the corrected transcribed caption
   
_, et_events = avs_prep.avs_combine_events(subjects=subjects, sessions=sessions, # TODO: set session to 10
                                               data_path=results_dir,
                                               preprocessed=True, fix_multi_saccades=False)
# add the fixation sequence number
et_events = avs_prep.add_fixation_sequence_position(et_events, add_saccade_sequence=False, verbose=False)

print(et_events.columns)
# only keep fixation events
et_events = et_events[et_events['type'] == 'fixation']

et_events = et_events[et_events['recording'] == 'scene']
# remove last fixaation on a scene

et_events = et_events[et_events['fix_sequence_from_last'] != 0]
# only scene where caption_task == 1
et_events = et_events[et_events['caption_task'] == 1]

print(et_events.columns)
# only keep events during the scene 

for subject in subjects:
    for sess in sessions:
        print(f"subject {subject}, session {sess}")
        log_fname    =  f"explog_transcribed_corrected_{str(subject).zfill(2)}_{str(sess).zfill(2)}.csv"
        exp_log_fname = os.path.join(data_path, output_prefix + str(subject).zfill(2) + '_' + str(sess).zfill(2) , log_fname)
        explog_sub_sess = pd.read_csv(exp_log_fname)
        print(explog_sub_sess.columns)
        # get the sceneIDs for this subject and session from the explog (only if caption_task == 1)
        sceneIDs = np.unique(explog_sub_sess.loc[explog_sub_sess['caption_task'] == 1, 'scene_ID'].values)
        for s_id in sceneIDs:
            # get the transcribed caption for this sceneID from the explog
            transcribed_caption = explog_sub_sess.loc[explog_sub_sess['scene_ID'] == s_id, 'trans_corrected'].values[0]
            et_events.loc[(et_events['sceneID'] == s_id) & (et_events['subject'] == subject) & (et_events['session'] == sess), 'transcribed_caption'] = transcribed_caption
            # get the scene image for this sceneID from the normal_image_dir
            scene_image_fname = str(int(s_id)).zfill(12) + "_MEG_size.jpg"
            # check if the scene image exists in the output_scene_dir
            if not os.path.exists(os.path.join(output_scene_dir, scene_image_fname)):
                scene_image_fname = os.path.join(normal_image_dir, scene_image_fname)
                # copy the scene image to the output_scene_dir
                os.system(f"cp {scene_image_fname} {output_scene_dir}")


        
        
# store the dataframe as a csv file
print(et_events.head)
et_events.to_csv(os.path.join(output_dir, "fix2cap_events.csv"), index=False)

print("done")


        
            

            












