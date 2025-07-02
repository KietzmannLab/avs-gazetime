import os
from pathlib import Path
import avs_machine_room.dataloader.tools.avs_directory_tools as avs_directory
import numpy as np
import pandas as pd
from avs_gazetime.utils.sensors_mapping import grads, mags

def configure_run(subject_id=None, ch_type="mag", sensor_selection="fixation", debug=False, hemi='lh', sensor2source=True):
    if subject_id is not None:
        subject_id = subject_id
    else:
        subject_id = 4
    
    subject = f"as{subject_id:02d}"
    sessions = np.arange(1, 10+1)  # sessions to include
    if debug: 
        sessions = np.arange(1, 2)
    s_freq = 500  # sampling frequency of MEG data in Hz
    if ch_type is None:
        ch_type = "grad"
    else:
        ch_type = ch_type
    
    if ch_type == "stc":
        meg_dir = Path(f"/share/klab/datasets/avs/population_codes/{subject}/source_space/beamformer/glasser/ori_normal/hem_both/filter_0.2_200/ica/")
        sensor_or_source = "source"
        out_key = "stc"
        
    elif ch_type not in ["grad", "mag"]:
        print(f"we run in source space: {ch_type}")
        meg_dir = Path(f"/share/klab/datasets/avs/population_codes/{subject}/source_space/beamformer/glasser/ori_normal/hem_{hemi}/filter_0.2_200/ica/")
        sensor_or_source = "source"
        out_key = "source"
    else:
        meg_dir = Path(f"/share/klab/datasets/avs/population_codes/{subject}/sensor/erf/filter_0.2_200/ica/")
        sensor_or_source = "sensor"
        out_key = "erf"
    meg_processed_dir = meg_dir
    _, et_dir, project_dir = avs_directory.get_data_dirs(server="uos", add_project_dir=True)
    print(f"project_dir: {project_dir}")
    plots_dir_behav = os.path.join(os.sep+"share", "klab", "psulewski", "psulewski", "active-visual-semantics-MEG", "results", "fullrun",
                             'analysis', 'gazetime',"submission_checks","behav")
    plots_dir_no_sub = os.path.join(os.sep+"share", "klab", "psulewski", "psulewski", "active-visual-semantics-MEG", "results", "fullrun",
                             'analysis', 'gazetime',"submission_checks","ica",out_key,"filter_0.2_200")
    plots_dir = os.path.join(plots_dir_no_sub, subject)
    
    subjects_dir = os.path.join(os.sep+"share", "klab", "datasets", "avs", "rawdir")

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)


    return {
        "SUBJECT_ID": subject_id,
        "SUBJECT": subject,
        "SESSIONS": sessions,
        "S_FREQ": s_freq,
        "CH_TYPE": ch_type,
        "MEG_DIR": meg_dir,
        "MEG_PROCESSED_DIR": meg_processed_dir,
        "ET_DIR": et_dir,
        "PROJECT_DIR": project_dir,
        "PLOTS_DIR_NO_SUB": plots_dir_no_sub,
        "PLOTS_DIR": plots_dir,
        "SENSOR_SELECTION": sensor_selection,
        "DEBUG": debug,
        "SENSOR_OR_SOURCE": sensor_or_source,
        "SENSOR2SOURCE": sensor2source,
        "PLOTS_DIR_BEHAV": plots_dir_behav,
        "SUBJECTS_DIR": subjects_dir
    }
    
# try to get the subject id from the jobscript

try:
    subject_id = int(os.environ["SUBJECT_ID_GAZETIME"])
    print(f"subject_id: {subject_id}")
    debug = False
except KeyError:
    subject_id = 5
    print("subject_id not found in environment variables")
    # set sebug to True
    debug = False
try:
    ch_type = os.environ["CH_TYPE_GAZETIME"]
    print(f"ch_type: {ch_type}")
except KeyError:
    ch_type = "stc" # TODO: make None again
    print("ch_type not found in environment variables")

try:
    sensor_selection = os.environ["SENSOR_SELECTION_GAZETIME"]
    print(f"sensor_selection: {sensor_selection}")
except KeyError:
    sensor_selection = "fixation"
    print("sensor_selection not found in environment variables")
    

    

config = configure_run(subject_id=subject_id, ch_type=ch_type, sensor_selection=sensor_selection, debug=debug)
# extract the vars from the dict so that they can be accessed directly by importing this module
SUBJECT_ID = config["SUBJECT_ID"]
SUBJECT= config["SUBJECT"]
SESSIONS = config["SESSIONS"]
S_FREQ = config["S_FREQ"]
CH_TYPE = config["CH_TYPE"]
MEG_DIR = config["MEG_DIR"]
MEG_PROCESSED_DIR = config["MEG_PROCESSED_DIR"]
ET_DIR = config["ET_DIR"]
PROJECT_DIR = config["PROJECT_DIR"]
PLOTS_DIR_NO_SUB = config["PLOTS_DIR_NO_SUB"]
PLOTS_DIR = config["PLOTS_DIR"]
PLOTS_DIR_BEHAV = config["PLOTS_DIR_BEHAV"]
SENSOR_SELECTION = config["SENSOR_SELECTION"]
DEBUG = config["DEBUG"]
SENSOR_OR_SOURCE = config["SENSOR_OR_SOURCE"]
SENSOR2SOURCE = config["SENSOR2SOURCE"]
SUBJECTS_DIR = config["SUBJECTS_DIR"]


#print all the variables
print(f"SUBJECT_ID: {SUBJECT_ID}")
print(f"SUBJECT: {SUBJECT}")
print(f"SESSIONS: {SESSIONS}")
print(f"S_FREQ: {S_FREQ}")
print(f"CH_TYPE: {CH_TYPE}")
print(f"MEG_DIR: {MEG_DIR}")
print(f"MEG_PROCESSED_DIR: {MEG_PROCESSED_DIR}")
print(f"ET_DIR: {ET_DIR}")
print(f"PROJECT_DIR: {PROJECT_DIR}")
print(f"PLOTS_DIR_NO_SUB: {PLOTS_DIR_NO_SUB}")
print(f"PLOTS_DIR: {PLOTS_DIR}")
print(f"SENSOR_SELECTION: {SENSOR_SELECTION}")
print(f"DEBUG: {DEBUG}")
print(f"SENSOR_OR_SOURCE: {SENSOR_OR_SOURCE}")

