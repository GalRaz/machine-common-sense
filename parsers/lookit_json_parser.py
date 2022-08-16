from datetime import datetime
import pandas as pd
from string import digits
import json
import numpy as np

PARENT_CONTROL = True

def get_lookit_trial_times():

    f = open('lookit_info/lookit_info_sessionB.json')
    exp_info = json.load(f)

    # dataframe in which info will be accumulated
    trial_timing_info = pd.DataFrame()

    for session_info in exp_info:

        # get key of recording start
        recording_start_key = [v for v in session_info['exp_data'].keys() if v.endswith('start-recording-with-image')]

        # only continue if this part of the dictionary has a 'start-recording-with-image' frame
        if recording_start_key:
            child_id = session_info['child']['hashed_id']

            # timestamp of start of recording
            video_onset = datetime.fromisoformat(session_info['exp_data'][recording_start_key[0]]['eventTimings'][5]['timestamp'][0:-1])

            # identify index of the trials we want: the first videoStarted, and last videoPaused
            for key, value in session_info['exp_data'].items():

                # only consider fam and test trials, no attention getters (and no prematurely terminated trials)
                if ('fam' in key or 'test' in key) and ('attention' not in key) and (len(value['eventTimings']) > 2):
                    print(key)

                    eventTypes = [timestamp_type['eventType'] for timestamp_type in value['eventTimings']]

                    # get locations of videoStarted and videoPaused
                    videoStartedIdx = ['videoStarted' in event for event in eventTypes]
                    videoPausedIdx = ['videoPaused' in event for event in eventTypes]
                    
                    parentEnded = False;

                    if PARENT_CONTROL:
                        # check whether parent termination can be found in the video
                        trialEndReason = value['trialEndReason']
                        if trialEndReason == "parentEnded":
                            parentEnded = True;


                    # find first place where 'videoStarted' appears
                    if any(videoStartedIdx):
                        video_start_idx = np.where(videoStartedIdx)[0][0]

                    # find last place where 'videoPaused' appears
                    if any(videoPausedIdx):
                        video_end_idx = np.where(videoPausedIdx)[0][-1]

                    if any(videoStartedIdx) and any(videoPausedIdx):
                        trial_timestamps = \
                        [{'child_id': child_id, 'video_onset': video_onset, 'trial_type': key, \
                                        'absolute_onset': datetime.fromisoformat(value['eventTimings'][video_start_idx]['timestamp'][0:-1]), \
                                        'absolute_offset': datetime.fromisoformat(value['eventTimings'][video_end_idx]['timestamp'][0:-1]),
                                        'trial_type_onset': eventTypes[video_start_idx],
                                        'trial_type_offset': eventTypes[video_end_idx], 
                                        'parentEnded': parentEnded}]

                        trial_timing_info = trial_timing_info.append(pd.DataFrame(trial_timestamps))

    # get trial onset/offset relative to onset of video recording
    trial_timing_info['relative_onset'] = trial_timing_info['absolute_onset'] - trial_timing_info['video_onset']
    trial_timing_info['relative_onset'] = trial_timing_info['relative_onset'].apply(lambda x: x.total_seconds() * 1000)

    trial_timing_info['relative_offset'] = trial_timing_info['absolute_offset'] - trial_timing_info['video_onset']
    trial_timing_info['relative_offset'] = trial_timing_info['relative_offset'].apply(lambda x: x.total_seconds() * 1000)

    # clean up trial type to be ready for parsing
    trial_timing_info['trial_type'] = trial_timing_info['trial_type'].str.replace('\d+-', '')

    # parse trial type into fam vs. test and scene
    trial_timing_info[['fam_or_test', 'scene']] = trial_timing_info['trial_type'].str.split('-', 1, expand = True)

    # sort whole df by trial onset
    trial_timing_info.sort_values(by='absolute_onset', inplace=True)

    # get trial number using absolute onsets
    trial_timing_info['trial_number'] = trial_timing_info.groupby('child_id').cumcount()+1

    return trial_timing_info

trial_timing_info = get_lookit_trial_times()
trial_timing_info.to_csv('lookit_info/lookit_trial_timing_info_sessionB.csv')

## ADD: whether or not parent ended trial
