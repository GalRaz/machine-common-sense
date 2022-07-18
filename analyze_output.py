import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from scipy.stats import pearsonr

from Scripts.video import get_frame_information

# global directory path variables. make these your folder names under MCS
ICATCHER_DIR = '/nese/mit/group/saxelab/users/galraz/icatcher_tests/icatcher_plus/icatcher_output/mcs'

# trial info
TRIAL_INFO_DIR = 'lookit_info/lookit_trial_timing_info_sessionA.csv'

# directory for videos
VID_DIR = '/nese/mit/group/saxelab/users/galraz/mcs/videos/exp1/sessionA'

# add absolute path to iCatcher repo
ICATCHER = '/Users/gracesong/dev/iCatcher'
# sys.path.append(ICATCHER)

LOOKAWAY_CRITERION = False

# time until lookaway criterion starts
lookaway_onset_tolerance = 5;
lookaway_criterion_duration = 3;

# amount of time to check before the end of the trial for off looks
time_before_end_check_off = 4000

###################
## HELPER FUNCTIONS ##
####################
def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

###################
## ANALYSIS SCRIPT ##
####################
def run_analyze_output(data_filename="AGENT_sessionA_output_noLA.csv", session=None):
    """
    Given an iCatcher output directory and Datavyu input and output 
    files, runs iCatcher over all videos in vid_dir that have not been
    already run, computes looking times for all iCatcher outputs, and
    compares with Datavyu looking times. 
    data_filename (string): name of file you want comparison data to be written
            to. Must have .csv ending. 
    session (string): ID of the experiment session. If session is not
            specified, looks for videos only within VID_DIR, otherwise
            searches within [VID_DIR]/session[session]
    """
    for filename in listdir_nohidden(ICATCHER_DIR):
        child_id = filename.split('.')[0]

        print(filename)
        
        # skip if child data already added
        output_file = Path(data_filename)
        if output_file.is_file():
            output_df = pd.read_csv(data_filename, index_col=0)
            ids = output_df['child'].unique()
            if child_id in ids: 
                print(child_id + ' already processed')
                continue
        
        vid_path = VID_DIR + '/'
        if session:
            vid_path += "session" + session + '/'
        vid_path = vid_path + child_id + ".mp4"

        # get timestamp for each frame in the video
        print('getting frame information for {}...'.format(vid_path))
        timestamps, length = get_frame_information(vid_path)
        if not timestamps:
            print('video not found for {} in {} folder'.format(child_id, VID_DIR))
            continue
        
        # initialize df with time stamps for iCatcher file
        icatcher_path = ICATCHER_DIR + '/' + filename
        icatcher = read_convert_output(icatcher_path, timestamps)

        # get trial onsets and offsets from input file, match to iCatcher file
        trial_sets, df = get_trial_sets(child_id)
        assign_trial(icatcher, trial_sets)
        # sum on looks and off looks for each trial
        icatcher_times = get_on_off_times(icatcher)
        # datavyu_times = get_output_times(output_file)

        # check whether number of trials from trial info is the same as 
        if icatcher['trial'].max() != len(df):
            print('mismatch in # of trials between icatcher and session info: {} in {} folder'.format(child_id, VID_DIR))
            continue

        write_to_csv(data_filename, child_id, icatcher_times, session, df['fam_or_test'], df['scene'], df['parentEnded'], icatcher)


#####################
## HELPER FUNCTIONS ##
#####################

def read_convert_output(filename, stamps):
    """
    Given a tabular data file containing columns for frames and looks,
    converts to pandas DataFrame with another column mapping each frame
    to its time stamp in the video
    
    filename (string): name of tabulated iCatcher output file in format
    '[CHILD_ID]_annotation.txt'
    stamps (List[int]): time stamp for each frame, where stamps[i] is the 
    time stamp at frame i
    rtype: DataFrame
    """

    npz = np.load(filename)
    df = pd.DataFrame([])

    lst = npz.files

    # looking coding
    print('looking coding')
    print(npz[lst[0]])

    df['frame'] = range(1, len(npz[lst[0]]) + 1)
    df['on_off'] = ['on' if frame > 0 else 'off' for frame in npz[lst[0]]]
    df['confidence'] = npz[lst[1]]

    # convert frames to ms using frame rate
    df['time_ms'] = stamps
    df['time_ms'] = df['time_ms'].astype(int)
    
    return df


def get_trial_sets(child_id):
    """
    Finds corresponding Datavyu input file for given iCatcher output file
    and returns a list of [onset, offset] times for each trial in 
    milliseconds
    
    input_file (string): name of Datavyu input file
    rtype: List[List[int]]
    """
    df = pd.read_csv(TRIAL_INFO_DIR)

    # get part of df from current child
    df = df[df['child_id'] == child_id] 

    # there's two different file formats -- updated as needed 
    df_sets = df[['relative_onset', 'relative_offset', 'parentEnded']]
    df_sets = df_sets.rename(columns={"relative_onset": "onset", "relative_offset": "offset"})
    
    df_sets.dropna(inplace=True)

    trial_sets = []
    for _, trial in df_sets.iterrows():
        trial_sets.append([int(trial['onset']), int(trial['offset'])])

    def unique(sequence):
        seen = set()
        return [x for x in sequence if not (tuple(x) in seen or seen.add(tuple(x)))]

    return unique(trial_sets), df


def assign_trial(df, trial_sets):
    """
    Given trial onsets and offsets, makes a 'trial' column in df mapping indicating
    which trial each frame belongs in, or 0 if no trial
    
    df (DataFrame): pandas Dataframe with time information
    trial_sets (List[List[int]]): list of trial [onset, offset] pairs in ms
    rtype: None
    """
    
    # mapping function
    def map_to_range(value, ranges):
        """
        Modifies df to have a column mapping value to one of the ranges provided, or 0 if not 
        """
        for start, end in ranges:
            if value in range(start, end + 1): 
                return ranges.index([start, end]) + 1
        return 0
    # rewrite this with logicals
    df['trial'] = df['time_ms'].apply(lambda x: map_to_range(x, trial_sets))


def get_on_off_times(df):
    """
    Calculates the total on and off look times per trial and returns a list of 
    [on time, off time] pairs for each trial in seconds
    
    df (DataFrame): DataFrame containing trial information per frame
    stamps (List[int]): time stamp for each frame, where stamps[i] is the 
    time stamp at frame i
    rtype: List[List[float]]
    """
    n_trials = df['trial'].max()
    looking_times = [[0, 0, 0] for trial in range(n_trials)]
    
    # separate times by trial
    trial_groups = df.groupby(['trial'])

    print(df)

    # iterate through trials
    for trial_num, group in trial_groups:
        # 0 means does not belong in a trial
        if trial_num == 0:
            continue

        last_look, start_time = None, None

        # initialize "last_time" as first timestamp in trial
        last_time = group['time_ms'].iloc[0]

        # get index of frame x sec before end to investigate lookaway before before the end of trial
        trial_end_period = group['time_ms'].iloc[-1] - time_before_end_check_off

        # iterate through looks
        for index, row in group.iterrows():
            time, look = row['time_ms'], row['on_off']

            # start of on or off look
            if not(last_look and start_time):
                last_look, start_time = look, time
                look_time = 0

            # count time spent looking off in last 4 seconds of the trial

            ## DEBUG: check that the 4 sec cutoff works
            if time > trial_end_period:
                ind = ['on', 'off'].index(look)
        
                if ind == 1: 
                    looking_times[trial_num - 1][2] += time - last_time

            if look == last_look:
                look_time = (time - start_time) / 1000

                # look away criterion: if criterion is met and it's more than x sec into the trial
                if LOOKAWAY_CRITERION and look == 'off' and look_time > lookaway_criterion_duration and \
                (time - group.iloc[1,:]['time_ms']) > lookaway_onset_tolerance*1000:

                    break

            # end of a look or end of trial
            else:
                ind = ['on', 'off'].index(last_look)
                looking_times[trial_num - 1][ind] += look_time

                # reset values
                last_look, start_time = None, None

            # save time from previous frame
            last_time = time

        # special case where entire trial is one look
        if last_look and start_time:
                ind = ['on', 'off'].index(last_look)
                looking_times[trial_num - 1][ind] += look_time 

    looking_times = [[round(on, 3), round(off, 3), round(time_off_before_end/1000, 3)] for on, off, time_off_before_end in looking_times]
    
    return looking_times

def convert(date_time):
    s = pd.Series(pd.to_timedelta(date_time))
    datetime_str = s.dt.total_seconds() * 1e3
    return datetime_str


def write_to_csv(data_filename, child_id, icatcher_data, session, trial_type, stim_type, parent_ended, icatcher):
    """
    checks if output file is in directory. if not, writes new file
    containing looking times computed by iCatcher and Datavyu for child
    with Lookit ID id. 
    
    child_id (string): unique child ID associated with subject
    icatcher_data (List[List[int]]): list of [on times, off times] per trial
                calculated form iCatcher
    datavyu_data (List[List[int]]): list of [on times, off times] per trial
                calculated form iCatcher
    session (string): the experiment session the participant was placed in
    rtype: None
    """
    trial_num = [i + 1 for i in range(len(icatcher_data))]
    id_arr = [child_id] * len(icatcher_data)
    
    # calculate confidence
    confidence = icatcher[(icatcher['on_off'] == 'on') & (icatcher['trial'] != 0)].groupby('trial')[['confidence']].mean()
    
    # check whether a trial is missing, which can happen if there are no on frames in a trial
    missing_trials = list(set(trial_num) ^ set(confidence.index))
    
    for trial in missing_trials:
        confidence.at[trial] = 0 
    
    data = {
        'child': id_arr,
        'session': [session] * len(trial_num),
        'trial_num': trial_num,
        'trial_type': trial_type,
        'stim_type': stim_type,
        'confidence': list(confidence.squeeze()),
        'iCatcher_on(s)': [trial[0] for trial in icatcher_data],
        'iCatcher_off(s)': [trial[1] for trial in icatcher_data],
        't_spent_off_at_trial_end': [trial[2] for trial in icatcher_data],
        'parentEnded': parent_ended
    }

    # if lookaway criterion is set, t_spent_off_at_trial_end is not calculated so remove here
    if LOOKAWAY_CRITERION:
        data.pop('t_spent_off_at_trial_end', None)

    df = pd.DataFrame(data)

    output_file = Path(data_filename)
    if not output_file.is_file():
        df.to_csv(data_filename)
        return
    
    output_df = pd.read_csv(data_filename, index_col=0)
    ids = output_df['child'].unique()

    if child_id not in ids:
        output_df = output_df.append(df, ignore_index=True)
        output_df.to_csv(data_filename)
    

if __name__ == "__main__":
    run_analyze_output()