import pandas as pd
import os

def convert(date_time):
    s = pd.Series(pd.to_timedelta(date_time))
    datetime_str = s.dt.total_seconds() * 1e3
    return datetime_str

def onset_offset_manual_lookit(manual_direc, manual_annotated_file, lookit_file):
    """
    params:
    manual_annotated_file (str) - hand annotated file for one child
    lookit_file (str) - lookit file containing all children
    """
    # load both files
    lookit_df = pd.read_csv(lookit_file)
    manual_df = pd.read_csv(os.path.join(manual_direc, manual_annotated_file))

     # get child name - determines subset of lookit dataframe
    file_split = manual_annotated_file.replace('.csv','').split('_')
    child_name = file_split[1]


    # specify column names
    if file_split[-1] == 'AW':
        trial_type = 'Trials_x'
        trial_onset = 'Trials_onset'
        trial_offset = 'Trials_offset'
        trial_ordinal = "Trials_ordinal"

    elif file_split[-1] == 'GS':
        trial_type = 'Trials.x'
        trial_onset = 'Trials.onset'
        trial_offset = 'Trials.offset'
        trial_ordinal = "Trials.ordinal"
    
    manual_df = manual_df.loc[(manual_df[trial_type] == 'f') | (manual_df[trial_type] == 't')]

    if file_split[-1] == 'AW':
        manual_df = manual_df.drop_duplicates(subset=[trial_onset])
        manual_df = manual_df.assign(Index=range(len(manual_df))).set_index('Index')
   
    
    # change format of lookit timestamps into milliseconds
    lookit_df['relative_onset'] = convert(lookit_df['relative_onset'])
    lookit_df['relative_offset'] = convert(lookit_df['relative_offset'])

    # get length trial for lookit
    lookit_df = lookit_df.assign(trial_length = lookit_df['relative_offset'] - lookit_df['relative_onset'])

    # get length trial for hand annotated
    manual_df = manual_df.assign(trial_length = manual_df[trial_offset] - manual_df[trial_onset])

    # get lookit rows for only this child 
    child_df = lookit_df.loc[lookit_df['child_id'] == child_name]
    child_df = child_df.assign(Index=range(len(child_df))).set_index('Index')


    # add lookit info to finaldata
    manual_df = manual_df.assign(lookit_onset = child_df['relative_onset'])
    manual_df = manual_df.assign(lookit_offset = child_df['relative_offset'])
    manual_df = manual_df.assign(lookit_trial_length = child_df['trial_length'])

    # compute distance between trial length
    manual_df = manual_df.assign(lookit_trial_length_diff = child_df['trial_length'] - manual_df['trial_length'])
    
    # compute distance between onsets 
    manual_df = manual_df.assign(lookit_onset_diff = child_df['relative_onset'] - manual_df[trial_onset])


    # compute distance between offsets 
    manual_df = manual_df.assign(lookit_offset_diff = child_df['relative_offset'] - manual_df[trial_offset]) 
    
    manual_df = manual_df.assign(manual_trial_onset = manual_df[trial_onset])
    manual_df = manual_df.assign(manual_trial_offset = manual_df[trial_offset])
    manual_df = manual_df.assign(manual_trial_ordinal = manual_df[trial_ordinal])
    manual_df = manual_df[["manual_trial_onset","manual_trial_offset","manual_trial_ordinal",
                        "trial_length","lookit_onset","lookit_offset", 
                        "lookit_trial_length","lookit_trial_length_diff",
                        "lookit_onset_diff", "lookit_offset_diff"]]
    manual_df.insert(0,'child_name',child_name)
    manual_df.insert(1,'same_num_trials',len(child_df)==len(manual_df))
    manual_df.insert(2,'annotator',file_split[-1])

    

    
    
    all_children.append(manual_df)
   



if __name__ == '__main__':
    lookit_file = "lookit_trial_timing_info.csv"
    manual_directory = 'InputFiles'

    all_children = []
    for file in os.listdir(manual_directory):
        if file.endswith('.csv'):
            onset_offset_manual_lookit(manual_directory, file, lookit_file)

    all_df = pd.concat(all_children)

    all_df.to_csv("manual_lookit_comparison_both.csv")  