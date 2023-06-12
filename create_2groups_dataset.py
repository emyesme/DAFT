# This is a sample Python script.

# libraries
from collections import Counter
import nibabel as nib
import pandas as pd
import numpy as np
import datetime
import random
import shutil
import h5py
import os
import gc

# directory
DIR_1 = os.path.join("path_to_dataset_part1")
DIR_2 = os.path.join("path_to_dataset_part2")


# take tps: time points
def take_tps(path, tp):
    path_flair = os.path.join(path, tp, 'flair_mni.nii.gz')
    path_lesions = os.path.join(path, tp, 'location_labeled_lesions.nii.gz')
    path_t1 = os.path.join(path, tp, 't1_mni.nii.gz')
    path_brain_mask = os.path.join(path, tp, 'brainmask_mni.nii.gz')

    if not (os.path.isfile(path_flair) and
            os.path.isfile(path_lesions) and
            os.path.isfile(path_t1) and
            os.path.isfile(path_brain_mask)):
        raise Exception("File does not exist take_tps")

    return path_flair, path_t1, path_lesions, path_brain_mask


def write_hdf5(stage, df, info):
    # creating a HDF5 file, w for write access
    # hf object with a bunch of associated methods
    hf = h5py.File(str(datetime.datetime.now()) + '_' + stage + '.h5', 'w')
    # for images in the training set
    labels = []
    acum_seq1 = []
    acum_seq2 = []
    count_seq1 = 0
    count_seq2 = 0
    i = 0
    for (_, row), (_, info) in zip(df.iterrows(), info.iterrows()):

        # extract info
        row = pd.to_numeric(row)
        groupMS = info['group']
        label = info['2group']
        name = info['patient_name']
        study_date = info['study_date']

        # create general group
        group = hf.create_group(str(i))

        # create group
        data = hf.create_group(str(i) + '/Left-Hippocampus')


        flair, t1, lesion_mask, brain_mask = take_tps(os.path.join(DIR_2,
                                                                   'datasetname_' + groupMS,
                                                                   name),
                                                      study_date)

        # putting images as different channel for input
        flair = nib.load(flair).get_fdata()
        t1 = nib.load(t1).get_fdata()
        lesion_mask = nib.load(lesion_mask).get_fdata()
        brain_mask = nib.load(brain_mask).get_fdata()

        # this label mask have several labels as to represent
        # different types of lesions depending on different factors
        # binarize this so the cnn does not take this as weights for
        # those roi of the image
        b_lmask = lesion_mask > 0
        b_lmask = b_lmask.astype(int)

        # take out non brain stuff
        t1_masked = t1[brain_mask > 0]
        flair_masked = flair[brain_mask > 0]

        # indexes of voxels that are part of the brain
        mask = np.where(brain_mask != 0)
        t1 = np.zeros(lesion_mask.shape)
        flair = np.zeros(lesion_mask.shape)

        count = 0
        for m, n, k in zip(mask[0], mask[1], mask[2]):
            # reconstruction of volume
            t1[m, n, k] = t1_masked[count] + 1
            flair[m, n, k] = flair_masked[count] + 1
            count += 1

        # stacking the mask + flair + t1
        input_images = np.concatenate((b_lmask[np.newaxis, ...], flair[np.newaxis, ...], t1[np.newaxis, ...]), axis=0)

        counting_seq1 = Counter(np.array(flair[brain_mask > 0], dtype=np.int32))
        counting_seq2 = Counter(np.array(t1[brain_mask > 0], dtype=np.int32))

        if len(acum_seq1) == 0:
            acum_seq1 = counting_seq1
            count_seq1 = len(flair[brain_mask > 0])
        else:
            acum_seq1 = np.add(acum_seq1, counting_seq1)
            count_seq1 += len(flair[brain_mask > 0])
        #
        if len(acum_seq2) == 0:
            acum_seq2 = counting_seq2
            count_seq2 = len(t1[brain_mask > 0])
        else:
            acum_seq2 = np.add(acum_seq2, counting_seq2)
            count_seq2 += len(t1[brain_mask > 0])

        # clean images because of space needs
        del t1, flair, lesion_mask, brain_mask, b_lmask

        data.create_dataset('vol_with_bg', dtype='f', compression='gzip',
                            data=input_images)

        # create dataset tabular
        group.create_dataset('tabular', data=row[:])

        # cleaning images and tabular because of space needs
        del input_images, row

        # create attributes
        # RID scalar - image ID - unique identifier
        group.attrs["RID"] = name

        # visit code - string
        group.attrs["VISCODE"] = study_date

        # labels #

        # DX - diagnosis - CIS, PP, SP, RR
        group.attrs["DX"] = label

        # CN - time-to-event conversion was observed yes or no and a scalar attribute time
        # group["CN"] = 'yes'
        # group["time"] = '01-01-1999'
        i += 1

        gc.collect()
    # end for
    gc.collect()
    # stats #

    # initial stats group
    stats = hf.create_group('stats')

    # tabular stats group
    tabular = stats.create_group("tabular")

    # columns names
    # columns that are strings have to be encoded this way to be accepted by hdf5
    dt = h5py.string_dtype(encoding='utf-8')

    # TODO: add shape manually shape
    name_columns = np.array(df.columns)
    tabular.create_dataset('columns', shape=name_columns.shape, dtype=dt, data=name_columns)
    # means
    means = df.mean(axis=0)
    tabular.create_dataset('mean', shape=means.shape, dtype='f', data=means)
    # stddevs
    stddevs = df.std(axis=0)
    tabular.create_dataset('stddev', shape=stddevs.shape, dtype='f', data=stddevs)

    # image stats group
    roi = stats.create_group("Left-Hippocampus")

    dataset_name = roi.create_group("vol_with_bg")

    # mean and std
    sum_seq1 = [i * value for i, value in enumerate(acum_seq1)]
    sum_seq2 = [i * value for i, value in enumerate(acum_seq2)]

    mean_values = np.array([np.sum(sum_seq1) / count_seq1, np.sum(sum_seq2) / count_seq2])
    # print("mean.shape ", mean_values)
    dataset_name.create_dataset('mean', shape=mean_values.shape, dtype='f', data=mean_values)

    # flair, t1
    numerator_seq1 = [value * (i - mean_values[0]) ** 2 for i, value in enumerate(acum_seq1)]
    numerator_seq2 = [value * (i - mean_values[1]) ** 2 for i, value in enumerate(acum_seq2)]

    std_values = np.array(
        [np.sqrt(np.sum(numerator_seq1) / (count_seq1 - 1)), np.sqrt(np.sum(numerator_seq2) / (count_seq2 - 1))])
    # print("std.shape ", std_values)
    dataset_name.create_dataset('stddev', shape=std_values.shape, dtype='f', data=std_values)

    # close the file
    hf.close()


# filenames
def write_hdf5_fold_part4(stage, h5_part1, h5_part2, h5_part3, h5_part4):
    # creating a HDF5 file, w for write access
    # hf object with a bunch of associated methods

    shutil.copyfile(h5_part4, str(datetime.datetime.now()) + '_test_' + stage + '.h5')
    shutil.copyfile(h5_part3, str(datetime.datetime.now()) + '_val_' + stage + '.h5')

    # fold1 #
    i = 0
    # train
    with h5py.File(str(datetime.datetime.now()) + '_train_' + stage + '.h5', "w") as hf_train:
        # part 1
        with h5py.File(h5_part1, "r") as hf_part1:
            for image_uid, g in hf_part1.items():
                if image_uid == "stats":
                    part1_mean = g['Left-Hippocampus']['vol_with_bg']['mean'][()]
                    part1_std = g['Left-Hippocampus']['vol_with_bg']['stddev'][()]
                    part1_tabular_columns = g['tabular']['columns'][()]
                    part1_tabular_mean = g['tabular']['mean'][()]
                    part1_tabular_std = g['tabular']['stddev'][()]
                else:
                    hf_part1.copy(str(image_uid), hf_train, name=str(i))
                    i = i + 1

        # part 2
        with h5py.File(h5_part2, "r") as hf_part2:
            for image_uid, g in hf_part2.items():
                if image_uid == "stats":
                    part2_mean = g['Left-Hippocampus']['vol_with_bg']['mean'][()]
                    part2_std = g['Left-Hippocampus']['vol_with_bg']['stddev'][()]
                    part2_tabular_mean = g['tabular']['mean'][()]
                    part2_tabular_std = g['tabular']['stddev'][()]
                else:
                    hf_part2.copy(str(image_uid), hf_train, name=str(i))
                    i = i + 1

        # stats #

        # initial stats group
        stats = hf_train.create_group('stats')

        # tabular stats group
        tabular = stats.create_group("tabular")

        # columns names
        # columns that are strings have to be encoded this way to be accepted by hdf5
        dt = h5py.string_dtype(encoding='utf-8')

        name_columns = np.array(part1_tabular_columns)
        tabular.create_dataset('columns', shape=name_columns.shape, dtype=dt, data=name_columns)
        # means
        means = (part1_tabular_mean + part2_tabular_mean) / 2
        print("stats tabular means shape ", means.shape)
        tabular.create_dataset('mean', shape=means.shape, dtype='f', data=means)
        # stddevs
        stddevs = np.maximum(part1_tabular_std, part2_tabular_std)
        print("stats tabular stddevs shape ", stddevs.shape)
        tabular.create_dataset('stddev', shape=stddevs.shape, dtype='f', data=stddevs)

        # image stats group
        roi = stats.create_group("Left-Hippocampus")

        dataset_name = roi.create_group("vol_with_bg")

        mean_values = (part1_mean + part2_mean) / 2
        dataset_name.create_dataset('mean', shape=mean_values.shape, dtype='f', data=mean_values)

        std_values = np.maximum(part1_std, part2_std)
        dataset_name.create_dataset('stddev', shape=std_values.shape, dtype='f', data=std_values)

    hf_train.close()


def cleaning_table(data_df_1, data_df_2):
    # drop unnecessary information from data_df_2

    # first sequence related variables of t1 and flair
    data_df_2 = data_df_2.loc[:, ~data_df_2.columns.str.contains('T1_', case=True)]
    data_df_2 = data_df_2.loc[:, ~data_df_2.columns.str.contains('FLAIR_', case=True)]

    # second specific unnecessary columns picked by hand
    data_df_2.drop(columns=['auto_qc_remarks',
                            'auto_qc_consequences',
                            'auto_qc_status',
                            'Gad_enhancing_lesion_volume',
                            'time',
                            'scanner',
                            'T1w_contrast',
                            'study_uri',
                            'study_id',
                            'pipeline',
                            'report_type',
                            'patient_uri',
                            'project_id',
                            'patient_id',
                            'patient_sex'],
                   inplace=True)

    # drop categorical columns I do not need
    data_df_1.drop(columns=['patient_birth_date', 'MS_onset', 'MS_group', 'Relabeled exam'], inplace=True)

    # we have to join the tables properly by:
    # - patient name
    # - study date

    df = pd.merge(data_df_1,
                  data_df_2,
                  how='inner',
                  left_on=['patient_name', 'study_date'],
                  right_on=['patient_name', 'date'],
                  validate='one_to_one')

    # drop duplicated columns by values - transpose is expensive
    df = df.T.drop_duplicates().T

    # drop columns with all nans
    df = df.dropna(axis=1, how='all')

    # 519 rows remain
    # duplicate values of the followup patient that are empty in the baseline of the patient

    patients_names = df["patient_name"].to_numpy()
    # delete duplicates
    patients_names = list(dict.fromkeys(patients_names))
    to_drop = []
    for patient_name in patients_names:
        tps_idxs = df.index[df['patient_name'].str.contains(patient_name)].tolist()
        # delete other tp for this patients, tp different of baseline and followup
        # drop tps
        to_drop.extend([tps_idxs[0]])
        to_drop.extend(tps_idxs[2:])

    df.drop(to_drop, inplace=True)
    print("before drop nan ", df.isna().sum().sum())
    df.dropna(inplace=True)
    print("after drop nan ", df.isna().sum().sum())

    # do categorical to numerical with pivotal encoding
    # because when giving the mean and stddev will not have sense

    # grouping the cis-rr and sp-pp patients
    conditions = [
        (df["group"].str.contains("CIS")),
        (df["group"].str.contains("RR")),
        (df["group"].str.contains("SP")),
        (df["group"].str.contains("PP"))
    ]

    values = ["CISRR", "CISRR", "SPPP", "SPPP"]

    df["2group"] = np.select(conditions, values)

    # keep label in another variable
    info = df[['study_date', 'patient_name', 'group', '2group']].copy()
    df.drop(columns=['study_date', 'patient_name', 'group', '2group'], inplace=True)

    # specific for rochelyon old
    i = df[df['MSFC'].str.contains('Err')].index  # get index of the row that contain Err substring
    df.drop(i, inplace=True)
    info.drop(i, inplace=True)

    # change age and group to pivotal
    new_cols_sex = pd.get_dummies(df.patient_sex)
    df = pd.concat([df, new_cols_sex], axis='columns')
    df.drop(columns=['patient_sex'], axis='columns', inplace=True)

    # change type of the column
    df = df.apply(pd.to_numeric)
    # other ways to change the type of columns

    # standard scale data
    df.iloc[:, 0:-1] = df.iloc[:, 0:-1].apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)

    return df, info


def usable_images(df, info):
    # this can be done also with command line
    # get into each folder of RR, SP, PP and CIS and
    # du --summarize --human-readable * | grep 4.0K | wc -l
    # the elements of the folder not output of the previous command

    _, folders, _ = next(os.walk(os.path.join(DIR_2)))
    # should be datasetname CIS, SP, PP, RR
    data_names = []
    #  rejoining the columns to split.
    columns_names = info.columns.to_numpy().reshape(-1)
    columns_names = np.concatenate((columns_names, df.columns.to_numpy().reshape(-1)), axis=0)
    usable = pd.DataFrame(columns=columns_names)
    tabular_names = info['patient_name'].to_numpy()

    # CIS, SP, PP, RR
    for folder in folders:
        _, cases, _ = next(os.walk(os.path.join(DIR_2, folder)))
        # now in the patients for this MS group
        for case in cases:
            # if it has not been drop in the tabular data
            if case not in tabular_names:
                # go to next iteration do not save case
                continue
            # obtaining this way the cleaning list inconsistent to the folders
            _, tps_folder, _ = next(os.walk(os.path.join(DIR_2, folder, case)))
            # if there is no tps in that case
            if tps_folder:  # taking into account bool([]) == False -> True
                # tps in the dataframe
                tps_df = info[info['patient_name'].str.contains(case)]['study_date'].to_numpy()
                # use the first of the intersection
                tps = list(set(tps_folder).intersection(tps_df))
                tps.sort()
                # save the name for the split
                data_names.append(case)

                for tp in tps:
                    idx = info.index[info['patient_name'].str.contains(case) & info['study_date'].str.contains(tp)]

                    # save the valid info we want to use
                    usable.loc[len(usable.index)] = np.concatenate((info.loc[idx].to_numpy().reshape(-1),
                                                                    df.loc[idx].to_numpy().reshape(-1)),
                                                                   axis=0)

    return data_names, usable


def manualSplit(df_main):
    # create empty datasets
    df_parts = [pd.DataFrame(),  # part 1
                pd.DataFrame(),  # part 2
                pd.DataFrame(),  # part 3
                pd.DataFrame()]  # part 4

    # parts distribution
    # cisrr, sppp
    n_parts = [[10, 10],  # part 1
               [10, 11],  # part 2
               [10, 12],  # part 3
               [11, 12]]  # part 4

    # cis, rr, sp, pp
    ms_groups_idx = [df_main.index[df_main["2group"] == "CISRR"].tolist(),
                     df_main.index[df_main["2group"] == "SPPP"].tolist()]
    #
    # print("ms_groups_idx ", ms_groups_idx)
    for i, part in enumerate(n_parts):
        # print(" part  ", part)
        for num, ms_group in enumerate(ms_groups_idx):
            # take a sample form the array of idxs
            sample_indexes = random.sample(ms_group, part[num])
            # get the sample corresponding to the indexes
            sample = df_main.iloc[sample_indexes]
            ms_groups_idx[num] = [element for element in ms_groups_idx[num] if element not in set(sample_indexes)]

            df_parts[i] = pd.concat([df_parts[i], sample], axis='rows')

    # check
    for i, df_part in enumerate(df_parts):
        print("df part ", len(df_part.index[df_part["2group"] == "CISRR"]))
        print("df part ", len(df_part.index[df_part["2group"] == "SPPP"]))

    return df_parts[0], df_parts[1], df_parts[2], df_parts[3]


def read_hf(hf):
    contando = [0, 0]
    for key, value in hf.items():
        if "stats" in key:
            print("with stats key ", value.keys())
            break
        if "CISRR" in value.attrs["DX"]:
            contando[0] += 1
        elif "SPPP" in value.attrs["DX"]:
            contando[2] += 1
        else:
            break

    print(" CISRR, SPPP")
    print(contando)


def fileSplit(df_main, file_part1, file_part2, file_part3, file_part4):
    names_part1 = pd.read_csv(file_part1)["patient_name"].values.tolist()
    names_part2 = pd.read_csv(file_part2)["patient_name"].values.tolist()
    names_part3 = pd.read_csv(file_part3)["patient_name"].values.tolist()
    names_part4 = pd.read_csv(file_part4)["patient_name"].values.tolist()

    df_part1, df_part2, df_part3, df_part4 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for index, register in df_main.iterrows():
        if register["patient_name"] in names_part1:
            df_part1 = df_part1.append(register, ignore_index=True)
        elif register["patient_name"] in names_part2:
            df_part2 = df_part2.append(register, ignore_index=True)
        elif register["patient_name"] in names_part3:
            df_part3 = df_part3.append(register, ignore_index=True)
        elif register["patient_name"] in names_part4:
            df_part4 = df_part4.append(register, ignore_index=True)

    return df_part1, df_part2, df_part3, df_part4


def data_to_hdf5_part4(path1, path2, file_part1, file_part2, file_part3, file_part4):
    # data
    path_tabular_1 = os.path.join(path1)
    path_tabular_2 = os.path.join(path2)

    data_df_1 = pd.read_csv(path_tabular_1)
    data_df_2 = pd.read_csv(path_tabular_2)

    # clean the tabular info and split one fold train 0.7 and test 0.3
    clean_df, info = cleaning_table(data_df_1, data_df_2)
    # splitting
    # get valid patient names from images availability
    data_names, data = usable_images(clean_df, info)

    df_part1, df_part2, df_part3, df_part4 = fileSplit(data, file_part1, file_part2, file_part3, file_part4)

    part1_info = df_part1[["study_date", "patient_name", "2group", "group"]]
    df_part1.drop(columns=["study_date", "patient_name", "2group", "group"], inplace=True)

    part2_info = df_part2[["study_date", "group", "patient_name", "2group"]]
    df_part2.drop(columns=["study_date", "group", "patient_name", "2group"], inplace=True)

    part3_info = df_part3[["study_date", "patient_name", "2group", "group"]]
    df_part3.drop(columns=["study_date", "patient_name", "2group", "group"], inplace=True)

    part4_info = df_part4[["study_date", "patient_name", "2group", "group"]]
    df_part4.drop(columns=["study_date", "patient_name", "2group", "group"], inplace=True)

    print('data_part1.shape ', df_part1.shape)
    write_hdf5("part1", df_part1, part1_info)
    print('data_part2.shape ', df_part2.shape)
    write_hdf5("part2", df_part2, part2_info)
    print('data_part3.shape ', df_part3.shape)
    write_hdf5("part3", df_part3, part3_info)
    print('data_part4.shape ', df_part4.shape)
    write_hdf5("part4", df_part4, part4_info)


def read_h5(filename):
    hf = h5py.File(filename, 'r')
    contando = [0, 0, 0, 0]
    names = np.array([])
    for key, value in hf.items():
        if "stats" in key:
            print("with stats key ", value.keys())
            break
        if "CISRR" in value.attrs["DX"]:
            contando[0] += 1
            names = np.append(names, [value.attrs["RID"]])
        elif "SPPP" in value.attrs["DX"]:
            contando[2] += 1
            names = np.append(names, [value.attrs["RID"]])
        else:
            break

    print(" CISRR, SPPP")
    print(contando)
    return names


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    data_to_hdf5_part4(os.path.join(DIR_1, "dataframe_1.csv"),
                       os.path.join(DIR_2, "dataframe_2.csv"),
                       '/home/ecarvajal/test/manual_part1.csv',
                       '/home/ecarvajal/test/manual_part2.csv',
                       '/home/ecarvajal/test/manual_part3.csv',
                       '/home/ecarvajal/test/manual_part4.csv')


    write_hdf5_fold_part4("fold1",
                          '/home/ecarvajal/data/part1.h5',
                          '/home/ecarvajal/data/part2.h5',
                          '/home/ecarvajal/data/part3.h5',
                          '/home/ecarvajal/data/part4.h5')

    write_hdf5_fold_part4("fold2",
                          '/home/ecarvajal/data/part1.h5',
                          '/home/ecarvajal/data/part3.h5',
                          '/home/ecarvajal/data/part2.h5',
                          '/home/ecarvajal/data/part4.h5')
    
    write_hdf5_fold_part4("fold3",
                          '/home/ecarvajal/data/part2.h5',
                          '/home/ecarvajal/data/part3.h5',
                          '/home/ecarvajal/data/part1.h5',
                          '/home/ecarvajal/data/part4.h5')
