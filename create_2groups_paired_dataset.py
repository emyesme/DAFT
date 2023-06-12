# This is a sample Python script.
from collections import Counter
import nibabel as nib
import pandas as pd
import numpy as np
import datetime
import h5py
import os
import gc

# directory
DIR_1 = os.path.join("path_to_dataset_part1")
DIR_2 = os.path.join("path_to_datset_part2")


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
    acum_seq01 = []
    acum_seq11 = []
    acum_seq02 = []
    acum_seq12 = []
    count_seq1 = 0
    count_seq2 = 0
    i = 0

    for idx in range(0, len(info)-1, 2):

        # so we still work with the two step size to take the two timepoints
        row0 = df.iloc[idx]
        row1 = df.iloc[idx + 1]

        # extract rows of interest from df
        info0 = info.iloc[idx]
        info1 = info.iloc[idx + 1]

        # extract info
        row0 = pd.to_numeric(row0)
        name0 = info0['patient_name']
        groupMS0 = info0['group']
        label0 = info0['2group']
        study_date0 = info0['study_date']

        row1 = pd.to_numeric(row1)
        groupMS1 = info1['group']
        label1 = info1['2group']
        label = "1" if label0 != label1 else "0"# if labels are different 1 if no 0
        name1 = info1['patient_name']
        study_date1 = info1['study_date']

        # create group
        data = hf.create_group(str(i))

        # create 2 groups for tps
        tp0 = data.create_group('tp0/Left-Hippocampus')
        tp1 = data.create_group('tp1/Left-Hippocampus')

        flair_tp0, t1_tp0, lesion_mask_tp0, brain_mask_tp0 = take_tps(os.path.join(DIR_2,
                                                                                   'RocheLyon_' + groupMS0,
                                                                                   name0),
                                                                      study_date0)

        flair_tp1, t1_tp1, lesion_mask_tp1, brain_mask_tp1 = take_tps(os.path.join(DIR_2,
                                                                                   'RocheLyon_' + groupMS1,
                                                                                   name1),
                                                                      study_date1)

        # putting images as different channel for input
        flair_tp0 = nib.load(flair_tp0).get_fdata()
        t1_tp0 = nib.load(t1_tp0).get_fdata()
        lesion_mask_tp0 = nib.load(lesion_mask_tp0).get_fdata()
        brain_mask_tp0 = nib.load(brain_mask_tp0).get_fdata()

        flair_tp1 = nib.load(flair_tp1).get_fdata()
        t1_tp1 = nib.load(t1_tp1).get_fdata()
        lesion_mask_tp1 = nib.load(lesion_mask_tp1).get_fdata()
        brain_mask_tp1 = nib.load(brain_mask_tp1).get_fdata()

        # this label mask have several labels as to represent
        # different types of lesions depending of different factors
        # binarize this so the cnn does not take this as weights for
        # those roi of the image
        b_lmask_tp0 = lesion_mask_tp0 > 0
        b_lmask_tp0 = b_lmask_tp0.astype(int)

        b_lmask_tp1 = lesion_mask_tp1 > 0
        b_lmask_tp1 = b_lmask_tp1.astype(int)

        # stacking the mask + flair + t1
        input_images_tp0 = np.concatenate(
            (b_lmask_tp0[np.newaxis, ...], flair_tp0[np.newaxis, ...], t1_tp0[np.newaxis, ...]), axis=0)

        input_images_tp1 = np.concatenate(
            (b_lmask_tp1[np.newaxis, ...], flair_tp1[np.newaxis, ...], t1_tp1[np.newaxis, ...]), axis=0)

        counting_seq01 = Counter(np.array(flair_tp0[brain_mask_tp0 > 0], dtype=np.int32))
        counting_seq02 = Counter(np.array(t1_tp0[brain_mask_tp0 > 0], dtype=np.int32))

        counting_seq11 = Counter(np.array(flair_tp1[brain_mask_tp1 > 0], dtype=np.int32))
        counting_seq12 = Counter(np.array(t1_tp1[brain_mask_tp1 > 0], dtype=np.int32))

        if len(acum_seq01) == 0:
            acum_seq01 = counting_seq01
            count_seq01 = len(flair_tp0[brain_mask_tp0 > 0])
        else:
            acum_seq01 = np.add(acum_seq01, counting_seq01)
            count_seq01 += len(flair_tp0[brain_mask_tp0 > 0])

        if len(acum_seq11) == 0:
            acum_seq11 = counting_seq11
            count_seq11 = len(flair_tp1[brain_mask_tp1 > 0])
        else:
            acum_seq11 = np.add(acum_seq11, counting_seq11)
            count_seq11 += len(flair_tp1[brain_mask_tp1 > 0])

        ################

        if len(acum_seq02) == 0:
            acum_seq02 = counting_seq02
            count_seq02 = len(t1_tp0[brain_mask_tp0 > 0])
        else:
            acum_seq02 = np.add(acum_seq02, counting_seq02)
            count_seq02 += len(t1_tp0[brain_mask_tp0 > 0])

        if len(acum_seq12) == 0:
            acum_seq12 = counting_seq12
            count_seq12 = len(t1_tp1[brain_mask_tp1 > 0])
        else:
            acum_seq12 = np.add(acum_seq12, counting_seq12)
            count_seq12 += len(t1_tp1[brain_mask_tp1 > 0])

        # clean images because of space needs
        del t1_tp0, flair_tp0, lesion_mask_tp0, brain_mask_tp0, b_lmask_tp0
        del t1_tp1, flair_tp1, lesion_mask_tp1, brain_mask_tp1, b_lmask_tp1

        tp0.create_dataset('vol_with_bg', dtype='f', compression='gzip',
                           data=input_images_tp0)

        tp1.create_dataset('vol_with_bg', dtype='f', compression='gzip',
                           data=input_images_tp1)

        # create dataset tabular
        tp0.create_dataset('tabular', data=row0[:])

        tp1.create_dataset('tabular', data=row1[:])

        # cleaning images and tabular because of space needs
        del input_images_tp0, row0
        del input_images_tp1, row1

        # create attributes
        # RID scalar - image ID - unique identifier
        tp0.attrs["RID"] = name0

        # visit code - string
        tp0.attrs["VISCODE"] = study_date0

        # labels #

        # DX - diagnosis - CIS, PP, SP, RR
        tp0.attrs["DX"] = label
        tp0.attrs["GROUP"] = label0

        # create attributes
        # RID scalar - image ID - unique identifier
        tp1.attrs["RID"] = name1

        # visit code - string
        tp1.attrs["VISCODE"] = study_date1

        # labels #
        tp1.attrs["DX"] = label

        # DX - diagnosis - CIS, PP, SP, RR
        tp1.attrs["GROUP"] = label1


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

    aux_seq01, aux_seq11, aux_seq02, aux_seq12 = np.zeros(max(acum_seq01.keys()) + 1), np.zeros(max(acum_seq11.keys()) + 1), np.zeros(max(acum_seq02.keys()) + 1), np.zeros(max(acum_seq12.keys()) + 1)
    for key01, value01 in acum_seq01.items():
        aux_seq01[key01] = value01
    for key11, value11 in acum_seq11.items():
        aux_seq11[key11] = value11
    for key02, value02 in acum_seq02.items():
        aux_seq02[key02] = value02
    for key12, value12 in acum_seq12.items():
        aux_seq12[key12] = value12

    acum_seq01 = aux_seq01
    acum_seq11 = aux_seq11
    acum_seq02 = aux_seq02
    acum_seq12 = aux_seq12

    acum_seq1 = np.concatenate((acum_seq01, acum_seq11), axis=0)
    acum_seq2 = np.concatenate((acum_seq02, acum_seq12), axis=0)

    sum_seq1 = [i * value for i, value in enumerate(acum_seq1)]
    sum_seq2 = [i * value for i, value in enumerate(acum_seq2)]

    mean_values = np.array([np.sum(sum_seq1) / count_seq1, np.sum(sum_seq2) / count_seq2])
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

    import shutil

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

        # TODO: add shape manually shape
        name_columns = np.array(part1_tabular_columns)
        tabular.create_dataset('columns', shape=name_columns.shape, dtype=dt, data=name_columns)
        # means
        means = (part1_tabular_mean + part2_tabular_mean) / 2
        tabular.create_dataset('mean', shape=means.shape, dtype='f', data=means)
        # stddevs
        stddevs = np.maximum(part1_tabular_std, part2_tabular_std)
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
    df.dropna(inplace=True)

    # grouping the cis-rr and sp-pp patients
    conditions = [
        (df["group"].str.contains("CIS")),
        (df["group"].str.contains("RR")),
        (df["group"].str.contains("SP")),
        (df["group"].str.contains("PP"))
    ]

    values = ["CISRR", "CISRR", "SPPP", "SPPP"]

    df["2group"] = np.select(conditions, values)

    # do categorical to numerical with pivotal encoding
    # because when giving the mean and stddev will not have sense

    # keep label in another variable
    info = df[['study_date', 'patient_name', 'group', '2group']].copy()
    df.drop(columns=['study_date', 'patient_name', 'group', '2group'], inplace=True)

    # change age and group to pivotal
    new_cols_sex = pd.get_dummies(df.patient_sex)
    df = pd.concat([df, new_cols_sex], axis='columns')
    df.drop(columns=['patient_sex'], axis='columns', inplace=True)

    # change type of the column
    df = df.apply(pd.to_numeric)

    # standard scale data
    df.iloc[:, 0:-1] = df.iloc[:, 0:-1].apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)

    return df, info


def usable_images(df, info, num_tps):
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
    # print("columns names ", columns_names)
    usable = pd.DataFrame(columns=columns_names)
    # print("usable columns ", usable.columns)
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

                # if there is not enough cases for the required time points
                if num_tps > len(tps_folder) - 1:
                    tabular_names = np.delete(tabular_names, np.where(tabular_names == case))
                    continue
                # tps in the dataframe
                tps_df = info[info['patient_name'].str.contains(case)]['study_date'].to_numpy()
                # use the first of the intersection
                tps = list(set(tps_folder).intersection(tps_df))
                tps.sort()
                # get the index of the patient that match the case
                # in the second time point available
                # since some variables in the table represent counting of appearing, disappearing and volumes
                # of lesions in different zones of the brain
                # will save th tp that have images

                # save the name for the split
                data_names.append(case)

                for tp in tps:
                    idx = info.index[info['patient_name'].str.contains(case) & info['study_date'].str.contains(tp)]

                    # save the valid info we want to use
                    usable.loc[len(usable.index)] = np.concatenate((info.loc[idx].to_numpy().reshape(-1),
                                                                    df.loc[idx].to_numpy().reshape(-1)),
                                                                   axis=0)

    return data_names, usable


def fileSplit(df_main, file_part1, file_part2, file_part3, file_part4):
    names_part1 = pd.read_csv(file_part1)["patient_name"].values.tolist()
    names_part2 = pd.read_csv(file_part2)["patient_name"].values.tolist()
    names_part3 = pd.read_csv(file_part3)["patient_name"].values.tolist()
    names_part4 = pd.read_csv(file_part4)["patient_name"].values.tolist()

    df_part1 = df_main.set_index('patient_name').loc[names_part1].reset_index()
    df_part2 = df_main.set_index('patient_name').loc[names_part2].reset_index()
    df_part3 = df_main.set_index('patient_name').loc[names_part3].reset_index()
    df_part4 = df_main.set_index('patient_name').loc[names_part4].reset_index()

    return df_part1, df_part2, df_part3, df_part4


def read_hf(hf):
    contando = [0, 0, 0, 0]
    for key, value in hf.items():
        if "stats" in key:
            print("with stats key ", value.keys())
            break
        if "CIS" in value.attrs["DX"]:
            contando[0] += 1
        elif "RR" in value.attrs["DX"]:
            contando[1] += 1
        elif "SP" in value.attrs["DX"]:
            contando[2] += 1
        elif "PP" in value.attrs["DX"]:
            contando[3] += 1
        else:
            break
    print(" CIS, RR, SP, PP")
    print(contando)


def data_to_hdf5_part4(path1, path2, num_tps, file_part1, file_part2, file_part3, file_part4):
    # data
    path_tabular_1 = os.path.join(path1)
    path_tabular_2 = os.path.join(path2)

    data_df_1 = pd.read_csv(path_tabular_1)
    data_df_2 = pd.read_csv(path_tabular_2)

    # clean the tabular info
    clean_df, info = cleaning_table(data_df_1, data_df_2)
    # splitting
    # get valid patient names from images availability
    data_names, data = usable_images(clean_df, info, num_tps)


    df_part1, df_part2, df_part3, df_part4 = fileSplit(data, file_part1, file_part2, file_part3, file_part4)

    part1_info = df_part1[["study_date", "patient_name", "group",'2group']]
    df_part1.drop(columns=["study_date", "patient_name", "group", '2group'], inplace=True)

    part2_info = df_part2[["study_date", "group", "patient_name", '2group']]
    df_part2.drop(columns=["study_date", "group", "patient_name", '2group'], inplace=True)

    part3_info = df_part3[["study_date", "patient_name", "group", '2group']]
    df_part3.drop(columns=["study_date", "patient_name", "group", '2group'], inplace=True)

    part4_info = df_part4[["study_date", "patient_name", "group", '2group']]
    df_part4.drop(columns=["study_date", "patient_name", "group", '2group'], inplace=True)

    # TODO: works with several folds
    print('data_part1.shape ', df_part1.shape)
    write_hdf5("part1", df_part1, part1_info)
    print('data_part2.shape ', df_part2.shape)
    write_hdf5("part2", df_part2, part2_info)
    print('data_part3.shape ', df_part3.shape)
    write_hdf5("part3", df_part3, part3_info)
    print('data_part4.shape ', df_part4.shape)
    write_hdf5("part4", df_part4, part4_info)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    data_to_hdf5_part4(os.path.join(DIR_1, "dataframe_rochelyon_4groups.csv"),
                       os.path.join(DIR_2, "dataframe_cross_rochelyon.csv"),
                       1,
                       '/home/ecarvajal/test/part1_mix_balanced.csv',
                       '/home/ecarvajal/test/part2_mix_balanced.csv',
                       '/home/ecarvajal/test/part3_mix_balanced.csv',
                       '/home/ecarvajal/test/part4_mix_balanced.csv')


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
