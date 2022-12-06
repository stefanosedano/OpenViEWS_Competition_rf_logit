import re
import pandas as pd
import numpy as np




def build_dep_frame(myObj):
    dataframe = prepare_dep_frame(myObj)

    #clean
    keep_columns = []
    for col in dataframe.columns:
        if "_T" in col:
            if (not ("_T0" in col)):
                keep_columns.append(col)

    keep_columns = keep_columns + myObj.link_columns

    dataframe = dataframe[keep_columns]

    return dataframe




def prepare_dep_frame(myObj):
    ## lags future (minus)
    ##
    dataset = myObj.dataset_cm.copy()
    periods = [1, 2, 3, 4, 5, 6, 7]
    for dep_var_col in myObj.dep_var_cols:
        for period in periods:
            dataset[
                dep_var_col + "_T" + str(period)
                ] = dataset.groupby("country_id")[dep_var_col].shift(-period)

        ## Add the Log. Change
        for period in periods:
            dataset[
                dep_var_col + "_LOGCHANGE_T" + str(period)
                ] = np.log(
                dataset[dep_var_col + "_T" + str(period)] + 1
            ) - np.log(
                dataset[dep_var_col] + 1
            )

        ## append "_T0" label to all dep_var columns
        dataset = dataset.rename(
            columns={dep_var_col: dep_var_col + "_T0"}
        )
    return dataset

########### IND

def build_indep_frame(myObj):
    dataframe = prepare_indep_frame(myObj)

    #clean
    keep_columns = []
    for col in dataframe.columns:
        if (("_T" in col) or ("PAST_MEDIAN" in col)):
            if (not ("_T0" in col)):
                keep_columns.append(col)

    keep_columns = keep_columns + myObj.link_columns

    dataframe = dataframe[keep_columns]

    return dataframe




def prepare_indep_frame(myObj):
    ## lags future (minus)
    ##
    dataset = myObj.dataset_cm.copy()
    periods = [1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
    for dep_var_col in myObj.dep_var_cols:
        for period in periods:
            dataset[
                dep_var_col + "_T-" + str(period)
                ] = dataset.groupby("country_id")[dep_var_col].shift(period)

        ## Add the Log. Change
        for period in periods:
            dataset[
                dep_var_col + "_LOGCHANGE_T-" + str(period)
                ] = np.log(dataset[dep_var_col + "_T-" + str(period)] + 1) - \
                    np.log(dataset[dep_var_col] + 1
            )



        def log_change_these(row):
            return (np.log(row["FATALITIES_PAST_MEDIAN"] + 1)) - (np.log(row[dep_var_col] + 1))

        columns_var_past=[]
        for col in dataset.columns:
            if dep_var_col + "_T-" in col:
                if not("LOGCHANGE" in col):
                    columns_var_past.append(col)

        dataset["FATALITIES_PAST_SUM"] = dataset[columns_var_past].sum(axis=1)

        dataset["FATALITIES_PAST_MEDIAN"] = dataset[columns_var_past].median(axis=1)
        dataset[f"PAST_MEDIAN_{dep_var_col}"] = dataset.apply(log_change_these, axis=1)

        dataset.loc[dataset[f"PAST_MEDIAN_{dep_var_col}"].isnull(), f"PAST_MEDIAN_{dep_var_col}"] = dataset["FATALITIES_PAST_SUM"]

        dataset = dataset.drop(columns=["FATALITIES_PAST_SUM","FATALITIES_PAST_MEDIAN"])

        ## append "_T0" label to all dep_var columns
        dataset = dataset.rename(
            columns={dep_var_col: dep_var_col + "_T0"}
        )

    return dataset

