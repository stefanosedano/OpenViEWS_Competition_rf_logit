import sys,os
import numpy as np
import pandas as pd
import pickle
from variables.UCDP.ucdp import UCDP
from model.random_forest.random_forest import RandomForest
from metrics import metrics

def calculate_cut_off(period,last_usable_date):
    """
    Calcualte cut-off giving the period and last usable date

    if last usable date = 202008
    and period 2

    cut_of_yearmo_period = forecast_date - mo = 202006
    """
    import datetime
    from datetime import date
    from dateutil.relativedelta import relativedelta

    forecast_date = datetime.datetime.strptime(str(last_usable_date), '%Y%m').date()
    cut_of_yearmo_period = forecast_date + relativedelta(months=-(period))

    prediction = datetime.datetime.strptime(str(last_usable_date), '%Y%m').date() + relativedelta(months=(period))

    return int(cut_of_yearmo_period.strftime("%Y%m")),int(prediction.strftime("%Y%m"))

def split_train_test(full_dataset_ucdp,cut_of_yearmo_period,period,test_date):

    Y_label = f"ged_count_sb_LOGCHANGE_T{period}"

    trainX = full_dataset_ucdp.indep_frame.loc[full_dataset_ucdp.indep_frame.yearmo <=cut_of_yearmo_period]
    trainY = full_dataset_ucdp.dep_frame.loc[full_dataset_ucdp.dep_frame.yearmo <= cut_of_yearmo_period][Y_label]

    testX = full_dataset_ucdp.indep_frame.loc[full_dataset_ucdp.indep_frame.yearmo == test_date]
    testY = full_dataset_ucdp.dep_frame.loc[full_dataset_ucdp.dep_frame.yearmo == test_date][Y_label]

    return trainX, trainY, testX, testY

#----------------------------------------------------------------------------------------------------------------------

full_dataset_ucdp = UCDP()

"""
1. ‘True’ forecasts for sb conflict for each of the months October 2020 through March 2021, based on
data up to and including August 2020.
"""
"""
test yearmo == 202008, periods T2,T3,T4,T5,T6,T7
"""

last_usable_date = 202008
steps=[2,3,4,5,6,7]

list_of_datasets=[]

for period in steps:
    cut_of_yearmo_period,prediction_date = calculate_cut_off(period, last_usable_date)
    trainX, trainY, testX, testY = split_train_test(full_dataset_ucdp,cut_of_yearmo_period,period,last_usable_date)

    dataset = {
        "trainX": trainX,
        "trainY": trainY,
        "testX": testX,
        "testY": testY,
        "period": period,
        "cut_off": last_usable_date,
        "prediction_date":prediction_date
    }
    list_of_datasets.append(dataset)



for dataset in list_of_datasets:
    period = dataset["period"]
    prediction_date = dataset["prediction_date"]
    last_usable_date = dataset["cut_off"]
    outputfile = f"tasks/task1/Period_{period}_CutOff_{last_usable_date}_Prediction_{prediction_date}.csv"
    if not os.path.exists(outputfile):
        print(outputfile)
        out = RandomForest(dataset,full_dataset_ucdp)
        out.dataset["predictions"].to_csv(outputfile,index=False)


all_dataframes =[]
task1_dir = "tasks/task1"
for file in os.listdir(task1_dir):
    if file.endswith(".csv"):
        mydataframe = pd.read_csv(os.path.join(task1_dir, file))
        mydataframe["run"] = file
        all_dataframes.append(mydataframe)
all_dataframes = pd.concat(all_dataframes, axis=0)

outputfile = f"tasks/tasks_aggregated_outputs/task1_all.csv"
all_dataframes.to_csv(outputfile,index=False)

print(np.sum(all_dataframes[["Y_true"]].values == 0))
print(np.sum(all_dataframes[["Y_true"]].values > 0))
print(np.sum(all_dataframes[["Y_true"]].values < 0))
print(all_dataframes[["Y_true"]]["Y_true"].count())

outputfile = f"tasks/tasks_aggregated_outputs/task1_to_deliver.csv"
all_dataframes[["month_id","country_id","run","rf_logit"]].to_csv(outputfile,index=False)

print("no_change")
print(metrics.get_all_taddas(all_dataframes["Y_true"],np.zeros_like(all_dataframes["Y_true"])))

print("rf")
print(metrics.get_all_taddas(all_dataframes["Y_true"],all_dataframes["rf"]))

print("rf_logit")
print(metrics.get_all_taddas(all_dataframes["Y_true"],all_dataframes["rf_logit"]))

print("median")
print(metrics.get_all_taddas(all_dataframes["Y_true"],all_dataframes["median"]))





