import sys,os
import numpy as np
import pandas as pd
import pickle
from variables.UCDP.ucdp import UCDP
from model.random_forest.random_forest import RandomForest
from metrics import metrics
import datetime
from datetime import date
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')


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
3. A similar set of six sets of forecasts for sb conflict for each of the months January 2014 through
December 2016, for six different steps forward s, based on data up to one month before the step
counter starts.
"""

start_forecast = 201401
end_forecast = 201612

all_forcast_dates = []
#all date from start to end
new_date = start_forecast
while (new_date <= end_forecast):
    all_forcast_dates.append(new_date)
    new_date = datetime.datetime.strptime(str(new_date), '%Y%m').date()
    new_date = new_date + relativedelta(months=(1))
    new_date = int(new_date.strftime("%Y%m"))

list_of_datasets=[]
for pdate in all_forcast_dates:
    for period in [2,3,4,5,6,7]:
        pdate_date = datetime.datetime.strptime(str(pdate), '%Y%m').date()
        last_usable_date = pdate_date + relativedelta(months=(-period))
        last_usable_date = int(last_usable_date.strftime("%Y%m"))
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
    outputfile = f"tasks/task3/Period_{period}_CutOff_{last_usable_date}_Prediction_{prediction_date}.csv"
    if not os.path.exists(outputfile):
        print(outputfile)
        out = RandomForest(dataset,full_dataset_ucdp)
        out.dataset["predictions"].to_csv(outputfile,index=False)
    else:
        print("wierd")


all_dataframes =[]
task1_dir = "tasks/task3"
for file in os.listdir(task1_dir):
    if file.endswith(".csv"):
        mydataframe = pd.read_csv(os.path.join(task1_dir, file))
        mydataframe["run"] = file
        all_dataframes.append(mydataframe)
all_dataframes = pd.concat(all_dataframes, axis=0)

outputfile = f"tasks/task3/task3_all.csv"
all_dataframes.to_csv(outputfile,index=False)

outputfile = f"tasks/task3/task3_to_deliver.csv"
all_dataframes[["month_id","country_id","run","rf_logit"]].to_csv(outputfile,index=False)

#def process_logit_results(row,p):
#    out_value = 0
#    if ((row["rf"] > 0) and (row["logit_escalation"] > p / 100)):
#        out_value = row["rf"]
#    if ((row["rf"] < 0) and (row["logit_deescalation"] > p / 100)):
#        out_value = row["rf"]
#    return out_value

#for p in range (0,100):
#
#    all_dataframes["rf_logit"] = all_dataframes.apply(process_logit_results,args=(p,), axis=1)
#    print(metrics.tadda_score(all_dataframes["Y_true"], all_dataframes["rf_logit"],1),p)
#
#all_dataframes["rf_logit"] = all_dataframes.apply(process_logit_results,args=(90,), axis=1)


print("no_change")
print(metrics.get_all_taddas(all_dataframes["Y_true"],np.zeros_like(all_dataframes["Y_true"])))

print("rf")
print(metrics.get_all_taddas(all_dataframes["Y_true"],all_dataframes["rf"]))

print("rf_logit")
print(metrics.get_all_taddas(all_dataframes["Y_true"],all_dataframes["rf_logit"]))

print("median")
print(metrics.get_all_taddas(all_dataframes["Y_true"],all_dataframes["median"]))






