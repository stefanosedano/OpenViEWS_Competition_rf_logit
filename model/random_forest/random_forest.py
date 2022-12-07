import pickle
import mlflow
import numpy as np


from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from statsmodels.formula.api import logit
import numpy as np


def logit_classifier(dataset,full_dataset_ucdp):


    results_ready = 0
    remove_one_var = 0
    list_of_variable = [w.replace('-', '_') for w in dataset['most_important_variables']]

    while not(results_ready):
        try:


            train_X =dataset["trainX"]
            test_X =dataset["testX"]
            train_y =dataset["trainY"]
            train_X.columns = [w.replace('-', '_') for w in train_X.columns]
            test_X.columns = [w.replace('-', '_') for w in test_X.columns]
            test_X["predictionRF"] = dataset["predictionRF"]

            # select y_pred > 0
            train_y_logit = train_y > 0
            train_y_logit = train_y_logit.astype(int)
            train_X["Y_bool"] = train_y_logit
            formula_linear_regression = "Y_bool ~ " + ("+").join(list_of_variable)
            list_of_variable_widh_dep = list_of_variable.copy()
            list_of_variable_widh_dep.append("Y_bool")
            current_model_logit = logit(formula_linear_regression, data=train_X[list_of_variable_widh_dep]).fit()
            #subset test record with rf prediction > 0 ---> escalation

            predictions_escalation = current_model_logit.predict(test_X[list_of_variable])
            test_X["logit_escalation"] = predictions_escalation

            # select y_pred < 0
            train_y_logit = train_y < 0
            train_y_logit = train_y_logit.astype(int)
            train_X["Y_bool"] = train_y_logit
            formula_linear_regression = "Y_bool ~ " + ("+").join(list_of_variable)
            list_of_variable_widh_dep = list_of_variable.copy()
            list_of_variable_widh_dep.append("Y_bool")
            current_model_logit = logit(formula_linear_regression, data=train_X[list_of_variable_widh_dep]).fit()
            #subset test record with rf prediction > 0 ---> escalation

            predictions_deescalation = current_model_logit.predict(test_X[list_of_variable])
            test_X["logit_deescalation"] = predictions_deescalation

            def process_logit_results(row):
                out_value = 0
                if ((row["predictionRF"] > 0) and (row["logit_escalation"] > 0.9)):
                    out_value = row["predictionRF"]

                if ((row["predictionRF"] < 0) and (row["logit_deescalation"] > 0.9)):
                    out_value =  row["predictionRF"]

                return out_value

            test_X["predictionRFLOGIT"] = 0.0
            test_X["predictionRFLOGIT"] = test_X.apply(process_logit_results, axis=1)

            predictions = pd.DataFrame()
            predictions["month_id"] = test_X["month_id"]
            predictions["country_id"] = test_X["country_id"]
            predictions["country_name"] = test_X["country_name"]
            predictions["yearmo"] = test_X["yearmo"]
            predictions["logit_escalation"] = test_X["logit_escalation"]
            predictions["logit_deescalation"] = test_X["logit_deescalation"]
            predictions["rf"] = test_X["predictionRF"]
            predictions["rf_logit"] = test_X["predictionRFLOGIT"]
            predictions["rf_logit"] = test_X["predictionRFLOGIT"]
            predictions["median"] = test_X["PAST_MEDIAN_ged_count_sb"]
            predictions["Y_true"] = dataset["testY"]


            dataset["predictions"] = predictions

            results_ready = 1

        except:
            remove_one_var = remove_one_var + 1
            list_of_variable = list_of_variable[:-remove_one_var]
            pass

    return dataset


def train_period_model(train_X, train_y):
    random_state = 1234

    # TODO: Hyperparameter Tuning

    features = train_X.columns

    model = RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=random_state,
        verbose=0
    ).fit(train_X[features], train_y)




    ### Train final Model on full Training Data
    model.fit(train_X[features], train_y)

    return model, features



class RandomForest:

    def __init__(self, dataset,full_dataset_ucdp):
        #dataset = {
        #    "trainX": trainX,
        #    "trainY": trainY,
        #    "testX": testX,
        #    "testY": testY,
        #    "period": period,
        #    "cut_off": last_usable_date,
        #    "prediction_date": prediction_date
        #}

        train_X = dataset["trainX"].drop(columns=full_dataset_ucdp.link_columns)
        train_y = dataset["trainY"].drop(columns=full_dataset_ucdp.link_columns)
        test_X = dataset["testX"].drop(columns=full_dataset_ucdp.link_columns)
        test_Y = dataset["testY"].drop(columns=full_dataset_ucdp.link_columns)
        features = train_X.columns

        model, model_features = train_period_model(train_X, train_y)

        # make a prediction
        y_pred = model.predict(test_X)

        # most important features
        sorted_idx = model.feature_importances_.argsort()
        a = list(reversed(model_features[sorted_idx]))
         #b= list(reversed(model.feature_importances_[sorted_idx]))
        list_of_variable = a[0:100]

        dataset["predictionRF"] = y_pred

        dataset["most_important_variables"] = list_of_variable

        #use the logit classifier to improove the results
        dataset = logit_classifier(dataset,full_dataset_ucdp)

        self.dataset = dataset






