import sys,os
import numpy as np
import pandas as pd
import re
import pickle
from variables.UCDP import ucdp_build


class UCDP:
    def __init__(self):

        ged_cm = pd.pandas.read_parquet("variables/UCDP/datasets/ged_cm.parquet").reset_index()
        skeleton_cm = pd.pandas.read_parquet("variables/UCDP/datasets/skeleton_cm_africa.parquet").reset_index()

        self.dataset_cm = ged_cm.merge(skeleton_cm,how="right")


        # variables to consider
        self.dep_var_cols = ["ged_best_sb","ged_count_sb","ged_best_os","ged_count_os","ged_best_ns","ged_count_ns"]
        self.link_columns = ["month_id","country_id","country_name","yearmo"]

        def build_yearmo(row):
            return int(f"{row.year}{str(row.month).zfill(2)}")
        self.dataset_cm["yearmo"] = self.dataset_cm.apply(build_yearmo,axis=1)

        # fill na with zero
        self.dataset_cm = self.dataset_cm.fillna(0)

        self.dataset_cm = self.dataset_cm.drop(columns=["in_africa","month","year"])

        self.dep_frame = ucdp_build.build_dep_frame(self)
        self.indep_frame = ucdp_build.build_indep_frame(self)

        ##set overall limits
        self.dep_frame = self.dep_frame.loc[self.dep_frame.yearmo>=199101]
        self.dep_frame = self.dep_frame.loc[self.dep_frame.yearmo <= 202008]
        self.indep_frame = self.indep_frame.loc[self.indep_frame.yearmo >= 199101]
        self.indep_frame = self.indep_frame.loc[self.indep_frame.yearmo <= 202008]



if __name__ == "__main__":
    from variables.UCDP.ucdp import UCDP

    full_dataset_ucdp = UCDP()


