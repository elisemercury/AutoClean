from AutoClean import AutoClean
import pandas as pd
import numpy as np
import random

class Tests:

    def __init__(self, data_path):
        outcome_all = {}
        self.outcome_all = outcome_all
        try:
            for dataset in data_path:
                original_df, output_df = Tests.import_data(self, data_path)
                output_df = Tests.add_missing(self, output_df)
                output_df = Tests.run_AutoClean(self, output_df)
                evaluation = Tests.result(self, output_df, original_df)
                outcome_all[dataset] = evaluation
        except Exception as e:
            print(e)
        return outcome_all

    def import_data(self, data_path): #'data/cars/cars.csv'
        try:
            input = pd.read_csv(data_path)
            output = input.copy()
        except Exception as e:
            print(e)
        return input, output

    def add_missing(self, input_df):

        # add a few NANs
        try:
            ix = [(row, col) for row in range(input_df.shape[0]) for col in range(input_df.shape[1])]
            for row, col in random.sample(ix, int(round(.05*len(ix)))):
                try:
                    input_df.iat[row, col] = np.nan
                except:
                    pass
        except Exception as e:
            print(e)
        return input_df

    def run_AutoClean(self, input_df):
        try:
            autoclean = AutoClean(input_df)
            input_df = autoclean.output_data
        except Exception as e:
            print(e)
        return input_df

    def result(self, output_df, original_df):
        try:
            outcome_dataset = {}
            cols_num = output_df.select_dtypes(include=np.number).columns 
            for feature in output_df.columns:
                if feature in cols_num:
                    rmse = np.sqrt(np.mean((output_df[feature]-original_df[feature])**2))
                    outcome_dataset[feature] = {'rmse': rmse}
                else:
                    num_correct = sum(p == t for p, t in zip(output_df[feature], original_df[feature]))
                    acc = (num_correct/len(original_df[feature])) * 100.0
                    outcome_dataset[feature] = {'acc': acc}
        except Exception as e:
            print(e)
        
        return outcome_dataset
