from AutoClean import AutoClean
import pandas as pd
import numpy as np
import random

class Tests:

    def run(data_path):
        outcome_all = {}
        
        for dataset in data_path:
            original_df, output_df = Tests._import_data(dataset)
            print("Imported dataset", dataset)
            output_df = Tests._add_missing(output_df)
            print("Added random missing values")
            output_df = Tests._run_AutoClean(output_df)
            print("Applied AutoClean")
            evaluation = Tests._result(output_df, original_df)
            print("Computed evaulation result")
            outcome_all[dataset] = evaluation          
        return outcome_all

    def _import_data(dataset): #'data/cars/cars.csv'
        try:
            inp = pd.read_csv(dataset)
            inp = inp.dropna()
            inp.reset_index(drop=True)
            out = inp.copy()
        except Exception as e:
            print(e)

        return inp, out
    
    def _add_missing(input_df):

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

    def _run_AutoClean(input_df):
        try:
            autoclean = AutoClean(input_df)
            input_df = autoclean.output_data
        except Exception as e:
            print(e)
        return input_df

    def _result(output_df, original_df):
        try:
            outcome_dataset = {}
            cols_num = output_df.select_dtypes(include=np.number).columns 
            for feature in original_df.columns:
                if feature in cols_num:
                    rmse = np.sqrt(np.mean((output_df[feature]-original_df[feature])**2))
                    outcome_dataset[feature] = {'rmse': round(rmse, 4)}
                else:
                    num_correct = sum(p == t for p, t in zip(output_df[feature], original_df[feature]))
                    acc = (num_correct/len(original_df[feature])) * 100.0
                    outcome_dataset[feature] = {'acc': round(acc, 4)}
        except Exception as e:
            print(e)
        
        return outcome_dataset         