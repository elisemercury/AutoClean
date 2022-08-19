# AutoClean 2022
# For detailed documentation and usage guide, please visit the official GitHub Repo.
# https://github.com/elisemercury/AutoClean

from timeit import default_timer as timer
import numpy as np
import pandas as pd
from math import isnan
from sklearn import preprocessing
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

'''
Modules are used by the AutoClean pipeline for data cleaning and preprocessing.
'''

class MissingValues:

    def handle(self, df, _n_neighbors=3):
        # function for handling missing values in the data
        if self.missing_num or self.missing_categ:
            logger.info('Started handling of missing values...', str(self.missing_num).upper())
            start = timer()
            self.count_missing = df.isna().sum().sum()

            if self.count_missing != 0:
                logger.info('Found a total of {} missing value(s)', self.count_missing)
                df = df.dropna(how='all')
                df.reset_index(drop=True)
                
                if self.missing_num: # numeric data
                    logger.info('Started handling of NUMERICAL missing values... Method: "{}"', str(self.missing_num).upper())
                    # automated handling
                    if self.missing_num == 'auto': 
                        self.missing_num = 'linreg'
                        lr = LinearRegression()
                        df = MissingValues._lin_regression_impute(self, df, lr)
                        self.missing_num = 'knn'
                        imputer = KNNImputer(n_neighbors=_n_neighbors)
                        df = MissingValues._impute(self, df, imputer, type='num')
                    # linear regression imputation
                    elif self.missing_num == 'linreg':
                        lr = LinearRegression()
                        df = MissingValues._lin_regression_impute(self, df, lr)
                    # knn imputation
                    elif self.missing_num == 'knn':
                        imputer = KNNImputer(n_neighbors=_n_neighbors)
                        df = MissingValues._impute(self, df, imputer, type='num')
                    # mean, median or mode imputation
                    elif self.missing_num in ['mean', 'median', 'most_frequent']:
                        imputer = SimpleImputer(strategy=self.missing_num)
                        df = MissingValues._impute_missing(self, df, imputer, type='num')
                    # delete missing values
                    elif self.missing_num == 'delete':
                        df = MissingValues._delete(self, df, type='num')
                        logger.debug('Deletion of {} NUMERIC missing value(s) succeeded', self.count_missing-df.isna().sum().sum())      

                if self.missing_categ: # categorical data
                    logger.info('Started handling of CATEGORICAL missing values... Method: "{}"', str(self.missing_categ).upper())
                    # automated handling
                    if self.missing_categ == 'auto':
                        self.missing_categ = 'logreg'
                        lr = LogisticRegression()
                        df = MissingValues._log_regression_impute(self, df, lr)
                        self.missing_categ = 'knn'
                        imputer = KNNImputer(n_neighbors=_n_neighbors)
                        df = MissingValues._impute(self, df, imputer, type='categ')
                    elif self.missing_categ == 'logreg':
                        lr = LogisticRegression()
                        df = MissingValues._log_regression_impute(self, df, lr)
                    # knn imputation
                    elif self.missing_categ == 'knn':
                        imputer = KNNImputer(n_neighbors=_n_neighbors)
                        df = MissingValues._impute(self, df, imputer, type='categ')  
                    # mode imputation
                    elif self.missing_categ == 'most_frequent':
                        imputer = SimpleImputer(strategy=self.missing_categ)
                        df = MissingValues._impute(self, df, imputer, type='categ')
                    # delete missing values                    
                    elif self.missing_categ == 'delete':
                        df = MissingValues._delete(self, df, type='categ')
                        logger.debug('Deletion of {} CATEGORICAL missing value(s) succeeded', self.count_missing-df.isna().sum().sum())
            else:
                logger.debug('{} missing values found', self.count_missing)
            end = timer()
            logger.info('Completed handling of missing values in {} seconds', round(end-start, 6))  
        else:
            logger.info('Skipped handling of missing values')
        return df

    def _impute(self, df, imputer, type):
        # function for imputing missing values in the data
        cols_num = df.select_dtypes(include=np.number).columns 

        if type == 'num':
            # numerical features
            for feature in df.columns: 
                if feature in cols_num:
                    if df[feature].isna().sum().sum() != 0:
                        try:
                            df_imputed = pd.DataFrame(imputer.fit_transform(np.array(df[feature]).reshape(-1, 1)))
                            counter = df[feature].isna().sum().sum() - df_imputed.isna().sum().sum()

                            if (df[feature].fillna(-9999) % 1  == 0).all():
                                df[feature] = df_imputed
                                # round back to INTs, if original data were INTs
                                df[feature] = df[feature].round()
                                df[feature] = df[feature].astype('Int64')                                        
                            else:
                                df[feature] = df_imputed
                            if counter != 0:
                                logger.debug('{} imputation of {} value(s) succeeded for feature "{}"', str(self.missing_num).upper(), counter, feature)
                        except:
                            logger.warning('{} imputation failed for feature "{}"', str(self.missing_num).upper(), feature)
        else:
            # categorical features
            for feature in df.columns:
                if feature not in cols_num:
                    if df[feature].isna().sum()!= 0:
                        try:
                            mapping = dict()
                            mappings = {k: i for i, k in enumerate(df[feature].dropna().unique(), 0)}
                            mapping[feature] = mappings
                            df[feature] = df[feature].map(mapping[feature])

                            df_imputed = pd.DataFrame(imputer.fit_transform(np.array(df[feature]).reshape(-1, 1)), columns=[feature])    
                            counter = sum(1 for i, j in zip(list(df_imputed[feature]), list(df[feature])) if i != j)

                            # round to integers before mapping back to original values
                            df[feature] = df_imputed
                            df[feature] = df[feature].round()
                            df[feature] = df[feature].astype('Int64')  

                            # map values back to original
                            mappings_inv = {v: k for k, v in mapping[feature].items()}
                            df[feature] = df[feature].map(mappings_inv)
                            if counter != 0:
                                logger.debug('{} imputation of {} value(s) succeeded for feature "{}"', self.missing_categ.upper(), counter, feature)
                        except:
                            logger.warning('{} imputation failed for feature "{}"', str(self.missing_categ).upper(), feature)
        return df

    def _lin_regression_impute(self, df, model):
        # function for predicting missing values with linear regression
        cols_num = df.select_dtypes(include=np.number).columns
        mapping = dict()
        for feature in df.columns:
            if feature not in cols_num:
                # create label mapping for categorical feature values
                mappings = {k: i for i, k in enumerate(df[feature])}
                mapping[feature] = mappings
                df[feature] = df[feature].map(mapping[feature])
        for feature in cols_num: 
                try:
                    test_df = df[df[feature].isnull()==True].dropna(subset=[x for x in df.columns if x != feature])
                    train_df = df[df[feature].isnull()==False].dropna(subset=[x for x in df.columns if x != feature])
                    if len(test_df.index) != 0:
                        pipe = make_pipeline(StandardScaler(), model)

                        y = np.log(train_df[feature]) # log-transform the data
                        X_train = train_df.drop(feature, axis=1)
                        test_df.drop(feature, axis=1, inplace=True)
                        
                        try:
                            model = pipe.fit(X_train, y)
                        except:
                            y = train_df[feature] # use non-log-transformed data
                            model = pipe.fit(X_train, y)
                        if (y == train_df[feature]).all():
                            pred = model.predict(test_df)
                        else:
                            pred = np.exp(model.predict(test_df)) # predict values

                        test_df[feature]= pred

                        if (df[feature].fillna(-9999) % 1  == 0).all():
                            # round back to INTs, if original data were INTs
                            test_df[feature] = test_df[feature].round()
                            test_df[feature] = test_df[feature].astype('Int64')
                            df[feature].update(test_df[feature])                          
                        else:
                            df[feature].update(test_df[feature])  
                        logger.debug('LINREG imputation of {} value(s) succeeded for feature "{}"', len(pred), feature)
                except:
                    logger.warning('LINREG imputation failed for feature "{}"', feature)
        for feature in df.columns: 
            try:   
                # map categorical feature values back to original
                mappings_inv = {v: k for k, v in mapping[feature].items()}
                df[feature] = df[feature].map(mappings_inv)
            except:
                pass
        return df

    def _log_regression_impute(self, df, model):
        # function for predicting missing values with logistic regression
        cols_num = df.select_dtypes(include=np.number).columns
        mapping = dict()
        for feature in df.columns:
            if feature not in cols_num:
                # create label mapping for categorical feature values
                mappings = {k: i for i, k in enumerate(df[feature])} #.dropna().unique(), 0)}
                mapping[feature] = mappings
                df[feature] = df[feature].map(mapping[feature])

        target_cols = [x for x in df.columns if x not in cols_num]
            
        for feature in df.columns: 
            if feature in target_cols:
                try:
                    test_df = df[df[feature].isnull()==True].dropna(subset=[x for x in df.columns if x != feature])
                    train_df = df[df[feature].isnull()==False].dropna(subset=[x for x in df.columns if x != feature])
                    if len(test_df.index) != 0:
                        pipe = make_pipeline(StandardScaler(), model)

                        y = train_df[feature]
                        train_df.drop(feature, axis=1, inplace=True)
                        test_df.drop(feature, axis=1, inplace=True)

                        model = pipe.fit(train_df, y)
                        
                        pred = model.predict(test_df) # predict values
                        test_df[feature]= pred

                        if (df[feature].fillna(-9999) % 1  == 0).all():
                            # round back to INTs, if original data were INTs
                            test_df[feature] = test_df[feature].round()
                            test_df[feature] = test_df[feature].astype('Int64')
                            df[feature].update(test_df[feature])                             
                        logger.debug('LOGREG imputation of {} value(s) succeeded for feature "{}"', len(pred), feature)
                except:
                    logger.warning('LOGREG imputation failed for feature "{}"', feature)
        for feature in df.columns: 
            try:
                # map categorical feature values back to original
                mappings_inv = {v: k for k, v in mapping[feature].items()}
                df[feature] = df[feature].map(mappings_inv)
            except:
                pass     
        return df

    def _delete(self, df, type):
        # function for deleting missing values
        cols_num = df.select_dtypes(include=np.number).columns 
        if type == 'num':
            # numerical features
            for feature in df.columns: 
                if feature in cols_num:
                    df = df.dropna(subset=[feature])
                    df.reset_index(drop=True)
        else:
            # categorical features
            for feature in df.columns:
                if feature not in cols_num:
                    df = df.dropna(subset=[feature])
                    df.reset_index(drop=True)
        return df                    

class Outliers:

    def handle(self, df):
        # function for handling of outliers in the data
        if self.outliers:
            logger.info('Started handling of outliers... Method: "{}"', str(self.outliers).upper())
            start = timer()  

            if self.outliers in ['auto', 'winz']:  
                df = Outliers._winsorization(self, df)
            elif self.outliers == 'delete':
                df = Outliers._delete(self, df)
            
            end = timer()
            logger.info('Completed handling of outliers in {} seconds', round(end-start, 6))
        else:
            logger.info('Skipped handling of outliers')
        return df     

    def _winsorization(self, df):
        # function for outlier winsorization
        cols_num = df.select_dtypes(include=np.number).columns    
        for feature in cols_num:           
            counter = 0
            # compute outlier bounds
            lower_bound, upper_bound = Outliers._compute_bounds(self, df, feature)    
            for row_index, row_val in enumerate(df[feature]):
                if row_val < lower_bound or row_val > upper_bound:
                    if row_val < lower_bound:
                        if (df[feature].fillna(-9999) % 1  == 0).all():
                                df.loc[row_index, feature] = lower_bound
                                df[feature] = df[feature].astype(int) 
                        else:    
                            df.loc[row_index, feature] = lower_bound
                        counter += 1
                    else:
                        if (df[feature].fillna(-9999) % 1  == 0).all():
                            df.loc[row_index, feature] = upper_bound
                            df[feature] = df[feature].astype(int) 
                        else:
                            df.loc[row_index, feature] = upper_bound
                        counter += 1
            if counter != 0:
                logger.debug('Outlier imputation of {} value(s) succeeded for feature "{}"', counter, feature)        
        return df

    def _delete(self, df):
        # function for deleting outliers in the data
        cols_num = df.select_dtypes(include=np.number).columns    
        for feature in cols_num:
            counter = 0
            lower_bound, upper_bound = Outliers._compute_bounds(self, df, feature)    
            # delete observations containing outliers            
            for row_index, row_val in enumerate(df[feature]):
                if row_val < lower_bound or row_val > upper_bound:
                    df = df.drop(row_index)
                    counter +=1
            df = df.reset_index(drop=True)
            if counter != 0:
                logger.debug('Deletion of {} outliers succeeded for feature "{}"', counter, feature)
        return df

    def _compute_bounds(self, df, feature):
        # function that computes the lower and upper bounds for finding outliers in the data
        featureSorted = sorted(df[feature])
        
        q1, q3 = np.percentile(featureSorted, [25, 75])
        iqr = q3 - q1

        lb = q1 - (self.outlier_param * iqr) 
        ub = q3 + (self.outlier_param * iqr) 

        return lb, ub    

class Adjust:

    def convert_datetime(self, df):
        # function for extracting of datetime values in the data
        if self.extract_datetime:
            logger.info('Started conversion of DATETIME features... Granularity: {}', self.extract_datetime)
            start = timer()
            cols = set(df.columns) ^ set(df.select_dtypes(include=np.number).columns) 
            for feature in cols: 
                try:
                    # convert features encoded as strings to type datetime ['D','M','Y','h','m','s']
                    df[feature] = pd.to_datetime(df[feature], infer_datetime_format=True)
                    try:
                        df['Day'] = pd.to_datetime(df[feature]).dt.day

                        if self.extract_datetime in ['auto', 'M','Y','h','m','s']:
                            df['Month'] = pd.to_datetime(df[feature]).dt.month

                            if self.extract_datetime in ['auto', 'Y','h','m','s']:
                                df['Year'] = pd.to_datetime(df[feature]).dt.year

                                if self.extract_datetime in ['auto', 'h','m','s']:
                                    df['Hour'] = pd.to_datetime(df[feature]).dt.hour

                                    if self.extract_datetime in ['auto', 'm','s']:
                                        df['Minute'] = pd.to_datetime(df[feature]).dt.minute

                                        if self.extract_datetime in ['auto', 's']:
                                            df['Sec'] = pd.to_datetime(df[feature]).dt.second
                        
                        logger.debug('Conversion to DATETIME succeeded for feature "{}"', feature)

                        try: 
                            # check if entries for the extracted dates/times are non-NULL, otherwise drop
                            if (df['Hour'] == 0).all() and (df['Minute'] == 0).all() and (df['Sec'] == 0).all():
                                df.drop('Hour', inplace = True, axis =1 )
                                df.drop('Minute', inplace = True, axis =1 )
                                df.drop('Sec', inplace = True, axis =1 )
                            elif (df['Day'] == 0).all() and (df['Month'] == 0).all() and (df['Year'] == 0).all():
                                df.drop('Day', inplace = True, axis =1 )
                                df.drop('Month', inplace = True, axis =1 )
                                df.drop('Year', inplace = True, axis =1 )   
                        except:
                            pass          
                    except:
                        # feature cannot be converted to datetime
                        logger.warning('Conversion to DATETIME failed for "{}"', feature)
                except:
                    pass
            end = timer()
            logger.info('Completed conversion of DATETIME features in {} seconds', round(end-start, 4))
        else:
            logger.info('Skipped datetime feature conversion')
        return df

    def round_values(self, df, input_data):
        # function that checks datatypes of features and converts them if necessary
        if self.duplicates or self.missing_num or self.missing_categ or self.outliers or self.encode_categ or self.extract_datetime:
            logger.info('Started feature type conversion...')
            start = timer()
            counter = 0
            cols_num = df.select_dtypes(include=np.number).columns
            for feature in cols_num:
                    # check if all values are integers
                    if (df[feature].fillna(-9999) % 1  == 0).all():
                        try:
                            # encode FLOATs with only 0 as decimals to INT
                            df[feature] = df[feature].astype('Int64')
                            counter += 1
                            logger.debug('Conversion to type INT succeeded for feature "{}"', feature)
                        except:
                            logger.warning('Conversion to type INT failed for feature "{}"', feature)
                    else:
                        try:
                            df[feature] = df[feature].astype(float)
                            # round the number of decimals of FLOATs back to original
                            dec = None
                            for value in input_data[feature]:
                                try:
                                    if dec == None:
                                        dec = str(value)[::-1].find('.')
                                    else:
                                        if str(value)[::-1].find('.') > dec:
                                            dec = str(value)[::-1].find('.')
                                except:
                                    pass
                            df[feature] = df[feature].round(decimals = dec)
                            counter += 1
                            logger.debug('Conversion to type FLOAT succeeded for feature "{}"', feature)
                        except:
                            logger.warning('Conversion to type FLOAT failed for feature "{}"', feature)
            end = timer()
            logger.info('Completed feature type conversion for {} feature(s) in {} seconds', counter, round(end-start, 6))
        else:
            logger.info('Skipped feature type conversion')
        return df

class EncodeCateg:

    def handle(self, df):
        # function for encoding of categorical features in the data
        if self.encode_categ:
            if not isinstance(self.encode_categ, list):
                self.encode_categ = ['auto']
            # select non numeric features
            cols_categ = set(df.columns) ^ set(df.select_dtypes(include=np.number).columns) 
            # check if all columns should be encoded
            if len(self.encode_categ) == 1:
                target_cols = cols_categ # encode ALL columns
            else:
                target_cols = self.encode_categ[1] # encode only specific columns
            logger.info('Started encoding categorical features... Method: "{}"', str(self.encode_categ[0]).upper())
            start = timer()
            for feature in target_cols:
                if feature in cols_categ:
                    # columns are column names
                    feature = feature
                else:
                    # columns are indexes
                    feature = df.columns[feature]
                try:
                    # skip encoding of datetime features
                    pd.to_datetime(df[feature])
                    logger.debug('Skipped encoding for DATETIME feature "{}"', feature)
                except:
                    try:
                        if self.encode_categ[0] == 'auto':
                            # ONEHOT encode if not more than 10 unique values to encode
                            if df[feature].nunique() <=10:
                                df = EncodeCateg._to_onehot(self, df, feature)
                                logger.debug('Encoding to ONEHOT succeeded for feature "{}"', feature)
                            # LABEL encode if not more than 20 unique values to encode
                            elif df[feature].nunique() <=20:
                                df = EncodeCateg._to_label(self, df, feature)
                                logger.debug('Encoding to LABEL succeeded for feature "{}"', feature)
                            # skip encoding if more than 20 unique values to encode
                            else:
                                logger.debug('Encoding skipped for feature "{}"', feature)   

                        elif self.encode_categ[0] == 'onehot':
                            df = EncodeCateg._to_onehot(df, feature)
                            logger.debug('Encoding to {} succeeded for feature "{}"', str(self.encode_categ[0]).upper(), feature)
                        elif self.encode_categ[0] == 'label':
                            df = EncodeCateg._to_label(df, feature)
                            logger.debug('Encoding to {} succeeded for feature "{}"', str(self.encode_categ[0]).upper(), feature)      
                    except:
                        logger.warning('Encoding to {} failed for feature "{}"', str(self.encode_categ[0]).upper(), feature)    
            end = timer()
            logger.info('Completed encoding of categorical features in {} seconds', round(end-start, 6))
        else:
            logger.info('Skipped encoding of categorical features')
        return df

    def _to_onehot(self, df, feature, limit=10):  
        # function that encodes categorical features to OneHot encodings    
        one_hot = pd.get_dummies(df[feature], prefix=feature)
        if one_hot.shape[1] > limit:
            logger.warning('ONEHOT encoding for feature "{}" creates {} new features. Consider LABEL encoding instead.', feature, one_hot.shape[1])
        # join the encoded df
        df = df.join(one_hot)
        return df

    def _to_label(self, df, feature):
        # function that encodes categorical features to label encodings 
        le = preprocessing.LabelEncoder()

        df[feature + '_lab'] = le.fit_transform(df[feature].values)
        mapping = dict(zip(le.classes_, range(len(le.classes_))))
        
        for key in mapping:
            try:
                if isnan(key):               
                    replace = {mapping[key] : key }
                    df[feature].replace(replace, inplace=True)
            except:
                pass
        return df  

class Duplicates:

    def handle(self, df):
        if self.duplicates:
            logger.info('Started handling of duplicates... Method: "{}"', str(self.duplicates).upper())
            start = timer()
            original = df.shape
            try:
                df.drop_duplicates(inplace=True, ignore_index=False)
                df = df.reset_index(drop=True)
                new = df.shape
                count = original[0] - new[0]
                if count != 0:
                    logger.debug('Deletion of {} duplicate(s) succeeded', count)
                else:
                    logger.debug('{} missing values found', count)
                end = timer()
                logger.info('Completed handling of duplicates in {} seconds', round(end-start, 6))

            except:
                logger.warning('Handling of duplicates failed')        
        else:
            logger.info('Skipped handling of duplicates')
        return df 