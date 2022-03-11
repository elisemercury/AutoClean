import numpy as np
import pandas as pd
from math import isnan
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn import preprocessing

from loguru import logger

class Modules:

    def _check_missings_num(self, df, _n_neighbors=3):
        # function for handling numerical missing values in the data
        if self.missings_num:
            logger.info('Started handling of missing values (numerical)... Method: "{}"', self.missings_num.upper())
            count_missings = df.isna().sum().sum()

            if count_missings != 0:
                logger.info('Found a total of {} missing value(s)', count_missings)

                # knn imputation (default)
                if self.missings_num == 'knn':
                    imputer = KNNImputer(n_neighbors=_n_neighbors)
                # mean imputation
                elif self.missings_num == 'mean':
                    imputer = SimpleImputer(strategy='mean')
                # median imputation
                elif self.missings_num == 'median':
                    imputer = SimpleImputer(strategy="median")
                # mode imputation
                elif self.missings_num == "mode":     
                    imputer = SimpleImputer(strategy="most_frequent")  

                # delete missing values
                elif self.missings_num == 'delete':
                    df = df.dropna()
                    df = df.reset_index(drop=True)
                    logger.debug('Deletion of {} missing value(s) succeeded', count_missings)
                    return df             

                # empty for future use
                else:
                    pass

                if imputer:
                    # impute only for numeric features
                    cols = df.select_dtypes(include=np.number).columns  
                    for col_name in cols: 
                            try:
                                imputed = pd.DataFrame(imputer.fit_transform(np.array(df[col_name]).reshape(-1, 1)), columns=[col_name])
                                counter = sum(1 for i, j in zip(list(imputed[col_name]), list(df[col_name])) if i != j)
                                if (df[col_name].fillna(-9999) % 1  == 0).all():
                                    df[col_name] = imputed
                                    df[col_name] = df[col_name].astype(int)                                        
                                else:
                                    df[col_name] = imputed
                                if counter != 0:
                                    logger.debug('{} imputation of {} value(s) succeeded for feature "{}"', self.missings_num.upper(), counter, col_name)
                            except:
                                logger.debug('{} imputation failed for feature "{}"', self.missings_num.upper(), col_name)
                        
            else:
                logger.debug('{} missing values found.', count_missings)

            logger.info('Completed handling of missing values (numerical)')
        return df

    def _round_values(self, df, _decimals=2):
        # function that checks datatypes of features and converts them if necessary
        logger.info('Started feature type conversion...')
        counter = 0
        cols = df.select_dtypes(include=np.number).columns
        for col_name in cols:

                # check if all values are integers
                if (df[col_name].fillna(-9999) % 1  == 0).all():
                    try:
                        # encode FLOATs with only 0 as decimals to INT
                        df[col_name] = df[col_name].astype(int)
                        counter += 1
                        logger.debug('Conversion to type INT succeeded for feature "{}"', col_name)
                    except:
                        logger.debug('Conversion to type INT failed for feature "{}"', col_name)
                else:
                    try:
                        # round FLOATs to 4 decimals
                        df[col_name] = df[col_name].round(decimals=_decimals)
                        counter += 1
                        logger.debug('Conversion to type FLOAT succeeded for feature "{}"', col_name)
                    except:
                        logger.debug('Conversion to type FLOAT failed for feature "{}"', col_name)

        logger.info('Completed feature type conversion for {} feature(s)', counter)
        return df

    def _check_missings_categ(self, df):
        # function for handling categorical missing values in the data
        if self.missings_categ:
            logger.info('Started handling of missing values (categorical)... Method: "{}"', self.missings_categ.upper())
            count_missings = df.isna().sum().sum()

            if count_missings != 0:
                logger.info('Found a total of {} missing value(s)', count_missings)
                
                # mode imputation (default)
                if self.missings_categ == 'mode':
                    imputer = SimpleImputer(strategy='most_frequent')
                    cols = set(df.columns) ^ set(df.select_dtypes(include=np.number).columns)   
                    for col_name in cols:
                            imputed = pd.DataFrame(imputer.fit_transform(np.array(df[col_name]).reshape(-1, 1)), columns=[col_name])
                            counter = sum(1 for i, j in zip(imputed[col_name], df[col_name]) if i != j)
                            df[col_name] = imputed
                            if counter != 0:
                                logger.debug('{} imputation of {} value(s) succeeded for feature "{}"', self.missings_categ.upper(), counter, col_name)
                
                # delete missing values
                elif self.missings_categ == 'delete':
                    df = df.dropna()
                    df = df.reset_index(drop=True)
                    logger.debug('Deletion of {} missing values succeeded', count_missings)
                    return df

                # empty for future use
                else:  
                    pass

            logger.info('Completed handling of missing values (categorical)')
        return df      

    def _check_outliers(self, df):
        #defines obersvations as outliers if they are outside of range [Q1-1.5*IQR ; Q3+1.5*IQR] whereas IQR is the interquartile range.
        if self.outliers:
            logger.info('Started handling of outliers... Method: "{}"', self.outliers.upper())
            cols = df.select_dtypes(include=np.number).columns    

            for col_name in cols:           
                counter = 0
                # compute outlier bounds
                lower_bound, upper_bound = Modules._compute_bounds(self, df, col_name)     

                # replace outliers by bounds
                if self.outliers == 'winz':     
                    for row_index, row_val in enumerate(df[col_name]):
                        if row_val < lower_bound or row_val > upper_bound:
                            if row_val < lower_bound:
                                if (df[col_name].fillna(-9999) % 1  == 0).all():
                                        df.loc[row_index, col_name] = lower_bound
                                        df[col_name] = df[col_name].astype(int) 
                                else:    
                                    df.loc[row_index, col_name] = lower_bound
                                counter += 1
                            else:
                                if (df[col_name].fillna(-9999) % 1  == 0).all():
                                    df.loc[row_index, col_name] = upper_bound
                                    df[col_name] = df[col_name].astype(int) 
                                else:
                                    df.loc[row_index, col_name] = upper_bound
                                counter += 1
                    if counter != 0:
                        logger.debug('Outlier {} imputation of {} value(s) succeeded for feature "{}"', self.outliers.upper(), counter, col_name)

                # delete observations containing outliers            
                elif self.outliers == 'delete':
                    for row_index, row_val in enumerate(df[col_name]):
                        if row_val < lower_bound or row_val > upper_bound:
                            df = df.drop(row_index)
                            counter +=1
                    df = df.reset_index(drop=True)
                    if counter != 0:
                        logger.debug('Deletion of {} outliers succeeded for feature "{}"', counter, col_name)
                
                # empty for future use
                else:
                    pass

            logger.info('Completed handling of outliers')
        return df     

    def _convert_datetime(self, df):
        if self.extract_datetime:
            logger.info('Started conversion of DATETIME features... Granularity: {}', self.extract_datetime)
            cols = set(df.columns) ^ set(df.select_dtypes(include=np.number).columns) 

            for col_name in cols: 
                try:
                    # convert features encoded as strings to type datetime ['D','M','Y','h','m','s']
                    df[col_name] = pd.to_datetime(df[col_name])

                    df['Day'] = pd.to_datetime(df[col_name]).dt.day

                    if self.extract_datetime in ['M','Y','h','m','s']:
                        df['Month'] = pd.to_datetime(df[col_name]).dt.month

                        if self.extract_datetime in ['Y','h','m','s']:
                            df['Year'] = pd.to_datetime(df[col_name]).dt.year

                            if self.extract_datetime in ['h','m','s']:
                                df['Hour'] = pd.to_datetime(df[col_name]).dt.hour

                                if self.extract_datetime in ['m','s']:
                                    df['Minute'] = pd.to_datetime(df[col_name]).dt.minute

                                    if self.extract_datetime in ['s']:
                                        df['Sec'] = pd.to_datetime(df[col_name]).dt.second
                    
                    logger.debug('Conversion to DATETIME succeeded for feature "{}"', col_name)

                    try: 
                        # check if entries for the extracted dates/times are valid, otherwise drop
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
                    logger.debug('Conversion to DATETIME failed for "{}"', col_name) 

            logger.info('Completed conversion of DATETIME features')
        return df

    def _encode_categ(self, df):
        if self.encode_categ[0]:
            cols = set(df.columns) ^ set(df.select_dtypes(include=np.number).columns) 
            
            # automated checking for optimal encoding
            if self.encode_categ[0] == "auto":
                logger.info('Started encoding categorical features... Method: "AUTO"')
                for col_name in cols:
                    try:
                        # skip encoding of datetime features
                        pd.to_datetime(df[col_name])
                        logger.debug('Skipped encoding for DATETIME feature "{}"', col_name)
                    except:
                        # ONEHOT encode if not more than 10 unique values to encode
                        if df[col_name].nunique() <=10:
                            # skip encoding if encoding leads more features than observations
                            #if int(df.shape[0]) < (int(df[cols[col_num]].nunique()) + int(df.shape[0])):
                            #    logger.debug('Encoding to {} skipped for feature "{}"', self.encode_categ[0].upper(), cols[col_num])
                            #else:
                            df = Modules._onehot_encode(self, df, col_name)
                            logger.debug('Encoding to ONEHOT succeeded for feature "{}"', col_name)

                        # LABEL encode if not more than 20 unique values to encode
                        elif df[col_name].nunique() <=20:
                            df = Modules._label_encode(self, df, col_name)
                            logger.debug('Encoding to LABEL succeeded for feature "{}"', col_name)

                        # skip encoding if more than 20 unique values to encode
                        else:
                            logger.debug('Encoding skipped for feature "{}"', col_name)        
                
            # check if only specific columns should be encoded
            elif len(self.encode_categ) == 2:
                logger.info('Started encoding categorical features... Method: "{}" on features "{}"', self.encode_categ[0], self.encode_categ[1])
                for i in self.encode_categ[1]:
                    # check if given columns are column names
                    if i in cols:
                        try:
                            if self.encode_categ[0] == 'onehot':
                                df = Modules._onehot_encode(self, df, i)
                                logger.debug('Encoding to {} succeeded for feature "{}"', self.encode_categ[0].upper(), i)
                            elif self.encode_categ[0] == 'label':
                                df = Modules._label_encode(self, df, i)
                                logger.debug('Encoding to {} succeeded for feature "{}"', self.encode_categ[0].upper(), i)
                            else:
                                pass
                        except:
                            logger.debug('Encoding to {} failed for feature "{}"', self.encode_categ[0].upper(), i)

                    # check given columns are column indexes
                    else:
                        try:
                            if self.encode_categ[0] == 'onehot':
                                df = self._onehot_encode(df, cols[i])
                                logger.debug('Encoding to {} succeeded for feature "{}"', self.encode_categ[0].upper(), cols[i])
                            elif self.encode_categ[0] == 'label':
                                df = self._label_encode(df, cols[i])
                                logger.debug('Encoding to {} succeeded for feature "{}"', self.encode_categ[0].upper(), cols[i])
                            else:
                                pass
                        except:
                            logger.debug('Encoding to {} failed for feature "{}"', self.encode_categ[0].upper(), i)

            # encode all columns
            else:
                logger.info('Started encoding categorical features... Method: "{}"', self.encode_categ[0], self.encode_categ[1])
                for col_name in cols:
                    try:
                        # skip encoding of datetime features
                        pd.to_datetime(df[col_name])
                        logger.debug('Skipped encoding for DATETIME feature "{}"', col_name)
                    except:
                        try:
                            if self.encode_categ[0] == 'onehot':
                                df = self._onehot_encode(df, col_name)
                                logger.debug('Encoding to {} succeeded for feature "{}"', self.encode_categ[0].upper(), col_name)
                            elif self.encode_categ[0] == 'label':
                                df = self._label_encode(df, col_name)
                                logger.debug('Encoding to {} succeeded for feature "{}"', self.encode_categ[0].upper(), col_name)
                            else:
                                pass                              
                        except:
                            logger.debug('Encoding to {} failed for feature "{}"', self.encode_categ[0].upper(), col_name)

            logger.info('Completed encoding of categorical features')
        return df

    def _onehot_encode(self, df, col, limit=15):        
        one_hot = pd.get_dummies(df[col], prefix=col)
        if one_hot.shape[1] > limit:
            logger.warning('ONEHOT encoding for feature "{}" creates {} new features. Consider LABEL encoding instead.', col, one_hot.shape[1])
        # join the encoded df
        df = df.join(one_hot)
        return df

    def _label_encode(self, df, col):
        le = preprocessing.LabelEncoder()
        
        df[col] = le.fit_transform(df[col].values)
        mapping = dict(zip(le.classes_, range(len(le.classes_))))
        
        for key in mapping:
            try:
                if isnan(key):               
                    replace = {mapping[key] : key }
                    df[col].replace(replace, inplace=True)
            except:
                pass
        return df

    def _compute_bounds(self, df, col):
        colSorted = sorted(df[col])
        
        q1, q3 = np.percentile(colSorted, [25, 75])
        iqr = q3 - q1

        lb = q1 - (self.outlier_param * iqr) 
        ub = q3 + (self.outlier_param * iqr) 

        return lb, ub
