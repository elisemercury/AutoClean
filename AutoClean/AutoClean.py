import sys
import pandas as pd
from Modules import *

from loguru import logger

class AutoClean:

    def __init__(self, input_data, missing_num='auto', missing_categ='auto', encode_categ=['auto'], extract_datetime='s', outliers='winz', outlier_param=1.5, logfile=True, verbose=False):  
        '''
        input_data (dataframe)..........Pandas dataframe
        missing_num (str)...............define how NUMERICAL missing values are handled
                                        'auto' = automated handling
                                        'knn' = uses K-NN algorithm for missing value imputation
                                        'mean','median' or 'most_frequent' = uses mean/median/mode imputatiom
                                        'delete' = deletes observations with missing values
        missing_categ (str).............define how CATEGORICAL missing values are handled
        encode_categ (list).............encode CATEGORICAL features, takes a list as input

                                        ['onehot'] = one-hot-encode all CATEGORICAL features
                                        ['label'] = label-encode all categ. features
                                        to encode only specific features add the column name or number: ['onehot', ['col1', 'col3']]
        extract_datetime (str)..........define whether DATETIME type features should be extracted into separate features
                                        to define granularity set to 'D'= day, 'M'= month, 'Y'= year, 'h'= hour, 'm'= minute or 's'= second
        outliers (str)..................define how outliers are handled
                                        'winz' = replaces outliers through winzoring
                                        'delete' = deletes observations containing outliers
                                        oberservations are considered outliers if they are outside the lower and upper bound [Q1-1.5*IQR, Q3+1.5*IQR], where IQR is the interquartile range
                                        to set a custom multiplier use the 'outlier_param' parameter
        outlier_param (int, float)......! recommended not to change default value
                                        define the multiplier for the outlier bounds
        logfile (bool)..................define whether to create a logile during the AutoClean process
                                        logfile will be saved in working directory as "autoclean.log"
        verbose (bool)..................define whether  AutoClean logs will be printed in console
        
        OUTPUT (dataframe)..............a cleaned Pandas dataframe, accessible through the 'output_data' instance
        '''    
        AutoClean._initialize_logger(self, verbose, logfile)
        
        output_data = input_data.copy()
    
        self.missing_num = missing_num
        self.missing_categ = missing_categ
        self.outliers = outliers
        self.encode_categ = encode_categ
        self.extract_datetime = extract_datetime
        self.outlier_param = outlier_param
        
        # validate the input parameters
        AutoClean._validate_params(self, output_data, verbose, logfile)
        
        # initialize our class and start the autoclean process
        self.output_data = AutoClean._clean_data(self, output_data, input_data)

    def _initialize_logger(self, verbose, logfile):
        # function for initializing the logging process
        logger.remove()
        if verbose == True:
            logger.add(sys.stderr, format='{time:DD-MM-YYYY HH:mm:ss.SS} - {level} - {message}')
        if logfile == True:    
            logger.add('autoclean.log', mode='w', format='{time:DD-MM-YYYY HH:mm:ss.SS} - {level} - {message}')

        return

    def _validate_params(self, df, verbose, logfile):
        # function for validating the input parameters of the autolean process
        logger.info('Started validation of input parameters...')
        
        if type(df) != pd.core.frame.DataFrame:
            raise ValueError('Invalid value for "df" parameter.')
        if self.missing_num not in [False, 'auto', 'knn', 'mean', 'median', 'most_frequent', 'delete']:
            raise ValueError('Invalid value for "missing_num" parameter.')
        if self.missing_categ not in [False, 'auto', 'knn', 'most_frequent', 'delete']:
            raise ValueError('Invalid value for "missing_categ" parameter.')
        if self.outliers not in [False, 'winz', 'delete']:
            raise ValueError('Invalid value for "outliers" parameter.')
        if len(self.encode_categ) > 2 and not isinstance(self.encode_categ, list) and self.encode_categ[0] not in [False, 'auto', 'onehot', 'label']:
            raise ValueError('Invalid value for "encode_categ" parameter.')
        if len(self.encode_categ) == 2:
            if not isinstance(self.encode_categ[1], list):
                raise ValueError('Invalid value for "encode_categ" parameter.')
        if not isinstance(self.outlier_param, int) and not isinstance(self.outlier_param, float):
            raise ValueError('Invalid value for "outlier_param" parameter.')  
        if self.extract_datetime not in [False, 'D','M','Y','h','m','s']:
            raise ValueError('Invalid value for "extract_datetime" parameter.')  
        if not isinstance(verbose, bool):
            raise ValueError('Invalid value for "verbose" parameter.')  
        if not isinstance(logfile, bool):
            raise ValueError('Invalid value for "logfile" parameter.')  

        logger.info('Completed validation of input parameters')

        return
            
    def _clean_data(self, df, input_data):
        # function for starting the autoclean process
        df = MissingValues.handle(self, df)
        df = Outliers.handle(self, df)
        
        df = Adjust.convert_datetime(self, df) 
        df = EncodeCateg.handle(self, df)
        
        df = Adjust.round_values(self, df, input_data)

        logger.info('AutoClean completed successfully')
        return df        