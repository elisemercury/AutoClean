import sys
import pandas as pd
from Modules import Modules

from loguru import logger

class AutoClean:

    def __init__(self, input_data, missings_num='knn', missings_categ='mode', encode_categ=['auto'], extract_datetime='s', outliers='winz', outlier_param=1.5, logfile=True, verbose=False):  
        '''
        input_data (dataframe)..........Pandas dataframe
        missings_num (str)..............define how NUMERIC missing values are handled
                                        'knn' = uses K-NN algorithm for missing value imputation
                                        'mean','median' or 'mode' = uses mean/median/mode imputatiom
                                        'delete' = deletes observations with missing values
        missings_categ (str)............define how CATEGORICAL missing values are handled
                                        'mode' = mode imputatiom
                                        'delete' = deletes observations with missing values
        encode_categ (list).............encode categorical features, takes a list as input
                                        ['onehot'] = one-hot-encode all categorical features
                                        ['label'] = label-encode all categ. features
                                        to encode only specific features add the column name or number: ['onehot', ['col1', 'col3']]
        extract_datetime (str)..........define whether datetime features should be extracted into separate features
                                        to define granularity set to 'D' = day, 'M' = month, 'Y' = year, 'h' = hour, 'm' = minute or 's' = second
        outliers (str)..................define how outliers are handled
                                        'winz' = replaces outliers through winzoring
                                        'delete' = deletes observations containing outliers
                                        oberservations are considered outliers if they are outside the lower and upper bounds
                                        of [Q1-1.5*IQR, Q3+1.5*IQR], whereas IQR is the interquartile range.
                                        to set a custom multiplier use the 'outlier_param' parameter
        outlier_param (int, float)......! recommended not to change default value
                                        define the multiplier for the outlier bounds
        logfile (bool)..................define whether to create a logile during the autoclean process
                                        logfile will be saved in working directory as "autoclean.log"
        verbose (bool)..................define whether  autoclean logs will be printed in console
        
        OUTPUT (dataframe)..............a cleaned Pandas dataframe, accessible through the 'output_data' instance
        '''    
        AutoClean._initialize_logger(self, verbose, logfile)
        
        output_data = input_data.copy()
    
        self.missings_num = missings_num
        self.missings_categ = missings_categ
        self.outliers = outliers
        self.encode_categ = encode_categ
        self.outlier_param = outlier_param
        self.extract_datetime = extract_datetime

        # validate the input parameters
        AutoClean._validate_params(self, output_data, verbose, logfile)
        
        # initialize our class and start the autoclean process
        self.output_data = AutoClean._clean_data(self, output_data)
        
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
        if self.missings_num != False and str(self.missings_num) not in ['knn', 'mean', 'median', 'mode', 'delete']:
            raise ValueError('Invalid value for "missings_num" parameter.')
        if self.missings_categ != False and str(self.missings_categ) not in ['mode', 'delete']:
            raise ValueError('Invalid value for "missings_categ" parameter.')
        if self.outliers != False and str(self.outliers) not in ['winz', 'delete']:
            raise ValueError('Invalid value for "outliers" parameter.')
        if len(self.encode_categ) > 2 and not isinstance(self.encode_categ, list) and self.encode_categ[0] != False and self.encode_categ[0] not in ['auto', 'onehot', 'label']:
            raise ValueError('Invalid value for "encode_categ" parameter.')
        if len(self.encode_categ) == 2:
            if not isinstance(self.encode_categ[1], list):
                raise ValueError('Invalid value for "encode_categ" parameter.')
        if not isinstance(self.outlier_param, int) and not isinstance(self.outlier_param, float):# and [i for i in self.outlier_param if not int(i) or not float(i)]:
            raise ValueError('Invalid value for "outlier_param" parameter.')  
        if self.extract_datetime != False and self.extract_datetime not in ['D','M','Y','h','m','s']:
            raise ValueError('Invalid value for "extract_datetime" parameter.')  

        logger.info('Completed validation of input parameters')

        return
            
    def _clean_data(self, df):
        # function for starting the autoclean process
        if self.missings_categ != "delete" and self.missings_num == "delete":
            df = Modules._check_missings_categ(self, df)
            df = Modules._check_missings_num(self, df)
        else:
            df = Modules._check_missings_num(self, df)
            df = Modules._check_missings_categ(self, df)
        
        df = Modules._check_outliers(self, df)
        
        df = Modules._convert_datetime(self, df) 
        df = Modules._encode_categ(self, df)
        
        df = Modules._round_values(self,df)

        logger.info('AutoClean completed successfully')
        return df