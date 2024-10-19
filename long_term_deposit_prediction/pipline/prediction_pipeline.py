import os
import sys

import numpy as np
import pandas as pd
from long_term_deposit_prediction.entity.config_entity import DepositPredictorConfig
from long_term_deposit_prediction.entity.s3_estimator import DepositEstimator
from long_term_deposit_prediction.exception import DepositException
from long_term_deposit_prediction.logger import logging
from long_term_deposit_prediction.utils.main_utils import read_yaml_file
from pandas import DataFrame


class DepositData:
    def __init__(self,
                age,
                job,
                marital,
                education,
                contact,
                month,
                duration,
                campaign,
                poutcome,
                emp_var_rate,
                cons_conf_idx
                ):
        """
        Deposit Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.age = age
            self.job = job
            self.marital = marital
            self.education = education
            self.contact = contact
            self.month = month
            self.duration = duration
            self.campaign = campaign
            self.poutcome = poutcome
            self.emp_var_rate = emp_var_rate
            self.cons_conf_idx = cons_conf_idx


        except Exception as e:
            raise DepositException(e, sys) from e

    def get_deposit_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from DepositData class input
        """
        try:
            
            deposit_input_dict = self.get_deposit_data_as_dict()
            return DataFrame(deposit_input_dict)
        
        except Exception as e:
            raise DepositException(e, sys) from e


    def get_deposit_data_as_dict(self):
        """
        This function returns a dictionary from DepositData class input 
        """
        logging.info("Entered get_deposit_data_as_dict method as DepositData class")

        try:
            input_data = {
                "age": [self.age],
                "job": [self.job],
                "marital": [self.marital],
                "education": [self.education],
                "contact": [self.contact],
                "month": [self.month],
                "duration": [self.duration],
                "campaign": [self.campaign],
                "poutcome": [self.poutcome],
                "emp_var_rate": [self.emp_var_rate],
                "cons_conf_idx": [self.cons_conf_idx],
            }

            logging.info("Created deposit data dict")

            logging.info("Exited get_deposit_data_as_dict method as DepositData class")

            return input_data

        except Exception as e:
            raise DepositException(e, sys) from e

class DepositClassifier:
    def __init__(self,prediction_pipeline_config: DepositPredictorConfig = DepositPredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            # self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise DepositException(e, sys)


    def predict(self, dataframe) -> str:
        """
        This is the method of DepositClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of DepositClassifier class")
            model = DepositEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result =  model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise DepositException(e, sys)