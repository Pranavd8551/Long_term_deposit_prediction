import sys

from pandas import DataFrame
from sklearn.pipeline import Pipeline

from long_term_deposit_prediction.exception import DepositException
from long_term_deposit_prediction.logger import logging



class TargetValueMapping:
    def __init__(self):
        self.yes:int = 0
        self.no:int = 1
    def _asdict(self):
        return self.__dict__
    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(),mapping_response.keys()))
    



    