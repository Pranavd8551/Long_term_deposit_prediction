from long_term_deposit_prediction.exception import DepositException
import sys
try:
    a=2/0
except Exception as e:
    raise DepositException(e,sys)