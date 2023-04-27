from src.utils import save_bin
import time
from src import logging
from src.constants import *
from src.components import DataIngestionPreparation
from src import logging 


STAGE = "Stage name" ## <<< change stage name 

# init logger
# logger()



def main():
    data_ing_and_prep = DataIngestionPreparation()
    data_ing_and_prep.load_dataset()
    data_ing_and_prep.shuffle_and_batch()
    data_ing_and_prep.encode_on_train_data()
    data_ing_and_prep.save_artifacts()
    

if __name__ == '__main__':
    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main()
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e