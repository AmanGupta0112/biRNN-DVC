from src.utils import save_bin
import time
from src import logging
from src.constants import *
from src.components import DataIngestionPreparation
from src.components import ModelTraining
from src import logging 


STAGE = "Stage name" ## <<< change stage name 

# init logger
# logger()



def main():
    data_ing_and_prep = DataIngestionPreparation()
    data_ing_and_prep.load_dataset()
    train_ds , test_ds = data_ing_and_prep.shuffle_and_batch()
    encoder = data_ing_and_prep.encode_on_train_data()
    model = ModelTraining()
    model.layers_defined(encoder)
    tb_cb,ckpt_cb = model.callbacks()
    model.model_compilation()
    history = model.model_training([tb_cb, ckpt_cb], train_ds, test_ds)
    model.model_evaluation()
    model.model_saving()
    sample_texts = [
        "The movie was cool. The animation and the graphics were out of the world. I would definitly recommend this movie.",
        "The movie was horrible. The animation and the graphics were terrible. I would never recommend this movie.",
        "Good day. I would recommend to watch a bad movie .",
        "Ok day. I would recommend an normal movie ."
    ]
    model.model_predict(sample_texts)



    
    # data_ing_and_prep.save_artifacts()
    

if __name__ == '__main__':
    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main()
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e