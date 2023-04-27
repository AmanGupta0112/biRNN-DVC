import tensorflow as tf
from src.constants import *
from src.utils import create_directories,save_json
from src import logging
import re
import os


class ModelTraining:
    def __init__(self):
        self.encoder = None
        self.LAYERS = None
        self.model = None
        self.tb_cb = None
        self.ckpt_cb = None

    def layers_defined(self,encoder):
        self.encoder = encoder
        embedding_layer = tf.keras.layers.Embedding(
            input_dim=len(self.encoder.get_vocabulary()),
            output_dim=OUTPUT_DIM,
            mask_zero=True
        )
        self.LAYERS = [
            self.encoder,
            embedding_layer,
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(64)
            ),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1)
        ]
        logging.info("Layers defined successfully!!")


    def callbacks(self):
        unique_log = re.sub(r"[\s:]", "_", time.asctime())
        create_directories([TB_ROOT_LOG_DIR])
        tb_log_dir = os.path.join(TB_ROOT_LOG_DIR,unique_log)
        self.tb_cb = tf.keras.callbacks.TensorBoard(log_dir=tb_log_dir)
        create_directories([f"{ARTIFACTS_DIR_PATH}/{CHECKPOINT_DIR}"])
        ckpt_file = os.path.join(f"{ARTIFACTS_DIR_PATH}/{CHECKPOINT_DIR}")
        self.ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_file,
            save_best_only=True
        )
        logging.info("Callbacks defined successfully!!")
        return [self.tb_cb, self.ckpt_cb]


    def model_compilation(self):
        self.model = tf.keras.Sequential(self.LAYERS)
        self.model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            metrics=['accuracy']
        )
        logging.info("Model compiled successfully!!")


    def model_training(self,callback_list, train_ds, test_ds):
        self.test_ds = test_ds
        self.train_ds = train_ds
        self.callback_list = callback_list

        logging.info("Model training started!!")
        self.history = self.model.fit(train_ds,
                                epochs=TRAINING_EPOCHS,
                                validation_data=test_ds,
                                validation_steps=30,
                                callbacks=self.callback_list)
        logging.info("Model trained successfully!!")
        return self.history

    def model_evaluation(self):
        test_loss, test_acc = self.model.evaluate(self.test_ds)
        logging.info(f"==================== >>>>>>> \nModel Accuracy : {test_acc} with loss : {test_loss}!!")

    def model_saving(self):
        model_dir = f"{ARTIFACTS_DIR_PATH}/{MODEL_DIR_PATH}"
        create_directories([model_dir])
        filepath = os.path.join(model_dir, f"{MODEL_NAME}.h5")
        self.model.save(filepath)
        logging.info("Model saved successfully!!")

    def model_predict(self,sample_texts):
        self.predicted_values = {}
        model = self.model.load_model(f"{ARTIFACTS_DIR_PATH}/{MODEL_DIR_PATH}/{MODEL_NAME}.h5")
        for sample_text in sample_texts:
            prediction = model.predict([sample_text])[0][0]
            
            self.predicted_values[sample_text] = prediction
            logging.info(
                f"Model sentiment analysis '{sample_text}' : {prediction}")
        create_directories([f"{ARTIFACTS_DIR_PATH}/{PREDICTION_DIR}"])
        save_json(f"{ARTIFACTS_DIR_PATH}/{PREDICTION_DIR}",
                  self.predicted_values)
