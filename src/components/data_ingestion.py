import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from src import logging
from src.constants import *
import joblib
from src.utils import save_bin,uniquename
import os


class DataIngestionPreparation:
    def __init__(self):
        self.dataset_name = "imdb_reviews"

    def load_dataset(self):
        dataset, info = tfds.load(
            self.dataset_name, with_info=True, as_supervised=True)
        self.train_ds, self.test_ds = dataset['train'], dataset['test']
        logging.info(f"{self.dataset_name} dataset downloaded with info: \n{info}")

    def shuffle_and_batch(self):
        self.train_ds = self.train_ds.shuffle(TRAINING_BUFFER_SIZE).batch(
                    TRAINING_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        self.test_ds = self.test_ds.batch(TRAINING_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        logging.info(f"Shuffle and Batched datasets!")

    def encode_on_train_data(self):
        self.encoder = tf.keras.layers.TextVectorization(
            max_tokens=VOCAB_SIZE)
        self.encoder.adapt(self.train_ds.map(lambda text, label: text))
        logging.info(f"Encoding is done!")
    
    def save_artifacts(self):
        self._save_encode()
        self._save_train_test_data()
    
    def _save_encode(self):
        file_name = uniquename()
        path = f"{ARTIFACTS_DIR_PATH}/{ENCODER_DIR_PATH}"
        create_directories([path])
        path = f"{path}/{file_name}_encoder.bin"
        save_bin(data=self.encoder, path=f"{path}")
        logging.info(f"Encoder is saved!")

    def _save_train_test_data(self):
        path = f"{ARTIFACTS_DIR_PATH}/{DATASET_DIR_PATH}"
        create_directories([path])
        file_name1 = uniquename()
        writer = tf.data.experimental.TFRecordWriter(file_name1)
        path = f"{path}/{file_name1}_test_ds.bin"
        writer.write(self.test_ds.map(tf.io.serialize_tensor))
        # save_bin(data=self.test_ds, path=f"{path}")
        file_name2 = uniquename()
        writer = tf.data.experimental.TFRecordWriter(file_name2)
        path = f"{path}/{file_name2}_train_ds.bin"
        writer.write(self.train_ds.map(tf.io.serialize_tensor))
        # save_bin(data=self.train_ds, path=f"{path}")
        logging.info(f"Train_ds and Test_ds is saved!")
