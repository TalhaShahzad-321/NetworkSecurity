from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact
import os
import sys
import numpy as np
import pandas as pd
import pymongo
from typing import List
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

MONGO_DB_URL = "mongodb+srv://talha_shahzad:Admin1234@cluster0.vasnc3w.mongodb.net/?appName=Cluster0"


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config

            # ==== DEBUG: CONFIG INFO ====
            logging.info("Initialized DataIngestion with config:")
            logging.info(f"  database_name: {self.data_ingestion_config.database_name}")
            logging.info(f"  collection_name: {self.data_ingestion_config.collection_name}")
            logging.info(f"  feature_store_file_path: {self.data_ingestion_config.feature_store_file_path}")
            logging.info(f"  training_file_path: {self.data_ingestion_config.training_file_path}")
            logging.info(f"  testing_file_path: {self.data_ingestion_config.testing_file_path}")
            logging.info(f"  train_test_split_ratio: {self.data_ingestion_config.train_test_split_ratio}")

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """
        Read data from mongodb and return as DataFrame
        """
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name

            logging.info("Connecting to MongoDB...")
            logging.info(f"MONGO_DB_URL: {MONGO_DB_URL}")
            logging.info(f"Database: {database_name}, Collection: {collection_name}")

            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]

            # ==== DEBUG: FETCH DOCUMENTS ====
            cursor = collection.find({}, projection=None).batch_size(500)

            docs = []
            for doc in cursor:
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])
                docs.append(doc)

            cursor.close()

            print(">>> DEBUG: Number of docs fetched from Mongo:", len(docs))

            df = pd.DataFrame(docs)

            print(">>> DEBUG: DataFrame shape right after Mongo read:", df.shape)
            print(">>> DEBUG: DataFrame head right after Mongo read:")
            print(df.head())

            if df.empty:
                logging.warning(
                    "Mongo collection returned ZERO documents. "
                    "Please check database_name, collection_name and that data is actually present."
                )

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)

            df.replace({"na": np.nan}, inplace=True)

            print(">>> DEBUG: DataFrame shape after cleaning:", df.shape)
            print(">>> DEBUG: DataFrame head after cleaning:")
            print(df.head())

            return df

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path

            logging.info(f"Saving data to feature store: {feature_store_file_path}")
            logging.info(f"Feature store dataframe shape: {dataframe.shape}")

            # creating folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)

            return dataframe

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
            print(">>> DEBUG: Inside split_data_as_train_test")
            print(">>> DEBUG: DataFrame shape:", dataframe.shape)
            print(">>> DEBUG: DataFrame head:")
            print(dataframe.head())

            # ==== SAFETY CHECK ====
            if dataframe is None or dataframe.shape[0] == 0 or dataframe.shape[1] == 0:
                msg = (
                    "Dataframe is empty (no rows/columns) before train_test_split. "
                    "Check MongoDB data, database/collection name, and DataIngestionConfig."
                )
                print(">>> DEBUG ERROR:", msg)
                logging.error(msg)
                raise NetworkSecurityException(msg, sys)

            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("Performed train test split on the dataframe")
            logging.info("Exited split_data_as_train_test method of Data_Ingestion class")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info("Exporting train and test file path.")

            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )
            logging.info("Exported train and test file path.")

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion process...")

            dataframe = self.export_collection_as_dataframe()
            logging.info(f"Dataframe shape after export_collection_as_dataframe: {dataframe.shape}")

            dataframe = self.export_data_into_feature_store(dataframe)
            logging.info(f"Dataframe shape after export_data_into_feature_store: {dataframe.shape}")

            self.split_data_as_train_test(dataframe)

            dataingestionartifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )

            logging.info("Data ingestion completed successfully.")
            logging.info(f"DataIngestionArtifact: {dataingestionartifact}")

            return dataingestionartifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
