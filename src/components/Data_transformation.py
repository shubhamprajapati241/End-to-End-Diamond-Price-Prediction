import os 
import sys 
import pandas as pd
import numpy as np 

from dataclasses import dataclass 
from src.logger import logging 
from src.exception import CustomException 

from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OrdinalEncoder, StandardScaler 
from src.utils.utils import save_object

from src.components.Data_ingestion import DataIngestion
from src.components.Model_trainer import ModelTrainer

from src.components.Model_evaluation import ModelEvaluation

@dataclass 
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("Artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_tranformation(self):
        try: 
            logging.info("Data transformation initiated")

            # define which columns should be ordinal encoded and which should be scaled 
            categorical_cols = ['cut', 'color', 'clarity']
            numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

            # define the custom ranking for each categorical variable 
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            ## defining the numerical and categorical pipeline
            logging.info('Pipeline Initiated')
            num_pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer()),
                    ("scaler", StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("ordinalencoder", OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories]))
                ]
            )
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_cols),
                    ("cat_pipeline", cat_pipeline, categorical_cols)
                ]
            )
            logging.info('Pipeline Completed')
            return preprocessor

        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            raise CustomException(e, sys)

    def initialize_data_transformation(self, train_path, test_path):
        try: 
            logging.info("Reading train and test data files")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data files completed")

            preprocessor_obj = self.get_data_tranformation()
            target_column_name = 'price'
            drop_columns = [target_column_name, 'id']

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing datasets.")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.fit_transform(input_feature_test_df)
            logging.info("Applied preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            logging.info("Preprocessed pickle file saved")

            return (
                train_arr,
                test_arr
            )

        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            raise CustomException(e, sys)

        
if __name__ == "__main__":
    # obj = DataIngestion()
    # train_path, test_path = obj.initiate_data_ingestion()

    # obj2 = DataTransformation()
    # train_arr, test_arr = obj2.initialize_data_transformation(train_path, test_path)

    # obj3 = ModelTrainer()
    # obj3.initate_model_training(train_arr, test_arr)

    # obj4 = ModelEvaluation()
    # obj4.initiate_model_evaluation(test_arr)
    pass

