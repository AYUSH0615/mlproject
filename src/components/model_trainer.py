import os
import sys
from  dataclasses import dataclass

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerconfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerconfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
            train_array[:,:-1],
            train_array[:,-1],
            test_array[:,:-1],
            test_array[:,-1]
        )
            models={
            "Linear Regression":LinearRegression(),
            "K-Neighbors Regressor":KNeighborsRegressor(),
            "Decision Tree":DecisionTreeRegressor(),
            "Random Forrest Regressor":RandomForestRegressor(),
            "AdaBoost Regressor":AdaBoostRegressor(),
             "CatBoosting Regressor":CatBoostRegressor(),
             "XgBoost Regressor":XGBRegressor(),
             "Gradient Regressor":GradientBoostingRegressor() }
            
            param={

          "Decision Tree":{
              'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
        
          },
          "Random Forrest Regressor":{
              'n_estimators':[8,17,64,30,130,300]
          },
          "Gradient Regressor":{
              'learning_rate':[0.1,.01,.5,.001],
              'n_estimators':[8,28,49,129,320]
          },
          "K-Neighbors Regressor":{
              'n_neighbors':[5,7,10,11],
              'weights':['uniform','distance']
          },
          "CatBoosting Regressor":{
              'depth':[6,8,10],
              'learning_rate':[0.01,0.1,0.5],
              'iterations':[25,30,80]
          },
          "AdaBoost Regressor":{
              'learning_rate':[.1,0.5,0.01,0.3],
              'n_estimators':[8,15,30,110,250]
          },
          "Linear Regression":{},
          "XgBoost Regressor":{
              'learning_rate':[0.01,0.1,0.2],
              'n_estimators':[50,100,200],
              'max_depth':[3,5,7]
          }
            }
            
              
        
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,
                                         y_test=y_test,models=models,param=param)
        

            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
            best_model=models[best_model_name]

            if best_model_score<0.7:
               raise CustomException("No best model found")
            logging.info("best found model on both train and test datasets")

            save_object(
            file_path=self.model_trainer_config.trained_model_file_path,
            obj=best_model
        )
                                         
            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted)
            return r2_square,best_model_name
    

        except Exception as e:
          raise CustomException(e,sys)