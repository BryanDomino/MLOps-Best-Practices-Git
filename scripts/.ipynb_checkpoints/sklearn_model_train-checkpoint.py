import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils._testing import ignore_warnings
import json
import os
import mlflow
import mlflow.sklearn
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow import MlflowClient
from mlflow.models.signature import infer_signature
 
#Read in data
path = str('/mnt/data/{}/WineQualityData.csv'.format(os.environ.get('DOMINO_PROJECT_NAME')))
df = pd.read_csv(path)
print('Read in {} rows of data'.format(df.shape[0]))
 
#rename columns to remove spaces
for col in df.columns:
    df.rename({col: col.replace(' ', '_')}, axis =1, inplace = True)
 
#Create is_red variable to store red/white variety as int    
df['is_red'] = df.type.apply(lambda x : int(x=='red'))
 
#Find all pearson correlations of numerical variables with quality
corr_values = df.corr(numeric_only=True).sort_values(by = 'quality')['quality'].drop('quality',axis=0)
 
#Keep all variables with above a 8% pearson correlation
important_feats=corr_values[abs(corr_values)>0.08]
 
#Get data set up for model training and evaluation
 
#Drop NA rows
df = df.dropna(how='any',axis=0)
#Split df into inputs and target
X = df[important_feats.keys()]
y = df['quality'].astype('float64')
 
# create a new MLFlow experiemnt
mlflow.set_experiment(experiment_name=os.environ.get('DOMINO_PROJECT_NAME') + " " + os.environ.get('DOMINO_STARTING_USERNAME'))

from sklearn.linear_model import LinearRegression
from domino_data_capture.data_capture_client import DataCaptureClient
import uuid
import datetime

features = ['density', 'volatile_acidity', 'chlorides', 'is_red', 'alcohol']

target = ["quality"]

# pred_client = PredictionClient(features, target)
data_capture_client = DataCaptureClient(features, target)

class WineQualityModel(mlflow.pyfunc.PythonModel):
    def __init__(self,model):
        self.model = model
    
    # Assumes model_input is a list of lists
    def predict(self, context, model_input, params=None):
        event_time = datetime.datetime.now(datetime.timezone.utc).isoformat()

        
        print(model_input)
        wine_id=model_input.pop("wine_id")
        print(wine_id)
        
        model_input_value = [list(model_input.values())]
        print(model_input_value)
        prediction = self.model.predict(model_input_value)
        print(prediction[0])
        features = model_input_value[0]
        print(features)
        
        
            # Capture this prediction event so Domino can keep track
        data_capture_client.capturePrediction(features, prediction, event_id=wine_id,
                                timestamp=event_time)
        return prediction



 
with mlflow.start_run():
    # Set MLFlow tag to differenciate the model approaches
    mlflow.set_tag("Model_Type", "sklearn")
    
    #Create 70/30 train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    signature = infer_signature(X_test, y_test)
 
    #initiate and fit Gradient Boosted Classifier
    print('Training model...')
    gbr = GradientBoostingRegressor(loss='ls',learning_rate = 0.15, n_estimators=75, criterion = 'mse')
    gbr.fit(X_train,y_train)

    sLength = len(X_test['density'])
    X_test2 =  X_test.assign(wine_id=pd.Series(np.random.randn(sLength)).values)

    model = WineQualityModel(gbr)
    data= {
        "density": 0.99,
        "volatile_acidity": 0.028,
        "chlorides": 0.05,
        "is_red": 0.1,
        "alcohol": 11.1,
        "wine_id": 12312312312.1
      }
    preds = model.predict(X_test2,data)
 
    #Predict test set
    print('Evaluating model on test data...')
    preds = gbr.predict(X_test)
 
    #View performance metrics and save them to domino stats!
    print("R2 Score: ", round(r2_score(y_test, preds),3))
    print("MSE: ", round(mean_squared_error(y_test, preds),3))
    
    mlflow.log_param("learning_rate", 0.15)
    mlflow.log_param("n_estimators", 75)
    # Save the metrics in MLFlow
    mlflow.log_metric("R2", round(r2_score(y_test, preds),3))
    mlflow.log_metric("MSE", round(mean_squared_error(y_test,preds),3))
 
    #Code to write R2 value and MSE to dominostats value for population in experiment manager
    with open('/mnt/artifacts/dominostats.json', 'w') as f:
        f.write(json.dumps({"R2": round(r2_score(y_test, preds),3),
                           "MSE": round(mean_squared_error(y_test,preds),3)}))
 
    #Write results to dataframe for visualizations
    results = pd.DataFrame({'Actuals':y_test, 'Predictions':preds})
 
    print('Creating visualizations...')
    #Add visualizations and save for inspection
    fig1, ax1 = plt.subplots(figsize=(10,6))
    plt.title('Sklearn Actuals vs Predictions Scatter Plot')
    sns.regplot( 
        data=results,
        x = 'Actuals',
        y = 'Predictions',
        order = 3)
    plt.savefig('/mnt/artifacts/actual_v_pred_scatter.png')
    mlflow.log_figure(fig1, 'actual_v_pred_scatter.png')
 
    fig2, ax2 = plt.subplots(figsize=(10,6))
    plt.title('Sklearn Actuals vs Predictions Histogram')
    plt.xlabel('Quality')
    sns.histplot(results, bins=6, multiple = 'dodge', palette = 'coolwarm')
    plt.savefig('/mnt/artifacts/actual_v_pred_hist.png')
    mlflow.log_figure(fig2, 'actual_v_pred_hist.png')
    
 
 
    #Saving trained model to serialized pickle object 
    
    import pickle 
    
    # save best model
    file = '/mnt/code/sklearn_gbm.pkl'
    pickle.dump(gbr, open(file, 'wb'))
    
    #mlflow.sklearn.log_model(gbr,
    #                         artifact_path="gbr_model",
    #                         signature=signature,
    #                         )
    model_info = mlflow.pyfunc.log_model(
        #registered_model_name="sklearn-model", 
        python_model = model, 
        artifact_path="sklearn-model"
    )
mlflow.end_run()

print('Script complete!')