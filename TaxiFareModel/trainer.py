# imports
from TaxiFareModel.data import get_data,clean_data
from TaxiFareModel.encoders import DistanceTransformer,TimeFeaturesEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from TaxiFareModel.utils import compute_rmse
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = "[FR] [Paris] [yassingofti] taxifare v1"

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
        ('dist_trans', DistanceTransformer()),
        ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        self.pipeline = pipe
        return pipe

    def run(self):
        """set and train the pipeline"""
        self.pipeline.fit(self.X, self.y)
        return self.pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)
        return rmse
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri("https://mlflow.lewagon.co/")
        return MlflowClient()
    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id
    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)



if __name__ == "__main__":

    # set X and y
    df = clean_data(get_data())
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    trainer = Trainer(X_train,y_train)
    trainer.set_pipeline()
    trainer.run()
    print(trainer.evaluate(X_test, y_test))
    yourname = "yassingofti"

    if yourname is None:
        print("please define your name, il will be used as a parameter to log")

    for model in ["linear", "Randomforest"]:
        client = trainer.mlflow_client
        run = client.create_run(trainer.mlflow_experiment_id)
        client.log_metric(run.info.run_id, "rmse", 4.5)
        client.log_param(run.info.run_id, "model", model)
        client.log_param(run.info.run_id, "student_name", yourname)
