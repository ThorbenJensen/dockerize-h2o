import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
import requests
import json


def train_model():
    # start h2o and load data
    h2o.init()
    h2o_df = h2o.load_dataset("iris.csv")
    # train model
    model = H2OGradientBoostingEstimator(ntrees=100,
                                         max_depth=4,
                                         learn_rate=0.1,
                                         model_id='latest')
    model.train(y="Species",
                training_frame=h2o_df)
    # save model to MOJO
    modelfile = model.download_mojo(path="models/", get_genmodel_jar=True)
    print("Model saved to " + modelfile)
    # save python model to disk
    h2o.save_model(model=model,
                   path=os.getcwd() + '/models/h2o_model',
                   force=True)

def predict():
    # predict
    url = 'http://localhost:54321/3/Predictions/models/latest/frames/new_data2'
    r = requests.post(url) \
                .json()
    prediction_url = 'http://localhost:54321' + r['predictions_frame']['URL']
    p = requests.get(prediction_url).json()
    columns = p.get('frames')[0]['columns']
    labels = [c['label'] for c in columns]
    data = [c['data'] for c in columns]
    print(labels)
    print(data)
    

if __name__ == '__main__':
    train_model()
    # add new data
    hdf = h2o.H2OFrame({"Sepal.Length": "5.1", "Sepal.Width": "3.5",
                        "Petal.Length": "1.4", "Petal.Width": "0.2"},
                        destination_frame='new_data2')
    # predict
    predict()

