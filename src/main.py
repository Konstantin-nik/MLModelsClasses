import pandas as pd
import yaml
from matplotlib import pyplot as plt

from models.linear_model import LinearModel
from src.models.knn_model import KnnModel

with open('src/params/params.yaml', 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

iris_df = pd.read_csv('src/sources/iris_model.csv')
iris_df.rename(columns={'Y': 'Type'}, inplace=True)

model_1_params = params['model_1']

model_1 = LinearModel(iris_df, model_1_params['test_size'], model_1_params['random_state'])
model_1.train()
predicted_1 = model_1.predict()


model_2_params = params['model_2']

model_2 = KnnModel(iris_df, model_2_params['test_size'], model_2_params['random_state'])
model_2.train()
predicted_2 = model_2.predict()


plt.scatter(predicted_1, range(len(predicted_1)), label='Model_1')
plt.legend()
plt.scatter(predicted_2, range(len(predicted_2)), label='Model_2')
plt.legend()
plt.show()

