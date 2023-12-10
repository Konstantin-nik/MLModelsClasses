from sklearn import model_selection, preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


class LinearModel:
    def __init__(self, df, test_size, random_state):
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.test_size = test_size
        self.random_state = random_state
        self.split_df(df, test_size, random_state)
        self.preprocessing_pipeline = Pipeline(
            [('scale', preprocessing.StandardScaler()),
             ('normalize', preprocessing.Normalizer())
             ])

    def split_df(self, df, test_size, random_state):
        self.X_train, self.X_test, \
            self.y_train, self.y_test = model_selection.train_test_split(df.drop('Type', axis=1), df['Type'],
                                                                         test_size=test_size,
                                                                         random_state=random_state)

    def train(self):
        fited_pipeline_1 = self.preprocessing_pipeline.fit(self.X_train)

        our_pipeline_2 = Pipeline(
            [('preprocessing_pipeline', fited_pipeline_1),
             ('classifier', LinearRegression())])

        self.fited_regression = our_pipeline_2.fit(X=self.X_train, y=self.y_train)

    def predict(self):
        return self.fited_regression.predict(self.X_test)
