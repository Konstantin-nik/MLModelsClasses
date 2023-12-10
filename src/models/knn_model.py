from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class KnnModel:
    def __init__(self, df, test_size, random_state):
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.test_size = test_size
        self.random_state = random_state
        self.split_df(df, test_size, random_state)
        self.preprocessing_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier())
        ])

    def split_df(self, df, test_size, random_state):
        self.X_train, self.X_test, \
            self.y_train, self.y_test = model_selection.train_test_split(df.drop('Type', axis=1), df['Type'],
                                                                         test_size=test_size,
                                                                         random_state=random_state)

    def train(self):
        self.fited_regression = self.preprocessing_pipeline.fit(X=self.X_train, y=self.y_train)

    def predict(self):
        return self.fited_regression.predict(self.X_test)
