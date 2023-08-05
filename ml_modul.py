from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class SklearnClassifier(object):
    def __init__(self, target_column, preprocessing_steps, models_param_grids, binary_classification=True):
        self.target_column = target_column
        self.preprocessing_steps = preprocessing_steps
        self.models_param_grids = models_param_grids
        self.binary_classification = binary_classification

    def fit(self, train_df):
        X_train = train_df.drop(self.target_column, axis=1)
        y_train = train_df[self.target_column]

        self.best_models = []

        for model, param_grid in self.models_param_grids:
            model_name, model_instance = model
            pipeline = Pipeline(steps=[('preprocessor', self.preprocessing_steps), model])

            grid_search = GridSearchCV(pipeline, param_grid=param_grid,
                                       scoring='f1' if self.binary_classification else 'f1_macro',
                                       cv=StratifiedKFold(n_splits=5), n_jobs=-1)
            grid_search.fit(X_train, y_train)
            self.best_models.append((model_name, grid_search.best_estimator_))

        # Select the best model
        self.best_model = max(self.best_models, key=lambda x: x[1].score(X_train, y_train))[1]

    def evaluate(self, test_df):
        X_test = test_df.drop(self.target_column, axis=1)
        y_test = test_df[self.target_column]

        y_pred = self.best_model.predict(X_test)

        print(classification_report(y_test, y_pred))

    def predict(self, data):
        X = data.drop(self.target_column, axis=1)

        prediction = self.best_model.predict(X)
        return prediction


if __name__ == "__main__":
    classifier = SklearnClassifier(
        target_column='HeartDisease',
        preprocessing_steps=ColumnTransformer(
            transformers=[
                ('num', StandardScaler(),
                 ['Age', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']),
                ('cat', OneHotEncoder(), ['Sex'])
            ]
        ),
        models_param_grids=[
            (('logreg', LogisticRegression()), {'logreg__C': [0.001, 0.01, 0.1, 1, 10, 100]}),
            (('dtree', DecisionTreeClassifier()),
             {'dtree__max_depth': [3, 4, 5], 'dtree__min_samples_split': [2, 3, 4, 5, 7, 10]}),
            (('xgboost', XGBClassifier()),
             {'xgboost__n_estimators': [50, 100, 200], 'xgboost__learning_rate': [0.01, 0.1, 0.3]})
        ],
        binary_classification=True
    )
