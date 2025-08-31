import os
import joblib

class ModelTrainer:
    def tain(self, model, X_train, Y_train):
        model.fit(X_train, Y_train)
        train_score = model.score(X_train, Y_train)
        return model, train_score

    def save_model(self, model, filepath: str) -> None:
        joblib.dump(model, filepath)

    def load_model(self, filepath: str):
        return joblib.load(filepath)