import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from yellowbrick.classifier import ConfusionMatrix


def main():
    base = pd.read_csv('data/insurance.csv', keep_default_na=False)
    # print(base.head())

    base = base.drop(columns=['Unnamed: 0'])
    print(base.head())

    # verify the shape
    print(base.shape)

    # verify null elements
    print(base.isnull().sum())

    # dependent variable
    y = base.iloc[:,7].values

    # independent variables
    X = base.drop(base.columns[7], axis=1).values

    # Feature encoding
    labelenconder = LabelEncoder()
    for feature in range(X.shape[1]):
        if X[:,feature].dtype == 'object':
            X[:,feature] = labelenconder.fit_transform(X[:,feature])
    
    # Dividing between training base and test base
    X_training, X_test, y_training, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

    # Model
    model = GaussianNB()
    model.fit(X_training, y_training)  # Training
    forecast = model.predict(X_test)  # Testing

    # Evaluation
    accuracy = accuracy_score(y_test, forecast)
    precision = precision_score(y_test, forecast, average='weighted')
    recall = recall_score(y_test, forecast, average='weighted')
    f1 = f1_score(y_test, forecast, average='weighted')

    print(f'Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nf1: {f1}\n')

    report = classification_report(y_test, forecast)
    print(report)

if __name__ == '__main__':
    main()