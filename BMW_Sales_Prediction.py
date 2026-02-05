# Importing necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class BMWSalesData:
    def __init__(self, filepath):
        self.filepath = filepath

    # Load Dataset
    def load_data(self):

        # Loading the dataset
        self.data = pd.read_csv(self.filepath)
        print(self.data.head())

    # Logistic Regression Pipeline
    def logistic_regression(self):

        # Defining numeric columns for different scalers
        num_standard = ['Mileage_KM', 'Price_USD']
        num_robust = ['Engine_Size_L']

        # Creating pipelines for different scalers
        trf1 = Pipeline(steps=[
            ('Standard_Scaler', StandardScaler())
        ])
        trf2 = Pipeline(steps=[
            ('Robust_Scaler', RobustScaler())
        ])

        # Defining categorical columns for encoding
        cat_onehot = ['Model', 'Region', 'Fuel_Type', 'Color', 'Transmission']

        # Creating pipelines for different encoders
        trf3 = Pipeline(steps=[
            ('OneHot_Encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combining the transformers using ColumnTransformer
        preprocessor = ColumnTransformer([
            ('Standard_Scaler', trf1, num_standard),
            ('Robust_Scaler', trf2, num_robust),
            ('OneHot_Encoder', trf3, cat_onehot)
        ])

        # Encoding target variable
        self.data['Sales_Classification'] = self.data['Sales_Classification'].map({'Low': 0, 'High': 1})

        # Creating the final pipeline with preprocessor and Logistic Regression model
        pipe = Pipeline(steps=[
            ('Preprocessing', preprocessor),
            ('Model' , LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])

        #Splitting the data into features and target variable 
        X = self.data.drop(['Sales_Classification', 'Sales_Volume'], axis=1)
        y = self.data['Sales_Classification']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fitting the pipeline
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Evaluating the model
        print('Logistic Regression Model Evaluation: ')
        print(f'Confusion Matrix: {confusion_matrix(y_test, y_pred)}')
        print(f'Classification Report: {classification_report(y_test, y_pred)}')

    def decisiontree_classifier(self):

        # Defining categorical columns for encoding
        cat_onehot = ['Model', 'Region', 'Fuel_Type', 'Color', 'Transmission']

        # Transformer for OneHot-encoding
        trf1 = Pipeline(steps=[
            ('OneHot_Encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Preprocessing the columns using ColumnTransformer
        preprocessor = ColumnTransformer([
            ('OneHot-Encoder', trf1, cat_onehot)
        ])

        # Encoding the target variable
        self.data['Sales_Classification'] = self.data['Sales_Classification'].map({'Low': 0, 'High': 1})

        # Creating the final pipeline with preprocessor and DecisionTree classifier
        pipe = Pipeline([
            ('preprocessing', preprocessor),
            ('Model', DecisionTreeClassifier(
    max_depth=7,
    min_samples_leaf=20,
    min_samples_split=40,
    class_weight='balanced',
    random_state=42
)
)
        ])

        #Splitting the data into features and target variable 
        X = self.data.drop(['Sales_Classification', 'Sales_Volume'], axis=1)
        y = self.data['Sales_Classification']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fitting the pipeline
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Evaluating the model
        print('Decision Tree Model Evaluation: ')
        print(f'Confusion Matrix: {confusion_matrix(y_test, y_pred)}')
        print(f'Classification Report: {classification_report(y_test, y_pred)}')

    def randomforest_classifier(self):

        # Defining categorical columns for encoding
        cat_onehot = ['Model', 'Region', 'Fuel_Type', 'Color', 'Transmission']

        # Transformer for OneHot-encoding
        trf1 = Pipeline(steps=[
            ('OneHot_Encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Preprocessing the columns using ColumnTransformer
        preprocessor = ColumnTransformer([
            ('OneHot-Encoder', trf1, cat_onehot)
        ])

        # Encoding the target variable
        self.data['Sales_Classification'] = self.data['Sales_Classification'].map({'Low': 0, 'High': 1})

        # Creating the final pipeline with preprocessor and RandomForest classifier
        pipe = Pipeline([
            ('preprocessing', preprocessor),
            ('Model', RandomForestClassifier(
    n_estimators=500,
    max_depth=8,
    min_samples_leaf=20,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42
            ))
        ])

        # Splitting the data into features and target variable 
        X = self.data.drop(['Sales_Classification', 'Sales_Volume'], axis=1)
        y = self.data['Sales_Classification']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fitting the pipeline
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Evaluating the model
        print('Random Forest Model Evaluation: ')
        print(f'Confusion Matrix: {confusion_matrix(y_test, y_pred)}')
        print(f'Classification Report: {classification_report(y_test, y_pred)}')

predictor = BMWSalesData('Dataset/BMW Sales.csv')
predictor.load_data()

# To run any model comment out other two functions of the class
predictor.logistic_regression()
# predictor.decisiontree_classifier()
# predictor.randomforest_classifier()