# this script reads classification data and trains/tests 4 ML models 

import pandas as pd
from sklearn.model_selection import train_test_split # used to set up the train/test data

from sklearn.pipeline import make_pipeline          # used to create testing pipeline
from sklearn.preprocessing import StandardScaler    # normalizes data

from sklearn.linear_model import LogisticRegression, RidgeClassifier                # training algos
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier     # trainig algos

from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle 

df = pd.read_csv('shoulder_press_coords.csv')

X = df.drop('class', axis=1) # features
y = df['class']# target value

# xy_train train the model --> randomly selected data from CSV as training dataset 
# xy_test test the trained model --> randomly selected data from the CSV to test the model
# testing partiion is 30% 
# random_state is a random variable seed value
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

# setup pipelines --> think of each pipe as seperate model
pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression()),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}

fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model

for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    print(algo, accuracy_score(y_test, yhat))

# saving the model 
with open('shoulder_press.pkl', 'wb') as f:
    pickle.dump(fit_models['rf'], f)

# trained model results:
# lr 1.0
# rc 1.0
# rf 1.0 <-- use this one but usually would test each model using metrics -- should do this for actual project
# gb 1.0
# these results are very sus in most cases (100% accuracy is not normal) but 
# since the classes are very easy to differentiate it makes sense 

