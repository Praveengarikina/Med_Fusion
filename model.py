import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split as tts, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the csv file
df = pd.read_csv("Reordered_Training.csv")

# Feature and target separation
x = df.iloc[:, :-1]  # All columns except the last one as features
y = df["prognosis"]

# Encode the target variable
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Save the label encoder
pickle.dump(encoder, open("label_encoder.pkl", "wb"))

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = tts(x, y, test_size=0.20, random_state=142)

# Define the parameter grid for GridSearchCV
param_grid = {
    'max_depth': [None, 5, 10, 15],  # Try different max depths
    'min_samples_split': [2, 5, 10],  # Try different min samples split
    'min_samples_leaf': [1, 2, 4]  # Try different min samples leaf
}

# Initialize the model
model = DecisionTreeClassifier()

# Perform GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(x_train, y_train)

# Get the best model
best_tree = grid_search.best_estimator_

# Save the best model
pickle.dump(best_tree, open("model.pkl", "wb"))

# Make predictions
predictions = best_tree.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
