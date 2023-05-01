# Import all the tools we need 

import warnings
import pickle
warnings.filterwarnings('ignore')

import pandas as pd



# Models from Scikit-learn
from sklearn.tree import DecisionTreeClassifier

# Model Evaluations 
from sklearn.model_selection import train_test_split



# Load csv file

def wrangle(filepath):
    #read csv file into a dataframe
    df = pd.read_csv(filepath)
    #return df 
    return df

# load data
df = wrangle("storedata.csv")
print("df shape:", df.shape)
print("Data head", df[:5])

# This will turn all of the string values into category values
for label, content in df.items():
    if pd.api.types.is_string_dtype(content):
        df[label] = content.astype("category").cat.as_ordered()

# Turn categorical variables into numbers
for label, content in df.items():
    # Check columns which *aren't* numeric
    if not pd.api.types.is_numeric_dtype(content):
        # We add the +1 because pandas encodes missing categories as -1
        df[label] = pd.Categorical(content).codes+1   
        

# Select dependent and independent variable 
X = df[["Competition score", "Floor Space", "Staff", "Location", "10 min population","20 min population", "Manager name"]]
y = df["Performance"]

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state = 42, test_size=0.2)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)



# Developing machine learning model 
model = DecisionTreeClassifier(
     min_samples_leaf = 7,
     criterion = 'gini'
)
model.fit(X_train, y_train)

# Make pickle file of our model
filename = 'spc_model.pk1'
pickle.dump(model, open(filename, 'wb'))