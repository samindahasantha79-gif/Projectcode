import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


# Load Dataset

print("Reading Dataset...")
df = pd.read_csv("data/feature_time_48k_2048_load_1.csv")


# Labels

def simplify_labels(label):
    if 'Normal' in label:
        return 'Normal'
    elif 'IR' in label:
        return 'Inner_Race_Fault'
    elif 'OR' in label:
        return 'Outer_Race_Fault'
    elif 'Ball' in label:
        return 'Ball_Fault'
    else:
        return label

# lable add to fault 
df['fault'] = df['fault'].apply(simplify_labels)


# 3. separate Features (X) and  Labels (y) 


X = df.drop('fault', axis=1) 
y = df['fault']


# separate Train and Test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# AI Model  Train 

print(" Training Random Forest AI Model...  ")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# test Accuracy

y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n Model Training complete")
print(f" Accuracy: {accuracy * 100:.2f}%\n")
print(" Classification Report:")
print(classification_report(y_test, y_pred))


# Model Save 

model_filename = 'rf_bearing_model.pkl'
joblib.dump(rf_model, model_filename)
print(f"\n Model saved '{model_filename}' succesfully")