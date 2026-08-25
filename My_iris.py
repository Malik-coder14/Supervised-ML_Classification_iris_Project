import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Flower Detective", layout = "centered")

st.title("Flower Detective")

st.write("Use flower meaasurements and let the model predict the iris species.")

# import the dataset
df = pd.read_csv("iris.csv")

# this gives user a choice: to show dataset or not on frontend.
if st.checkbox("Show the dataset"):
    st.dataframe(df)

feature_columns = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm"
]


X = df[feature_columns] # features / clues
y = df["species"] # Target / Label / answer

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# create and train the claassifier.
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test the model

test_predictions = model.predict(X_test)
accuracy = accuracy_score(y_test,test_predictions)
st.write(f"Model test accuracy: {accuracy: .0%}")

st.sidebar.header("Flower measurement")

sepal_length = st.sidebar.slider(
    "Sepal length (cm)",
    float(df["sepal_length_cm"].min()),
    float(df["sepal_length_cm"].max()),
    float(df["sepal_length_cm"].mean()),
    0.1,
)

sepal_width = st.sidebar.slider(
    "Sepal width (cm)",
    float(df["sepal_width_cm"].min()),
    float(df["sepal_width_cm"].max()),
    float(df["sepal_width_cm"].mean()),
    0.1,
)

petal_length = st.sidebar.slider(
    "Petal length (cm)",
    float(df["petal_length_cm"].min()),
    float(df["petal_length_cm"].max()),
    float(df["petal_length_cm"].mean()),
    0.1,
)
petal_width = st.sidebar.slider(
    "Petal width (cm)",
    float(df["petal_width_cm"].min()),
    float(df["petal_width_cm"].max()),
    float(df["petal_width_cm"].mean()),
    0.1,
)

input_flower = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns = feature_columns
)

prediction = model.predict(input_flower)[0]
probabilities = model.predict_proba(input_flower)[0]

st.subheader("Prediction")
st.success(f"Predicted Species: {prediction.title()}")

probability_table = pd.DataFrame(
    {"species": model.classes_, "probability": probabilities}).set_index("species")

st.subheader("Model confidence")
st.bar_chart(probability_table)
