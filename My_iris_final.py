import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

st.set_page_config(page_title='Flower Detective - ML Classifier', page_icon='🌸', layout='wide')
st.title('🌸 Flower Detective')
st.caption('Supervised Machine Learning Classification Project')
st.write('Use flower measurements to train a classification model and predict the Iris species.')

@st.cache_data
def load_default_data():
    return pd.read_csv('iris.csv')

try:
    default_df = load_default_data()
except FileNotFoundError:
    st.error("Could not find 'iris.csv'. Place iris.csv in the same folder as this Python file.")
    st.stop()

st.sidebar.header('📂 Dataset')
uploaded_file = st.sidebar.file_uploader('Upload an Iris-compatible CSV', type=['csv'])
df = pd.read_csv(uploaded_file) if uploaded_file is not None else default_df.copy()
if uploaded_file is not None:
    st.sidebar.success('Uploaded dataset loaded.')

feature_columns = ['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm']
target_column = 'species'
missing_columns = [c for c in feature_columns + [target_column] if c not in df.columns]
if missing_columns:
    st.error('The dataset is missing these required columns: ' + ', '.join(missing_columns))
    st.stop()
df = df.dropna(subset=feature_columns + [target_column]).copy()
X, y = df[feature_columns], df[target_column]
if y.nunique() < 2:
    st.error('The target column must contain at least two classes.')
    st.stop()

st.sidebar.header('⚙️ Model Settings')
model_name = st.sidebar.selectbox('Choose a classifier', ['Random Forest', 'Logistic Regression', 'K-Nearest Neighbors', 'Support Vector Machine'])
test_size = st.sidebar.slider('Test set size (%)', 10, 40, 20, 5)
random_state = st.sidebar.number_input('Random state', 0, 999, 42, 1)

if model_name == 'Random Forest':
    n_estimators = st.sidebar.slider('Number of trees', 10, 500, 100, 10)
if model_name == 'K-Nearest Neighbors':
    max_neighbors = max(2, min(20, len(df) - 1))
    n_neighbors = st.sidebar.slider('Number of neighbors (K)', 1, max_neighbors, min(5, max_neighbors), 1)

try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=int(random_state), stratify=y)
except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=int(random_state))

if model_name == 'Random Forest':
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=int(random_state))
elif model_name == 'Logistic Regression':
    model = Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(max_iter=1000, random_state=int(random_state)))])
elif model_name == 'K-Nearest Neighbors':
    model = Pipeline([('scaler', StandardScaler()), ('classifier', KNeighborsClassifier(n_neighbors=n_neighbors))])
else:
    model = Pipeline([('scaler', StandardScaler()), ('classifier', SVC(probability=True, random_state=int(random_state)))])

model.fit(X_train, y_train)
test_predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, test_predictions)

st.header('📊 Project Dashboard')
c1, c2, c3, c4 = st.columns(4)
c1.metric('Total Samples', len(df))
c2.metric('Features', len(feature_columns))
c3.metric('Classes', y.nunique())
c4.metric('Test Accuracy', f'{accuracy:.1%}')
st.info(f'Current model: **{model_name}** | Training samples: **{len(X_train)}** | Testing samples: **{len(X_test)}**')

tab1, tab2, tab3, tab4, tab5 = st.tabs(['📋 Dataset', '📈 Model Performance', '🌸 Prediction', '🔎 Feature Importance', 'ℹ️ About Project'])

with tab1:
    st.subheader('Dataset Explorer')
    a, b = st.columns(2)
    with a:
        st.write('### Species Distribution')
        st.bar_chart(y.value_counts())
    with b:
        st.write('### Feature Statistics')
        st.dataframe(df[feature_columns].describe().round(2), use_container_width=True)
    if st.checkbox('Show complete dataset'):
        st.dataframe(df, use_container_width=True)
    st.write('### Feature Relationship')
    selected_x = st.selectbox('X-axis feature', feature_columns, index=0)
    selected_y = st.selectbox('Y-axis feature', feature_columns, index=2)
    chart_data = df[[selected_x, selected_y]].copy()
    chart_data['Species'] = y.values
    st.scatter_chart(chart_data, x=selected_x, y=selected_y)

with tab2:
    st.subheader('Model Performance')
    st.metric('Accuracy', f'{accuracy:.2%}')
    report = classification_report(y_test, test_predictions, output_dict=True, zero_division=0)
    st.write('### Classification Report')
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
    labels = list(model.classes_)
    cm = confusion_matrix(y_test, test_predictions, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f'Actual: {x}' for x in labels], columns=[f'Predicted: {x}' for x in labels])
    st.write('### Confusion Matrix')
    st.dataframe(cm_df, use_container_width=True)
    st.write('### Confusion Matrix Chart')
    st.bar_chart(cm_df)

with tab3:
    st.subheader('🌸 Predict an Iris Species')
    st.write('Adjust the flower measurements and let the trained model predict its species.')
    p1, p2 = st.columns(2)
    with p1:
        sepal_length = st.slider('Sepal length (cm)', float(df['sepal_length_cm'].min()), float(df['sepal_length_cm'].max()), float(df['sepal_length_cm'].mean()), 0.1)
        sepal_width = st.slider('Sepal width (cm)', float(df['sepal_width_cm'].min()), float(df['sepal_width_cm'].max()), float(df['sepal_width_cm'].mean()), 0.1)
    with p2:
        petal_length = st.slider('Petal length (cm)', float(df['petal_length_cm'].min()), float(df['petal_length_cm'].max()), float(df['petal_length_cm'].mean()), 0.1)
        petal_width = st.slider('Petal width (cm)', float(df['petal_width_cm'].min()), float(df['petal_width_cm'].max()), float(df['petal_width_cm'].mean()), 0.1)
    input_flower = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=feature_columns)
    st.write('### Input Measurements')
    st.dataframe(input_flower, use_container_width=True)
    prediction = model.predict(input_flower)[0]
    probabilities = model.predict_proba(input_flower)[0]
    st.success(f'🌼 Predicted Species: **{str(prediction).title()}**')
    probability_table = pd.DataFrame({'Species': model.classes_, 'Probability': probabilities}).sort_values('Probability', ascending=False)
    probability_table['Probability (%)'] = (probability_table['Probability'] * 100).round(2)
    st.write('### Model Confidence')
    st.dataframe(probability_table[['Species', 'Probability (%)']], use_container_width=True)
    st.bar_chart(probability_table.set_index('Species')['Probability'])

with tab4:
    st.subheader('🔎 Feature Importance')
    if model_name == 'Random Forest':
        importance_df = pd.DataFrame({'Feature': feature_columns, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
        st.dataframe(importance_df.round(4), use_container_width=True)
        st.bar_chart(importance_df.set_index('Feature')['Importance'])
    else:
        st.info('Feature importance is shown here for Random Forest. Select Random Forest from the sidebar to view it.')

with tab5:
    st.subheader('About This Project')
    st.markdown('''### Supervised Machine Learning Classification

This project demonstrates supervised machine learning using flower measurements to classify Iris species.

**Input features:** Sepal length, Sepal width, Petal length, Petal width

**Target:** Iris species

**Available classifiers:** Random Forest, Logistic Regression, K-Nearest Neighbors, Support Vector Machine

**Evaluation metrics:** Accuracy, Precision, Recall, F1-score, Confusion matrix
''')

st.divider()
st.caption('🌸 Flower Detective | Supervised Machine Learning Classification Project')
