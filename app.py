from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

app = Flask(__name__)

# Load the dataset
data = pd.read_csv(r'C:\Users\91630\Desktop\Student_Performance.csv')

# Extract features and target variable
X = data.drop('Performance Index', axis=1)  # Features
y = data['Performance Index']  # Target variable

# Apply label encoding to binary categorical variables
label_encoder = LabelEncoder()
X['Extracurricular Activities'] = label_encoder.fit_transform(X['Extracurricular Activities'])

# Train regression models
linear_reg = LinearRegression()
decision_tree_reg = DecisionTreeRegressor()
random_forest_reg = RandomForestRegressor()

# Train models
linear_reg.fit(X, y)
decision_tree_reg.fit(X, y)
random_forest_reg.fit(X, y)

# Save models using pickle
with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(linear_reg, f)

with open('decision_tree_model.pkl', 'wb') as f:
    pickle.dump(decision_tree_reg, f)

with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(random_forest_reg, f)

# Load models from pickle files
with open('linear_regression_model.pkl', 'rb') as f:
    linear_reg = pickle.load(f)

with open('decision_tree_model.pkl', 'rb') as f:
    decision_tree_reg = pickle.load(f)

with open('random_forest_model.pkl', 'rb') as f:
    random_forest_reg = pickle.load(f)

@app.route('/loging')
def loging():
    return render_template('loging.html')
@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    new_student_features = {}
    for feature in X.columns:
        value = request.form.get(feature)
        new_student_features[feature] = value

    new_student_df = pd.DataFrame([new_student_features])
    new_student_df['Extracurricular Activities'] = label_encoder.transform(new_student_df['Extracurricular Activities'])

    # Predict performance index for the new student using each model
    predictions_new_student = {
        "Linear Regression": linear_reg.predict(new_student_df)[0],
        "Decision Tree Regression": decision_tree_reg.predict(new_student_df)[0],
        "Random Forest Regression": random_forest_reg.predict(new_student_df)[0]
    }

    # Find the model with the highest predicted performance index
    best_model_name = max(predictions_new_student, key=predictions_new_student.get)
    best_prediction = predictions_new_student[best_model_name]

    return render_template('result.html', predictions=predictions_new_student, best_model=best_model_name, best_prediction=best_prediction)

if __name__ == '__main__':
    app.run(debug=True)
