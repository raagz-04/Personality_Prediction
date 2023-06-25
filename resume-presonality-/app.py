from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd

app = Flask(__name__, template_folder='templates')

# Load the dataset
dataset = pd.read_csv('resume_dataset.csv')

# Home page - resume upload form
@app.route('/')
def home():
    return render_template('index.html')

# Handle resume upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    resume = request.files['resume']
    resume_text = resume.read().decode('utf-8')

    # Preprocess the resume text and extract features
    # Here, we are using CountVectorizer as an example
    vectorizer = CountVectorizer()
    resume_vec = vectorizer.fit_transform([resume_text])

    # Prepare the training data
    X_train = dataset[['Age', 'openness', 'neuroticism', 'conscientiousness', 'agreeableness', 'extraversion']]
    y_train = dataset['Personality (Class label)']

     # Print the shape of X_train and y_train
    print('Shape of X_train:', X_train.shape)
    print('Shape of y_train:', y_train.shape)

    # Fit the vectorizer on training data
    vectorizer.fit(X_train)

    # Transform the resume text using the trained vectorizer
    resume_vec = vectorizer.transform([resume_text])
     # Print the shape of resume_vec
    print('Shape of resume_vec:', resume_vec.shape)

    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Print the coefficients
    print('Coefficients:', model.coef_)
    
    
    # Predict the personality type
    prediction = model.predict(resume_vec)[0]
   

    return render_template('result.html',prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)






