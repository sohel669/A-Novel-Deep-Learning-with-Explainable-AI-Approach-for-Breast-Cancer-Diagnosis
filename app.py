from flask import Flask, render_template, request
from xgboost import XGBClassifier
import numpy as np
import pickle
import pandas as pd
import dill
from lime import lime_tabular
import base64
from io import BytesIO

ml_model=pickle.load(open('breast_cancer_detector.pickle', 'rb'))
with open('xai_model.pkl', 'rb') as file:
    explainer = dill.load(file)


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/results', methods=['POST'])
def results():
    
    mean_radius = request.form.get('mean_radius')
    mean_texture = request.form.get('mean_texture')
    mean_perimeter = request.form.get('mean_perimeter')
    mean_area = request.form.get('mean_area')
    mean_smoothness = request.form.get('mean_smoothness')
    mean_compactness = request.form.get('mean_compactness')
    mean_concavity = request.form.get('mean_concavity')
    mean_concave_points = request.form.get('mean_concave_points')
    mean_symmetry = request.form.get('mean_symmetry')
    mean_fractal_dimensions = request.form.get('mean_fractal_dimensions')

    radius_error = request.form.get('radius_error')
    texture_error = request.form.get('texture_error')
    perimeter_error = request.form.get('perimeter_error')
    area_error = request.form.get('area_error')
    smoothness_error = request.form.get('smoothness_error')
    compactness_error = request.form.get('compactness_error')
    concavity_error = request.form.get('concavity_error')
    concave_points_error = request.form.get('concave_points_error')
    symmetry_error = request.form.get('symmetry_error')
    fractal_dimension_error = request.form.get('fractal_dimension_error')

    worst_radius = request.form.get('worst_radius')
    worst_texture = request.form.get('worst_texture')
    worst_perimeter = request.form.get('worst_perimeter')
    worst_area = request.form.get('worst_area')
    worst_smoothness = request.form.get('worst_smoothness')
    worst_compactness = request.form.get('worst_compactness')
    worst_concavity = request.form.get('worst_concavity')
    worst_concave_points = request.form.get('worst_concave_points')
    worst_symmetry = request.form.get('worst_symmetry')
    worst_fractal_dimension = request.form.get('worst_fractal_dimension')
    

    # ================================
    #    hard code values, so remove later
    
    mean_radius = 14.56
    mean_texture = 332321.23
    mean_perimeter = 2233.78
    mean_area = 5323243.21323
    mean_smoothness = 0.14090
    mean_compactness = 0.39
    mean_concavity = 0.52
    mean_concave_points = 0.08
    mean_symmetry = 4330.24
    mean_fractal_dimensions = 1.59
    
    radius_error = 1.34
    texture_error = 3.78
    perimeter_error = 5.12
    area_error = 21.45
    smoothness_error = 0.04
    compactness_error = 0.09
    concavity_error = 0.13
    concave_points_error = 0.03
    symmetry_error = 0.06
    fractal_dimension_error = 0.08
    
    worst_radius = 16.23
    worst_texture = 35.78
    worst_perimeter = 98.12
    worst_area = 584.65
    worst_smoothness = 0.19
    worst_compactness = 0.44
    worst_concavity = 0.61
    worst_concave_points = 0.11
    worst_symmetry = 0.29
    worst_fractal_dimension = 1.67

    # ================================

    '''
    mean_radius = 13.54
    mean_texture = 14.36
    mean_perimeter = 87.46
    mean_area = 566.3
    mean_smoothness = 0.09779
    mean_compactness = 0.08129
    mean_concavity = 0.06664
    mean_concave_points = 0.04781
    mean_symmetry = 0.1885
    mean_fractal_dimensions = 0.05766
    
    radius_error = 0.2699
    texture_error = 0.7886
    perimeter_error = 2.058
    area_error = 23.56
    smoothness_error = 0.008462
    compactness_error = 0.0146
    concavity_error = 0.02387
    concave_points_error = 0.01315
    symmetry_error = 0.0198
    fractal_dimension_error = 0.023
    
    worst_radius = 15.11
    worst_texture = 19.26
    worst_perimeter = 99.7
    worst_area = 711.2
    worst_smoothness = 0.144
    worst_compactness = 0.1773
    worst_concavity = 0.239
    worst_concave_points = 0.1288
    worst_symmetry = 0.2977
    worst_fractal_dimension = 0.07259
    '''
    
    # Call your machine learning model to make a prediction
    # prediction = predict_cancer(mean_radius, mean_texture, ...)
    features_values=[mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness,mean_compactness,mean_concavity,mean_concave_points,mean_symmetry,mean_fractal_dimensions,radius_error,texture_error,perimeter_error,area_error,smoothness_error,compactness_error,concavity_error,concave_points_error,symmetry_error,fractal_dimension_error,worst_radius,worst_texture,worst_perimeter,worst_area,worst_smoothness,worst_compactness,worst_concavity,worst_concave_points,worst_symmetry,worst_fractal_dimension]
    features=[float(x) for x in features_values]
    feature_vals=[np.array(features)]

    feature_names= ['mean radius', 'mean texture', 'mean perimeter', 'mean area','mean smoothness', 'mean compactness', 'mean concavity','mean concave points', 'mean symmetry', 'mean fractal dimension','radius error', 'texture error', 'perimeter error', 'area error','smoothness error', 'compactness error', 'concavity error','concave points error', 'symmetry error','fractal dimension error', 'worst radius', 'worst texture','worst perimeter', 'worst area', 'worst smoothness','worst compactness', 'worst concavity', 'worst concave points','worst symmetry', 'worst fractal dimension']
    df=pd.DataFrame(feature_vals,columns=feature_names)
    prediction=ml_model.predict(df)

    if prediction==0:
        prediction_text="No Breast Cancer"
    else:
        prediction_text="Breast Cancer"
    print(prediction_text)
    predict_fn_xgb = lambda x: ml_model.predict_proba(x).astype(float)
    # Pass the user input values as the chosen instance (as a numpy array)
    choosen_instance = np.array(features)
    exp = explainer.explain_instance(choosen_instance, predict_fn_xgb, num_features=30)
    # Save the LIME report as an HTML file
    exp.save_to_file('static/lime_report.html')

    # Render the results page with the prediction and LIME report
    return render_template('results.html', prediction=prediction_text, lime_report='lime_report.html')




if __name__ == '__main__':
    app.run(debug=True)