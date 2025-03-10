import pandas as pd
import numpy as np

def predictMPG(model, input_features):
    """
    Predicts the miles per gallon (MPG) based on input vehicle features.
    
    Parameters:
        model: Trained regression model (RandomForestRegressor)
        input_features: A list or NumPy array containing values for 
                         ['cylinders', 'displacement', 'horsepower', 'weight',
                          'acceleration', 'model_year', 'origin', 'vehicle_age', 'make_label']

    Returns:
        Predicted MPG value
    """
    input_columns = ['cylinders', 'displacement', 'horsepower', 'weight',
                     'acceleration', 'model_year', 'origin', 'vehicle_age', 'make_label']
    
    if isinstance(input_features, (list, np.ndarray)):
        input_features = pd.DataFrame([input_features], columns=input_columns)
    
    mpg_pred = model.predict(input_features)
    return mpg_pred[0]

