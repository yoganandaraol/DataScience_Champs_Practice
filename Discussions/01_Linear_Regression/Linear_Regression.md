# Import Libraries
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport                          # Import Pandas Profiling (To generate Univariate Analysis)
pd.set_option('display.float_format',lambda x: '%5f' % x)
import matplotlib.pyplot as plt
import seaborn as sns




# Data Preprocessing (standardization)
`from sklearn.preprocessing import StandardScaler`

	scalar = StandardScaler().fit(df) 
	// calculate math for the values like avg.. min, max, normally distributed or not

	scalar.transform(df)

* OHE is not a mandatory step for all model preparation
ex. Decision Trees model does not infulenced by OHE

## Label Encoding
### Import label encoder 
from sklearn import preprocessing
### label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder()
### Encode labels in column 'Country'. 
data['Country']= label_encoder.fit_transform(data[‘Country']) 
print(data.head())


#  Model Selection

`from sklearn.model_selection import train_test_split`

`train_test_split(X, y, test_size=0.30, random_state=1)`



# Create a Model

`from sklearn.linear_model import LinearRegression`

	lr = LinearRegression()
	lr.fit(X_train, y_train)  // Creates a model
	lr.predict(X_train)


https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

y = mx + c
y = m1x1 + m2x2 .... + c

	print('Intercept:',lr.intercept_) // c value                                     
    print('Coefficients:',lr.coef_)	// m values

# Calculate accuracy of model

`from sklearn import metrics`

`from sklearn.metrics import r2_score`

	metrics.mean_absolute_error
	metrics.mean_squared_error
	np.sqrt( metrics.mean_squared_error(...))
	r2_score(y_train, y_pred_train) 