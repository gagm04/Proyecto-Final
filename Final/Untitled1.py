#!/usr/bin/env python
# coding: utf-8

# In[77]:


#import necessary libraries
import pandas as pd
import numpy as np
import sklearn as skl
import matplotlib as plt
import pyarrow as pya
import pyarrow.parquet as pq
import warnings
#%matplotlib inline
warnings.filterwarnings('ignore')

# Read in iris data set 
#dataset abierto
iris = pd.read_csv("https://raw.githubusercontent.com/Thinkful-Ed/curric-data-001-data-sets/master/iris/iris.data.csv")

# Add column names, features as indepent variables
iris.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']

# Split data into features and target
a = iris.iloc[:, 0:4] # first four columns of data frame with all rows
b = iris.iloc[:, 4:] # last column of data frame (species) with all rows

#import libraries
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_validate 
from sklearn.model_selection import train_test_split

# Train, test split, a is X as matrix, b as y the vector
#training incremental de los campos
a_train, a_test, b_train, b_test = train_test_split(a,b, test_size=0.40, random_state=0)

from sklearn.svm import LinearSVC
#from sklearn.svm import SVC

# Build Linear Support Vector Classifier
#fit method to train the algorithm on the training data passed as parameter
clf = LinearSVC()
clf.fit(a_train, b_train.values.ravel())

# Make predictions on test set
predictions = clf.predict(a_test)


from sklearn.metrics import accuracy_score

# Assess model accuracy
result = accuracy_score(b_test, predictions, normalize=True)

#evaluating the algorithm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 

#dataframe = iris.DataFrame()
table = pya.Table.from_pandas(iris)
pq.write_table(table, 'iris.parquet.format')
table_parquet = pq.read_table('iris.parquet.format')
#print(table_parquet)
#iris.head(10).style.highlight_max(color='lightgreen', axis=0) 


# In[74]:


#imprime rosoa los elementos en formato de tabla
table_pandas = table_parquet.to_pandas()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(table_pandas)


# In[73]:


#imprime porcentaje de exactitud
print(result)


# In[32]:


#imprime reporte de clasificaciones
print(classification_report(b_test, predictions))


# In[9]:


#imprime las predicciones
print(predictions)


# In[76]:


#imprime estadística con la matriz de confusión

print(confusion_matrix (b_test, predictions))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




