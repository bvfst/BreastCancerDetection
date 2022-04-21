Applying Logistic Regression, Decision Trees and L


```python
import warnings
warnings.filterwarnings('ignore')
```


```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report
from sklearn.tree import export_graphviz
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import copy
import math
```


```python
data = pd.read_csv("wdbc.data", header = None)
```

#### Splitting the data into Train, Eval and Test


```python
train_eval_set, test_set = train_test_split(data, test_size = 0.2, random_state = 42, stratify = data[1])
```


```python
train_set, eval_set = train_test_split(train_eval_set, test_size = 0.3, random_state = 42, stratify = train_eval_set[1])
```


```python
# print(data.shape)
# print(train_set.shape)
# print(eval_set.shape)
# print(test_set.shape)
```


```python
train_X = train_set.drop([0, 1], axis = 1)
train_Y = train_set[1]
```


```python
eval_X = eval_set.drop([0, 1], axis = 1)
eval_Y = eval_set[1]
```


```python
test_X = test_set.drop([0, 1], axis = 1)
test_Y = test_set[1]
```


```python
train_eval_set_X = train_eval_set.drop([0, 1], axis = 1)
train_eval_set_Y = train_eval_set[1]
```

#### Exploring the dataset


```python
train_set.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>223</th>
      <td>8812877</td>
      <td>M</td>
      <td>15.750</td>
      <td>20.25</td>
      <td>102.60</td>
      <td>761.3</td>
      <td>0.10250</td>
      <td>0.12040</td>
      <td>0.11470</td>
      <td>0.064620</td>
      <td>...</td>
      <td>19.560</td>
      <td>30.29</td>
      <td>125.90</td>
      <td>1088.0</td>
      <td>0.1552</td>
      <td>0.4480</td>
      <td>0.39760</td>
      <td>0.14790</td>
      <td>0.3993</td>
      <td>0.10640</td>
    </tr>
    <tr>
      <th>237</th>
      <td>883263</td>
      <td>M</td>
      <td>20.480</td>
      <td>21.46</td>
      <td>132.50</td>
      <td>1306.0</td>
      <td>0.08355</td>
      <td>0.08348</td>
      <td>0.09042</td>
      <td>0.060220</td>
      <td>...</td>
      <td>24.220</td>
      <td>26.17</td>
      <td>161.70</td>
      <td>1750.0</td>
      <td>0.1228</td>
      <td>0.2311</td>
      <td>0.31580</td>
      <td>0.14450</td>
      <td>0.2238</td>
      <td>0.07127</td>
    </tr>
    <tr>
      <th>87</th>
      <td>86135502</td>
      <td>M</td>
      <td>19.020</td>
      <td>24.59</td>
      <td>122.00</td>
      <td>1076.0</td>
      <td>0.09029</td>
      <td>0.12060</td>
      <td>0.14680</td>
      <td>0.082710</td>
      <td>...</td>
      <td>24.560</td>
      <td>30.41</td>
      <td>152.90</td>
      <td>1623.0</td>
      <td>0.1249</td>
      <td>0.3206</td>
      <td>0.57550</td>
      <td>0.19560</td>
      <td>0.3956</td>
      <td>0.09288</td>
    </tr>
    <tr>
      <th>61</th>
      <td>858981</td>
      <td>B</td>
      <td>8.598</td>
      <td>20.98</td>
      <td>54.66</td>
      <td>221.8</td>
      <td>0.12430</td>
      <td>0.08963</td>
      <td>0.03000</td>
      <td>0.009259</td>
      <td>...</td>
      <td>9.565</td>
      <td>27.04</td>
      <td>62.06</td>
      <td>273.9</td>
      <td>0.1639</td>
      <td>0.1698</td>
      <td>0.09001</td>
      <td>0.02778</td>
      <td>0.2972</td>
      <td>0.07712</td>
    </tr>
    <tr>
      <th>500</th>
      <td>914862</td>
      <td>B</td>
      <td>15.040</td>
      <td>16.74</td>
      <td>98.73</td>
      <td>689.4</td>
      <td>0.09883</td>
      <td>0.13640</td>
      <td>0.07721</td>
      <td>0.061420</td>
      <td>...</td>
      <td>16.760</td>
      <td>20.43</td>
      <td>109.70</td>
      <td>856.9</td>
      <td>0.1135</td>
      <td>0.2176</td>
      <td>0.18560</td>
      <td>0.10180</td>
      <td>0.2177</td>
      <td>0.08549</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>




```python
train_set.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 318 entries, 223 to 493
    Data columns (total 32 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   0       318 non-null    int64  
     1   1       318 non-null    object 
     2   2       318 non-null    float64
     3   3       318 non-null    float64
     4   4       318 non-null    float64
     5   5       318 non-null    float64
     6   6       318 non-null    float64
     7   7       318 non-null    float64
     8   8       318 non-null    float64
     9   9       318 non-null    float64
     10  10      318 non-null    float64
     11  11      318 non-null    float64
     12  12      318 non-null    float64
     13  13      318 non-null    float64
     14  14      318 non-null    float64
     15  15      318 non-null    float64
     16  16      318 non-null    float64
     17  17      318 non-null    float64
     18  18      318 non-null    float64
     19  19      318 non-null    float64
     20  20      318 non-null    float64
     21  21      318 non-null    float64
     22  22      318 non-null    float64
     23  23      318 non-null    float64
     24  24      318 non-null    float64
     25  25      318 non-null    float64
     26  26      318 non-null    float64
     27  27      318 non-null    float64
     28  28      318 non-null    float64
     29  29      318 non-null    float64
     30  30      318 non-null    float64
     31  31      318 non-null    float64
    dtypes: float64(30), int64(1), object(1)
    memory usage: 82.0+ KB
    


```python
data[1].value_counts()
```




    B    357
    M    212
    Name: 1, dtype: int64




```python
train_set.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3.180000e+02</td>
      <td>318.000000</td>
      <td>318.000000</td>
      <td>318.000000</td>
      <td>318.000000</td>
      <td>318.000000</td>
      <td>318.000000</td>
      <td>318.000000</td>
      <td>318.000000</td>
      <td>318.000000</td>
      <td>...</td>
      <td>318.000000</td>
      <td>318.000000</td>
      <td>318.000000</td>
      <td>318.000000</td>
      <td>318.000000</td>
      <td>318.000000</td>
      <td>318.000000</td>
      <td>318.000000</td>
      <td>318.000000</td>
      <td>318.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.947455e+07</td>
      <td>14.187748</td>
      <td>19.366667</td>
      <td>92.351635</td>
      <td>660.859748</td>
      <td>0.095678</td>
      <td>0.103543</td>
      <td>0.087529</td>
      <td>0.048458</td>
      <td>0.180300</td>
      <td>...</td>
      <td>16.364953</td>
      <td>25.816730</td>
      <td>107.861447</td>
      <td>891.043711</td>
      <td>0.132243</td>
      <td>0.255595</td>
      <td>0.275599</td>
      <td>0.115283</td>
      <td>0.290668</td>
      <td>0.083622</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.496882e+08</td>
      <td>3.548175</td>
      <td>4.337178</td>
      <td>24.491744</td>
      <td>357.611220</td>
      <td>0.014299</td>
      <td>0.052250</td>
      <td>0.076948</td>
      <td>0.038339</td>
      <td>0.027124</td>
      <td>...</td>
      <td>4.875143</td>
      <td>6.020909</td>
      <td>33.904947</td>
      <td>585.521940</td>
      <td>0.023867</td>
      <td>0.153330</td>
      <td>0.207647</td>
      <td>0.064318</td>
      <td>0.063623</td>
      <td>0.017157</td>
    </tr>
    <tr>
      <th>min</th>
      <td>8.670000e+03</td>
      <td>7.691000</td>
      <td>9.710000</td>
      <td>48.340000</td>
      <td>170.400000</td>
      <td>0.064290</td>
      <td>0.023440</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.106000</td>
      <td>...</td>
      <td>8.678000</td>
      <td>12.020000</td>
      <td>54.490000</td>
      <td>223.600000</td>
      <td>0.071170</td>
      <td>0.027290</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.156600</td>
      <td>0.055210</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>8.702288e+05</td>
      <td>11.710000</td>
      <td>16.392500</td>
      <td>75.007500</td>
      <td>420.300000</td>
      <td>0.084748</td>
      <td>0.063238</td>
      <td>0.029260</td>
      <td>0.019787</td>
      <td>0.161825</td>
      <td>...</td>
      <td>13.015000</td>
      <td>21.407500</td>
      <td>84.367500</td>
      <td>516.425000</td>
      <td>0.116600</td>
      <td>0.148600</td>
      <td>0.120225</td>
      <td>0.065518</td>
      <td>0.250525</td>
      <td>0.071155</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9.083315e+05</td>
      <td>13.425000</td>
      <td>18.760000</td>
      <td>86.735000</td>
      <td>554.300000</td>
      <td>0.095190</td>
      <td>0.093020</td>
      <td>0.060765</td>
      <td>0.033480</td>
      <td>0.177750</td>
      <td>...</td>
      <td>15.050000</td>
      <td>25.305000</td>
      <td>98.085000</td>
      <td>686.600000</td>
      <td>0.131150</td>
      <td>0.216550</td>
      <td>0.229000</td>
      <td>0.099425</td>
      <td>0.281650</td>
      <td>0.079525</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.910741e+06</td>
      <td>16.090000</td>
      <td>21.832500</td>
      <td>104.600000</td>
      <td>798.550000</td>
      <td>0.104050</td>
      <td>0.130500</td>
      <td>0.132050</td>
      <td>0.070095</td>
      <td>0.195300</td>
      <td>...</td>
      <td>19.197500</td>
      <td>30.342500</td>
      <td>127.000000</td>
      <td>1123.250000</td>
      <td>0.146925</td>
      <td>0.343475</td>
      <td>0.384700</td>
      <td>0.161375</td>
      <td>0.320400</td>
      <td>0.091640</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.113205e+08</td>
      <td>27.420000</td>
      <td>39.280000</td>
      <td>186.900000</td>
      <td>2501.000000</td>
      <td>0.163400</td>
      <td>0.345400</td>
      <td>0.426800</td>
      <td>0.201200</td>
      <td>0.304000</td>
      <td>...</td>
      <td>36.040000</td>
      <td>49.540000</td>
      <td>251.200000</td>
      <td>4254.000000</td>
      <td>0.222600</td>
      <td>0.937900</td>
      <td>1.252000</td>
      <td>0.275600</td>
      <td>0.663800</td>
      <td>0.173000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 31 columns</p>
</div>




```python
train_set.drop([0, 1]).hist(bins = 50, figsize=(20,15))
plt.show()
```


    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_17_0.png)
    



```python
for col in range(3, 32):
    sns.boxplot(y = train_set[1], x = col, data=train_set)
    plt.show()
```


    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_18_0.png)
    



    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_18_1.png)
    



    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_18_2.png)
    



    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_18_3.png)
    



    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_18_4.png)
    



    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_18_5.png)
    



    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_18_6.png)
    



    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_18_7.png)
    



    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_18_8.png)
    



    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_18_9.png)
    



    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_18_10.png)
    



    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_18_11.png)
    



    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_18_12.png)
    



    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_18_13.png)
    



    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_18_14.png)
    



    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_18_15.png)
    



    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_18_16.png)
    



    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_18_17.png)
    



    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_18_18.png)
    



    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_18_19.png)
    



    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_18_20.png)
    



    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_18_21.png)
    



    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_18_22.png)
    



    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_18_23.png)
    



    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_18_24.png)
    



    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_18_25.png)
    



    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_18_26.png)
    



    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_18_27.png)
    



    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_18_28.png)
    



```python
## Visualization of the data set
## From the above we select 22, 29 as the best features
from mpl_toolkits.mplot3d import Axes3D
i = 22
j = 29
X = train_set.loc[:, [i, j]] # we only take the important two features.
Y = copy.deepcopy(train_set[1])
Y[Y == "B"] = 0
Y[Y == "M"] = 1
x_min, x_max = X[i].min() - .05, X[i].max() + .05
y_min, y_max = X[j].min() - .05, X[j].max() + .05

plt.figure(figsize=(8, 6))

# Plot the training points
plt.scatter(X[i], X[j], c=Y.values, cmap=plt.cm.Paired)
plt.xlabel(i)
plt.ylabel(j)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
```




    ([], [])




    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_19_1.png)
    


#### Normalisation


```python
scaler = MinMaxScaler()
```


```python
train_X_norm = scaler.fit_transform(train_X)
```


```python
eval_X_norm = scaler.transform(eval_X)
```


```python
test_X_norm = scaler.transform(test_X)
```


```python
train_eval_X_norm = scaler.transform(train_eval_set_X)
```

#### Decision Trees


```python
distributions = dict(criterion = ["entropy", "gini"],
                     max_depth = range(2,10),
                     min_samples_leaf = range(5, 20)
)
```


```python
tree_clf = DecisionTreeClassifier()
```


```python
# clf = RandomizedSearchCV(tree_clf, distributions, random_state=42)
clf = GridSearchCV(tree_clf, distributions)

search = clf.fit(train_X, train_Y)

search.best_params_

tree_clf = DecisionTreeClassifier(**search.best_params_)
```


```python
tree_clf.fit(train_X, train_Y)
```




    DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=8)




```python
eval_pred = tree_clf.predict(eval_X)
```


```python
confusion_matrix(eval_Y, eval_pred)
```




    array([[83,  3],
           [ 3, 48]], dtype=int64)




```python
df = pd.DataFrame(classification_report(eval_Y, eval_pred, output_dict=True))
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>B</th>
      <th>M</th>
      <th>accuracy</th>
      <th>macro avg</th>
      <th>weighted avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precision</th>
      <td>0.965116</td>
      <td>0.941176</td>
      <td>0.956204</td>
      <td>0.953146</td>
      <td>0.956204</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.965116</td>
      <td>0.941176</td>
      <td>0.956204</td>
      <td>0.953146</td>
      <td>0.956204</td>
    </tr>
    <tr>
      <th>f1-score</th>
      <td>0.965116</td>
      <td>0.941176</td>
      <td>0.956204</td>
      <td>0.953146</td>
      <td>0.956204</td>
    </tr>
    <tr>
      <th>support</th>
      <td>86.000000</td>
      <td>51.000000</td>
      <td>0.956204</td>
      <td>137.000000</td>
      <td>137.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#To show overfitting and underfitting, with lower depth we should get underfitting 
    #and with higher depth we should get overfitting
complexity_values = range(1,10)

train_accuracies = []
test_accuracies = []
train_accuracies_recall = []
test_accuracies_recall = []

for complexity_value in complexity_values:
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=complexity_value, 
                                 random_state=42)
    
    clf.fit(train_X, train_Y)
    
    y_pred = clf.predict(eval_X)
    test_accuracies.append(accuracy_score(eval_Y, y_pred))
    test_accuracies_recall.append(recall_score(eval_Y, y_pred, pos_label="M"))
    
    y_pred = clf.predict(train_X)
    train_accuracies_recall.append(recall_score(train_Y, y_pred, pos_label="M"))
    train_accuracies.append(accuracy_score(train_Y, y_pred))

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Recall and Accuracy against complexity level in train and test sets')
line1, = ax1.plot(complexity_values, test_accuracies_recall,label='test_accuracies')
line2, = ax1.plot(complexity_values, train_accuracies_recall,label='train_accuracies')
    
line1, = ax2.plot(complexity_values, test_accuracies,label='test_accuracies')
line2, = ax2.plot(complexity_values, train_accuracies,label='train_accuracies')

ax1.set_xlabel("max_depth")
ax2.set_xlabel("max_depth")

ax1.set_ylabel("Recall")
ax2.set_ylabel("Accuracy")
ax1.legend((line1, line2), ('test', 'train'))
ax2.legend((line1, line2), ('test', 'train'))
fig.tight_layout()
plt.show()
```


    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_35_0.png)
    



```python
# We can see underfiting when max_depth <=2 where train and test 
    #are initially very close and training accuracy is low
# We can see overfitting when test accuracy decreases after 
    #max_depth = 3 and training accuracy keeps increasing
```

###### Exporting tree to a visual format


```python
f = open("hw_1_dt.dot", 'w')
export_graphviz(
    tree_clf,
    out_file=f,
    feature_names = train_set.drop([0, 1], axis = 1).columns,
    class_names= train_set[1].unique(),
    rounded = True,
    filled = True
)
```


```python
from graphviz import Source
path = 'hw_1_dt.dot'
s = Source.from_file(path)
s.view()
```




    'hw_1_dt.dot.pdf'




```python

```

#### KNN


```python
neigh = KNeighborsClassifier()
```


```python
distributions = dict(n_neighbors = range(3, 10),
                    weights = ["distance", "uniform"]
)
```


```python
# clf = RandomizedSearchCV(tree_clf, distributions, random_state=42)
clf = GridSearchCV(neigh, distributions)
```


```python
search = clf.fit(train_X_norm, train_Y)
```


```python
search.best_params_
```




    {'n_neighbors': 5, 'weights': 'distance'}




```python
neigh = KNeighborsClassifier(**search.best_params_)
```


```python
neigh.fit(train_X_norm, train_Y)
```




    KNeighborsClassifier(weights='distance')




```python
eval_pred = neigh.predict(eval_X_norm)
```


```python
confusion_matrix(eval_Y, eval_pred)
```




    array([[86,  0],
           [ 3, 48]], dtype=int64)




```python
df = pd.DataFrame(classification_report(eval_Y, eval_pred, output_dict=True))
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>B</th>
      <th>M</th>
      <th>accuracy</th>
      <th>macro avg</th>
      <th>weighted avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precision</th>
      <td>0.966292</td>
      <td>1.000000</td>
      <td>0.978102</td>
      <td>0.983146</td>
      <td>0.978840</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>1.000000</td>
      <td>0.941176</td>
      <td>0.978102</td>
      <td>0.970588</td>
      <td>0.978102</td>
    </tr>
    <tr>
      <th>f1-score</th>
      <td>0.982857</td>
      <td>0.969697</td>
      <td>0.978102</td>
      <td>0.976277</td>
      <td>0.977958</td>
    </tr>
    <tr>
      <th>support</th>
      <td>86.000000</td>
      <td>51.000000</td>
      <td>0.978102</td>
      <td>137.000000</td>
      <td>137.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#To show overfitting and underfitting, with less number of neighbors underfitting 
#happens and with too many neighbors overfitting occurs
complexity_values = range(1,20)

train_accuracies = []
test_accuracies=[]
train_accuracies_recall = []
test_accuracies_recall =[]

for complexity_value in complexity_values:
    clf = KNeighborsClassifier(n_neighbors=complexity_value, weights="uniform")
    clf.fit(train_X_norm, train_Y)
    y_pred = clf.predict(eval_X_norm)
    test_accuracies.append(accuracy_score(eval_Y, y_pred))
    test_accuracies_recall.append(recall_score(eval_Y, y_pred, pos_label="M"))
    
    y_pred = clf.predict(train_X_norm)
    train_accuracies_recall.append(recall_score(train_Y, y_pred, pos_label="M"))
    train_accuracies.append(accuracy_score(train_Y, y_pred))

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Recall and Accuracy against complexity level in train and test sets')
line1, = ax1.plot(complexity_values, test_accuracies_recall,label='test_accuracies')
line2, = ax1.plot(complexity_values, train_accuracies_recall,label='train_accuracies')
    
line1, = ax2.plot(complexity_values, test_accuracies,label='test_accuracies')
line2, = ax2.plot(complexity_values, train_accuracies,label='train_accuracies')

ax1.set_xlabel("Neighbors")
ax2.set_xlabel("Neighbors")

ax1.set_ylabel("Recall")
ax2.set_ylabel("Accuracy")
ax1.legend((line1, line2), ('test', 'train'))
ax2.legend((line1, line2), ('test', 'train'))
fig.tight_layout()
plt.show()
```


    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_53_0.png)
    



```python
#As number of neighbors increases the model is underfitting
#As the number of neighbors is low the model overfits on the data
```

#### Logistic Regression


```python
log_reg = LogisticRegression()
```


```python
distributions = dict(C= [1e-4, 1e-5, 0.001, 0.01, 0.1, 1, 10, 100, 1000], 
                    tol = [1e-3, 1e-4, 1e-2],
                    penalty = [
                        "l1",
                        "l2"],
                    solver=[
                        "lbfgs",
                        "liblinear",
                        "newton-cg"
                    ],
)
```


```python
# clf = RandomizedSearchCV(tree_clf, distributions, random_state=42)
clf = GridSearchCV(log_reg, distributions)
```


```python
search = clf.fit(train_X_norm, train_Y)
```


```python
best_params = copy.deepcopy(search.best_params_)
best_params
```




    {'C': 100, 'penalty': 'l1', 'solver': 'liblinear', 'tol': 0.01}




```python
log_reg = LogisticRegression(**search.best_params_)
```


```python
log_reg.fit(train_X_norm, train_Y)
```




    LogisticRegression(C=100, penalty='l1', solver='liblinear', tol=0.01)




```python
eval_pred = log_reg.predict(eval_X_norm)
```


```python
confusion_matrix(eval_Y, eval_pred)
```




    array([[84,  2],
           [ 1, 50]], dtype=int64)




```python
df = pd.DataFrame(classification_report(eval_Y, eval_pred, output_dict=True))
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>B</th>
      <th>M</th>
      <th>accuracy</th>
      <th>macro avg</th>
      <th>weighted avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precision</th>
      <td>0.988235</td>
      <td>0.961538</td>
      <td>0.978102</td>
      <td>0.974887</td>
      <td>0.978297</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.976744</td>
      <td>0.980392</td>
      <td>0.978102</td>
      <td>0.978568</td>
      <td>0.978102</td>
    </tr>
    <tr>
      <th>f1-score</th>
      <td>0.982456</td>
      <td>0.970874</td>
      <td>0.978102</td>
      <td>0.976665</td>
      <td>0.978144</td>
    </tr>
    <tr>
      <th>support</th>
      <td>86.000000</td>
      <td>51.000000</td>
      <td>0.978102</td>
      <td>137.000000</td>
      <td>137.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#To show overfitting and underfitting, with less C underfitting happens 
    #and with too much C overfitting occurs
complexity_values = [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000]

train_accuracies = []
test_accuracies=[]

train_accuracies_recall = []
test_accuracies_recall =[]

log_reg_params = search.best_params_

for complexity_value in complexity_values:
    log_reg_params["C"] = complexity_value
    clf = LogisticRegression(**log_reg_params)
    clf.fit(train_X_norm, train_Y)
    y_pred = clf.predict(eval_X_norm)
    test_accuracies.append(accuracy_score(eval_Y, y_pred))
    test_accuracies_recall.append(recall_score(eval_Y, y_pred, pos_label="M"))
    
    y_pred = clf.predict(train_X_norm)
    train_accuracies_recall.append(recall_score(train_Y, y_pred, pos_label="M"))
    train_accuracies.append(accuracy_score(train_Y, y_pred))
complexity_values = [math.log(x, 10) for x in complexity_values]
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Recall and Accuracy against complexity level in train and test sets')
line1, = ax1.plot(complexity_values, test_accuracies_recall,label='test_accuracies')
line2, = ax1.plot(complexity_values, train_accuracies_recall,label='train_accuracies')
    
line1, = ax2.plot(complexity_values, test_accuracies,label='test_accuracies')
line2, = ax2.plot(complexity_values, train_accuracies,label='train_accuracies')

ax1.set_xlabel("C")
ax2.set_xlabel("C")

ax1.set_ylabel("Recall")
ax2.set_ylabel("Accuracy")
ax1.legend((line1, line2), ('test', 'train'))
ax2.legend((line1, line2), ('test', 'train'))
fig.tight_layout()
plt.show()
```


    
![png](Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_files/Breast%20Cancer%20Detection%20using%20DT%2C%20LR%2C%20KNN_67_0.png)
    



```python
#As C increases the lambda decreases and we let the model 
    #overfit as the penalty is lower. The 
#When C is low there is a heavy penalty and the model 
    #underfits our data and it has lower accuracy.
```

#### Selected the best model as logistic as it has the best recall for our malignant class and also the best accuracy.


```python
#Selecting best parameters gotten from grid search
best_params
```




    {'C': 100, 'penalty': 'l1', 'solver': 'liblinear', 'tol': 0.01}



#### Best selected model and using both training and evaluation as training set and testing on the test set.


```python
log_reg = LogisticRegression(**best_params)
```


```python
log_reg.fit(train_eval_X_norm, train_eval_set_Y)
```




    LogisticRegression(C=100, penalty='l1', solver='liblinear', tol=0.01)




```python
y_pred = log_reg.predict(test_X_norm)
```


```python
df = pd.DataFrame(classification_report(eval_Y, eval_pred, output_dict=True))
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>B</th>
      <th>M</th>
      <th>accuracy</th>
      <th>macro avg</th>
      <th>weighted avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precision</th>
      <td>0.988235</td>
      <td>0.961538</td>
      <td>0.978102</td>
      <td>0.974887</td>
      <td>0.978297</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.976744</td>
      <td>0.980392</td>
      <td>0.978102</td>
      <td>0.978568</td>
      <td>0.978102</td>
    </tr>
    <tr>
      <th>f1-score</th>
      <td>0.982456</td>
      <td>0.970874</td>
      <td>0.978102</td>
      <td>0.976665</td>
      <td>0.978144</td>
    </tr>
    <tr>
      <th>support</th>
      <td>86.000000</td>
      <td>51.000000</td>
      <td>0.978102</td>
      <td>137.000000</td>
      <td>137.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
'''
We begin our classification by splitting the data into train, validation and test sets.
We continue by creating normalizing sets of our sets of data.
Exploring the dataset, I checked if the features had an even normal distribution 
    or were skewed in some direction with outliers in data.
This can be used for feature selection in knn, as knn needs feature selection.
Further looking at the box plots for the different classes we could identify 
    three good features and use them to visually depict the classification.
Searching for the best hyperparameters in all the three models by training and 
    using the validation dataset for testing.
Out of the three models we select the one with the best recall as we care more 
    about the recall of the malignant class.
This gave us logistic regression as the best model.
We also varied our parameters to check for underfitting and overfitting.
The goodness of our model is defined by the malignant class recall.
Finally we combined our training and validation datasets to predict on the 
    test set which gave us 96% recall. 
'''
```




    '\nWe begin our classification by splitting the data into train, validation and test sets.\nWe continue by creating normalizing sets of our sets of data.\nExploring the dataset, I checked if the features had an even normal distribution \n    or were skewed in some direction with outliers in data.\nThis can be used for feature selection in knn, as knn needs feature selection.\nFurther looking at the box plots for the different classes we could identify \n    three good features and use them to visually depict the classification.\nSearching for the best hyperparameters in all the three models by training and \n    using the validation dataset for testing.\nOut of the three models we select the one with the best recall as we care more \n    about the recall of the malignant class.\nThis gave us logistic regression as the best model.\nWe also varied our parameters to check for underfitting and overfitting.\nThe goodness of our model is defined by the malignant class recall.\nFinally we combined our training and validation datasets to predict on the \n    test set which gave us 96% recall. \n'




```python

```


```python

```
