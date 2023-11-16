
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statistics
from geopy.distance import geodesic


df_train = pd.read_csv('/content/train.csv')
df_train
df_train.info()
df_train['Festival'].value_counts()
df_train.describe()
df_train.describe(exclude = np.number).T
df_train.rename(columns={'Weatherconditions':'Weather_conditions'},inplace = True)

df_train['Weather_conditions'] = df_train['Weather_conditions'].apply(lambda x: x.split()[1])

df_train['Weather_conditions'] 
df_train['Time_taken(min)'] = df_train['Time_taken(min)'].apply(lambda x: int(x.split()[1]))

df_train['city_code']

df_train['Delivery_person_Age'] = df_train['Delivery_person_Age'].astype('float64')
df_train['Delivery_person_Ratings'] = df_train['Delivery_person_Ratings'].astype('float64')
df_train['multiple_deliveries'] = df_train['multiple_deliveries'].astype('float64')
df_train['Order_Date'] = pd.to_datetime(df_train['Order_Date'],format = '%d-%M-%Y')
df_train.drop(['ID','Delivery_person_ID'],axis = 1, inplace = True)


df_train.duplicated().value_counts()

df_train.replace('NaN',np.nan, regex = True,inplace = True)
df_train.isnull().sum().sort_values(ascending = False)

df_train['Weather_conditions'].value_counts()

cols = ['Delivery_person_Age','Delivery_person_Ratings','Weather_conditions','Road_traffic_density','multiple_deliveries','Festival','City']
fig , axes = plt.subplots(4,2,figsize=(20,15))
for i, j in enumerate(cols):
  row = i//2
  col = i % 2
  ax = axes[row,col]
  sns.countplot(data = df_train, x = j,order=df_train[j].value_counts().sort_index().index, ax=ax)
  ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


df_train['Delivery_person_Age'].fillna(np.random.choice(df_train['Delivery_person_Age']),inplace = True)
df_train['Weather_conditions'].fillna(np.random.choice(df_train['Weather_conditions']),inplace = True)
df_train['City'].fillna(df_train['City'].mode()[0],inplace = True)
df_train['Festival'].fillna(df_train['Festival'].mode()[0],inplace = True)
df_train['multiple_deliveries'].fillna(df_train['multiple_deliveries'].mode()[0],inplace = True)
df_train['Delivery_person_Ratings'].fillna(df_train['Delivery_person_Ratings'].median(),inplace = True)
df_train['Road_traffic_density'].fillna(df_train['Road_traffic_density'].mode()[0],inplace = True)



df_train['day'] = df_train['Order_Date'].dt.day


df_train['month'] = df_train['Order_Date'].dt.month
df_train['quarter'] = df_train['Order_Date'].dt.quarter
df_train['year'] = df_train['Order_Date'].dt.year
df_train['day_of_week'] = df_train['Order_Date'].dt.day_of_week.astype(int)
df_train['is_month_start'] = df_train['Order_Date'].dt.is_month_start.astype(int)
df_train['is_month_end'] = df_train['Order_Date'].dt.is_month_end.astype(int)
df_train['is_quarter_start'] = df_train['Order_Date'].dt.is_quarter_start.astype(int)
df_train['is_quarter_end'] = df_train['Order_Date'].dt.is_quarter_end.astype(int)
df_train['is_year_start'] = df_train['Order_Date'].dt.is_year_start.astype(int)
df_train['is_year_end'] = df_train['Order_Date'].dt.is_year_end.astype(int)
df_train['is_weekend'] = np.where(df_train['day_of_week'].isin([5,6]),1,0)


def calculate_time_diff(df_train):
    # Find the difference between ordered time & picked time
    df_train['Time_Orderd'] = pd.to_timedelta(df_train['Time_Orderd'])
    df_train['Time_Order_picked'] = pd.to_timedelta(df_train['Time_Order_picked'])
    df_train['Time_Order_picked_formatted'] = df_train['Order_Date'] + np.where(df_train['Time_Order_picked'] < df_train['Time_Orderd'], pd.DateOffset(days=1), pd.DateOffset(days=0)) + df_train['Time_Order_picked']
    df_train['Time_Ordered_formatted'] = df_train['Order_Date'] + df_train['Time_Orderd']
    df_train['order_prepare_time'] = (df_train['Time_Order_picked_formatted'] - df_train['Time_Ordered_formatted']).dt.total_seconds() / 60
    
    # Handle null values by filling with the median
    df_train['order_prepare_time'].fillna(df_train['order_prepare_time'].median(), inplace=True)
    
    # Drop all the time & date related columns
    df_train.drop(['Time_Orderd', 'Time_Order_picked', 'Time_Ordered_formatted', 'Time_Order_picked_formatted', 'Order_Date'], axis=1, inplace=True)
calculate_time_diff(df_train)

from geopy.distance import geodesic
df_train['distance'] = np.zeros(len(df_train))
restaurant_coordinates = df_train[['Restaurant_latitude','Restaurant_longitude']].to_numpy()
dilevery_location_coordinates =df_train[['Delivery_location_latitude','Delivery_location_longitude']].to_numpy()
df_train['distance'] = np.array([geodesic(restaurent,delivery) for restaurent, delivery in zip(restaurant_coordinates,dilevery_location_coordinates)])

df_train['distance'] = df_train['distance'].astype('str').apply(lambda x: x.split()[0])


df_train['distance'].astype(float)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


df_train['Road_traffic_density'] = encoder.fit_transform(df_train['Road_traffic_density'])
df_train['Type_of_order']= encoder.fit_transform(df_train['Type_of_order'])
df_train['Weather_conditions'] = encoder.fit_transform(df_train['Weather_conditions'])
df_train['Type_of_vehicle'] = encoder.fit_transform(df_train['Type_of_vehicle'])
df_train['Festival'] = encoder.fit_transform(df_train['Festival'])
df_train['City'] = encoder.fit_transform(df_train['City'])
df_train['city_code'] = encoder.fit_transform(df_train['city_code'])

x = df_train.drop('Time_taken(min)', axis =1)
y = df_train['Time_taken(min)']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

models = [ LinearRegression(),DecisionTreeRegressor(), RandomForestRegressor(), XGBRegressor()]
param = [
    {},
    {'max_depth' : [3,5,7]},
    {'n_estimators' : [100,200,300]},
    {'n_estimators' : [20,25,30],'max_depth' : [5,7,9]}

]

from sklearn.model_selection import GridSearchCV

param_grid = [
    {},  
    {'max_depth': [3, 5, 7]},
    {'n_estimators': [100, 200, 300]},
    {'n_estimators': [20, 25, 30], 'max_depth': [5, 7, 9]},
]

for i, model in enumerate(models):
    grid_search = GridSearchCV(model, param_grid[i], cv=5, scoring='r2')
    grid_search.fit(x_train, y_train)

    print(f"{model.__class__.__name__}:")
    print("Best parameters:", grid_search.best_params_)
    print("Best R2 score:", grid_search.best_score_)
    print()


model = XGBRegressor(n_estimators = 20, max_depth = 9)
model.fit(x_train, y_train)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(mae,mse,rmse,r2)