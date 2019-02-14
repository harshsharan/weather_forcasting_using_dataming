import os
import pandas as pd
os.chdir("C:/Users/Dell/Documents/Bikesharing")
os.getcwd()
data=pd.read_csv("day.csv",sep=',')
data.head()
data.tail()
data.rename(columns={'yr':'year','mnth':'month','cnt':'count','dteday':'dateday'},inplace=True)
data.shape
data.isnull().sum()
data.columns
continous = ['temp' , 'atemp' , 'hum' , 'windspeed' , 'casual' , 'registered' , 'count']
continous
#########################
data[continous].describe()
#########################
for i , name in enumerate(continous):
    plt.figure(figsize=(15,20))
    plt.subplots_adjust(hspace=0.7, wspace=0.7)
    b = ''    
    b += name
    plt.figure()    
    plt.title('Box plot of {}'.format(b))
    sns.boxplot(y = data[b])    
    plt.show()
 
 
 
##########################
data[ (data['instant']>46)& (data['instant']<53)]
################
data['hum'].replace(0.187917, 0.507463, inplace=True)
data['windspeed'].replace(0.507463, 0.187917, inplace=True)
 
 
 
 
 
 
 
 
 
 
 
mean_humidity=data[(data['instant'] >  62) & (data['instant'] < 75)]
#################
mean_humidity['hum'].describe()
########
data['hum'].replace(0,60250, inplace=True)
#####################
plt.figure(figsize=(10, 10))
plt.subplots_adjust(hspace=0.7, wspace=0.7)
plt.subplot(221)
sns.boxplot(y = data['hum'],color = 'brown')
plt.ylabel('humidity')
plt.title('plot of humidity')
 
plt.subplot(222)
sns.boxplot(y = data['windspeed'], color = 'green')
plt.ylabel('windspeed')
plt.title('plot of windspeed')
########################
plt.figure(figsize=(14, 15))
plt.subplots_adjust(hspace=0.7, wspace=0.7)
plt.subplot(441)
sns.distplot(data['temp'], hist=False, rug=True,color='green')
plt.ylabel('Density')
plt.title('Temperature Distribution')
 
plt.subplot(442)
sns.distplot(data['windspeed'], hist=False, rug=True,color='green')
plt.ylabel('Density')
plt.title('windspeed Distribution')
 
plt.subplot(443)
sns.distplot(data['hum'], hist=False, rug=True,color='green')
plt.ylabel('Density')
plt.title('Humidity Distribution')
#################################
plt.figure(figsize=(14, 15))
plt.subplots_adjust(hspace=0.7, wspace=0.7)
 
plt.subplot(441)
sns.distplot(data['count'], hist=False, rug=True,color='green')
plt.ylabel('density')
plt.title('count')
######################
data.columns
###########
data['season'].value_counts()
###############
plt.figure(figsize=(26,26))
plt.subplot(441)
plt.subplots_adjust(hspace=0.7, wspace=0.7)
#plt.ylabel('Density')
plt.title('workingdays Distribution')
sns.countplot(data['workingday'],color='brown',order = data['workingday'].value_counts().index)
 
#plt.xticks(rotation=45)
 
plt.subplot(442)
#plt.ylabel('Density')
plt.title('weathersit Distribution')
sns.countplot(data['weathersit'],color='green',order=data['weathersit'].value_counts().index)
 
plt.subplot(443)
#plt.ylabel('Density')
plt.title('holiday Distribution')
sns.countplot(data['holiday'],color='orange',order=data['holiday'].value_counts().index)
#####################
data.columns
###############
data['dateday']=data['dateday'].astype('datetime64')
###################
data_30 = data[:24]
#################
plt.figure(figsize=(16,6))
plt.plot(data_30['dateday'],data_30['count'])
plt.xlabel('date')
plt.ylabel('count')
plt.title('Date(24days) vs count')
#################
import matplotlib.pyplot as plt
plt.figure(figsize=(16,6))
plt.plot(data['dateday'],data['casual'])
plt.plot(data['dateday'],data['registered'],color = 'orange')
plt.xlabel('date')
plt.ylabel('Ride_count')
plt.title('Date vs Bicycle ride count')
plt.legend()
######################
plt.figure(figsize=(16,6))
plt.plot(data['dateday'],data['registered'],color = 'orange')
plt.xlabel('date')
plt.ylabel('Registered')
plt.title('Date vs Registered')
###################
plt.figure(figsize=(16,6))
plt.plot(data['dateday'],data['count'],color = 'green')
plt.xlabel('date')
plt.ylabel('Total_count')
plt.title('Date vs Total_count')
#######################
month_casual = data.groupby(['year','month'])['casual','registered'].mean()
month_casual1 = data.groupby(['weekday'])['casual','registered','count'].mean()
#########################
month_casual1
#################
month_casual = month_casual.reset_index()
####################
x = month_casual.reset_index()
x.drop(['year','month','index'],axis=1,inplace=True)
#######################
x.plot(title='Avg use by month')
plt.xlabel('Month')
plt.ylabel('Avgerage count')
#######################
month_casual1 = data.groupby(['weekday'])['casual','registered'].mean()
###############
month_casual1.plot(title='Average use by day')
##############################
data['week_day'] = (data['weekday'] > 5) & (data['weekday'] == 0)
data['week_day'] =  (data['weekday'] >= 1) & (data['weekday'] <= 5)
#################################
x =[]
for i in data['week_day']:
    if  i is False:
        x.append(1)  
    else:  
        x.append(0)
#################3333
data['week_end'] = x
#########################
data_weekbday = data.pivot_table('count',aggfunc='sum',index=['year','month'],columns='week_day')
######################
data_weekday  = data_weekbday.reset_index()
data_weekday.drop(['year','month'],axis=1,inplace=True)
#########################
data_weekday.plot()
plt.xlabel('Month')
plt.ylabel('Total Count')
plt.title('(Weekday) Month vs Total Count')
##################
casual_pattern_weekday = data.pivot_table('casual',aggfunc='sum',index=['year','month'],columns='week_day')
#########################
casual_pattern_weekday = casual_pattern_weekday.reset_index()
casual_pattern_weekday.drop(['year','month'],axis=1,inplace=True)
casual_pattern_weekday.plot()
plt.xlabel('Month')
plt.ylabel('casual_count')
plt.title('Month vs Casual')
##############################
data.columns
######################
holidays = data.pivot_table('count',aggfunc='mean',index=['month'],columns='holiday')
#####################
holidays.dropna(inplace = True)
####################
holidays = holidays.sort_values(by=1, ascending =False)
#################
holidays = holidays.reset_index()
################
holidays
#####################
sns.barplot(holidays.month,holidays[1],color="green",order=holidays.month)
plt.xlabel('Holiday Month')
plt.ylabel('Count')
plt.title('Holiday Month vs Count')
##########################
sns.violinplot(data['weathersit'],data['count'])
plt.title('Violin plot of weathersit vs count')
#######################
sns.violinplot(data['season'],data['count'])
plt.title('Violin plot of season vs count')
#############################
sns.lmplot('temp','count',data=data)
plt.xlabel('Temperature')
plt.ylabel('Count')
plt.title('Temperature vs Total Count')
#######################
data.groupby('season')['temp'].mean()
###################
t = sns.lmplot(x="temp", y="count", hue="season", col="season",data=data, aspect=.4, x_jitter=.1)
####################
sns.lmplot('hum','count',data=data)
plt.xlabel('Humidity')
plt.ylabel('Count')
plt.title('Humidity vs Count')
###############
data.groupby('season')['hum'].median()
###################
t = sns.lmplot(x='hum', y="count", hue="season", col="season",data=data, aspect=.4, x_jitter=.1)
####################
data.columns
#####################
data['week_day'].value_counts()
####################
names =  ['season', 'year', 'month', 'holiday', 'weekday','workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed','count']
########################
co = data[names].corr()
   
correlation = co
#plt.figure(figsize=(10,10))
plt.figure(figsize = (20,20))
g = sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix',xticklabels=True,yticklabels=True)
g.set_yticklabels(g.get_yticklabels(), rotation =0)
g.set_xticklabels(g.get_yticklabels(), rotation =90)
plt.title('Correlation between different fearures')
#######################
#model
data.columns
#################
x=pd.DatetimeIndex(data['dateday'])
###################
x = x.day
data['day'] = x
##################
columns = ['dateday','day','season', 'year', 'month', 'holiday', 'weekday','workingday', 'weathersit', 'temp', 'hum', 'windspeed', 'count']
###############
data_train = data[columns]
################
weather_sit = pd.get_dummies(data_train['weathersit'],prefix='weathersit')
seasons_dummy = pd.get_dummies(data_train['season'],prefix='season')
weekday_dummy = pd.get_dummies(data_train['weekday'],prefix='weekday')
##################
weather_sit.drop(['weathersit_3'],axis=1,inplace=True)
seasons_dummy.drop(['season_4'],axis=1,inplace=True)
weekday_dummy.drop(['weekday_6'],axis=1,inplace=True)
#####################
def nomalizar(x):
    norm  = (x-min(x))/(max(x)-min(x))    
    return norm
################
c = nomalizar(data['month'])
##############
data_train['month_norm'] = c
#######################
data_train1 = pd.concat([data_train,weather_sit,seasons_dummy,weekday_dummy],axis=1)
###################
data_train1.columns
######################
names_columns =['temp', 'hum', 'windspeed','workingday','holiday',        
                 'weathersit_1', 'weathersit_2', 'season_1', 'season_2',      
                 'season_3', 'weekday_0', 'weekday_1', 'weekday_2', 'weekday_3',      
                 'weekday_4', 'weekday_5']
###############################
test = data_train1.iloc[710:,:]
################
train = data_train1.iloc[:710,:]
################
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
###################
Reg = LinearRegression()
###################
Reg.fit(train[names_columns],train['count'])
###############
y_pred = Reg.predict(test[names_columns])
################
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
######################
np.sqrt(mean_squared_error(test['count'],y_pred))
#############################
r2_score(test['count'],y_pred)
#####################
importance = Reg.coef_
 
plt.figure(figsize=(20,20))
plt.subplots_adjust(hspace=0.7)
plt.subplot2grid((3,2),(0,1))
importance = pd.DataFrame({'importance':importance,'feature':names_columns})
importance = importance.sort_values(by ='importance',ascending=False)
sns.barplot(importance['importance'],importance['feature'])
#plt.xticks(rotation =90)
plt.title('LR feature importance')
plt.show()
############################
plt.figure(figsize=(10,3))
plt.plot(test['day'],test['count'],label='test')
plt.plot(test['day'],y_pred,label='predicted')
plt.legend()
plt.xlabel('Days')
plt.ylabel('count')
plt.title('LR Actual vs Predicted')
plt.show()
########################
data.columns
###################
columns
#################
columns =['season', 'holiday', 'weekday',       'workingday', 'weathersit', 'temp', 'hum', 'windspeed']
####################
test = data.iloc[710:,:]
train = data.iloc[:710, :]
###############
from sklearn.svm import LinearSVR
###################
Reg = LinearSVR()
#################
param_grid2 = {'C':[0.05,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,9,10,11,12,13,14,15,16,17]          
                            }
grid_search = GridSearchCV(Reg, param_grid=param_grid2,cv=3)
best_model= grid_search.fit(train[columns],train['count'])
#######################
best_model.best_params_
###################
Reg = LinearSVR(C=17)
################
Reg.fit(train[columns],train['count'])
y_pred = Reg.predict(test[columns])
######################
print(np.sqrt(mean_squared_error(test['count'],y_pred)))
print(r2_score(test['count'],y_pred))
#########################
importance= Reg.coef_
importance = pd.DataFrame({'importance':importance,'feature':columns})
importance = importance.sort_values(by ='importance',ascending=False)
importance
###########################
sns.barplot(importance['importance'],importance['feature'],color='blue')
plt.title('SVM feature importance')
plt.show()
####################
plt.figure(figsize=(10,3))
plt.plot(test['day'],test['count'],label='test',color='red')
plt.plot(test['day'],y_pred,label='predicted')
plt.legend()
plt.xlabel('Days')
plt.ylabel('Count')
plt.title('(SVM) Actual vs Predicted')
plt.show()
##########################################
from sklearn.tree import DecisionTreeRegressor
#################
Reg = DecisionTreeRegressor()
######################
param_grid2 = {'max_depth':[None,3,5,6,8,9,10,12,15,17,18,20],            
               # "min_samples_split": [2,3,4,5,6,7,8,10,15,20]                
               #"min_samples_leaf": [1,2,3,4,5,10,30]              
              }
grid_search = GridSearchCV(Reg, param_grid=param_grid2,cv=3)
best_model= grid_search.fit(train[columns],train['count'])
##################################
best_model.best_params_
######################
best_model.best_params_
##################
best_model.best_params_
####################
Reg = DecisionTreeRegressor(max_depth=6,min_samples_split=4,min_samples_leaf=1)
################################
Reg.fit(train[columns],train['count'])
y_pred = Reg.predict(test[columns])
#####################
print(np.sqrt(mean_squared_error(test['count'],y_pred)))
print(r2_score(test['count'],y_pred))
######################
importance= Reg.feature_importances_
importance = pd.DataFrame({'importance':importance,'feature':columns})
importance = importance.sort_values(by ='importance',ascending=False)
importance
##########################
sns.barplot(importance['importance'],importance['feature'],color='grey')
plt.title('Decision Tree Feature Imp')
plt.show()
####################################
plt.figure(figsize=(10,3))
plt.plot(test['day'],test['count'],label='test',color='black')
plt.plot(test['day'],y_pred,label='predicted',color='red')
plt.legend()
plt.xlabel('Days')
plt.ylabel('Count')
plt.title('(Decision tree) Actual vs Predicted')
plt.show()
##############################
from sklearn.ensemble import AdaBoostRegressor
########################
Reg = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=6))
########################
param_grid2 = {"n_estimators": [10,30,50,100,150,200,250,280,300,350,400,450,500]
               
              }
grid_search = GridSearchCV(Reg, param_grid=param_grid2,cv=3)
best_model= grid_search.fit(train[columns],train['count'])
############################
best_model.best_params_
####################
Reg = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=6),n_estimators=150)
###################
Reg.fit(train[columns],train['count'])
y_pred = Reg.predict(test[columns])
############################
print(np.sqrt(mean_squared_error(test['count'],y_pred)))
print(r2_score(test['count'],y_pred))
###############################
plt.figure(figsize=(10,3))
plt.plot(test['day'],test['count'],label='test')
plt.plot(test['day'],y_pred,label='predicted',color='green')
plt.legend()
plt.xlabel('Days')
plt.ylabel('Count')
plt.title('(Adaboost) Actual vs Predicted')
#############################
Reg.feature_importances_
########################
importance= Reg.feature_importances_
importance = pd.DataFrame({'importance':importance,'feature':columns})
importance = importance.sort_values(by ='importance',ascending=False)
importance
############################
sns.barplot(importance['importance'],importance['feature'],color='purple')
plt.title('Adboost with Decision Tree Feature Imp')
plt.show()
#####################
data.columns
######################
columns =['season', 'holiday', 'weekday',       'workingday', 'weathersit', 'temp', 'hum', 'windspeed']
############
test = data.iloc[:710,:]
#########################
train = data.iloc[:710,:]
###################
from sklearn.ensemble import RandomForestRegressor
########################
Reg = RandomForestRegressor(n_jobs=-1)
####################
param_grid2 = {#"n_estimators": [10,30,50,100,150,200,250,280,300,350,400,450,500],            
#"max_depth": [None,3,5,6,8,9,10,12,15,17,18,20],              
   
#"min_samples_split": [2,3,4,5,6,7,8,10,15,20],            
     "min_samples_leaf": [1,2,3,4,5,10,30],            
    # "max_leaf_nodes": [None,5,10,20,30, 40],            
    # "max_features": ['auto',0.5,'log2']              
            }
grid_search = GridSearchCV(Reg, param_grid=param_grid2,cv=3)
best_model= grid_search.fit(train[columns],train['count'])
#######################
best_model.best_params_
######################
best_model.best_params_
##########################
Reg = RandomForestRegressor(n_jobs=-1,n_estimators=260,max_depth=10,min_impurity_decrease=4,min_samples_leaf= 1,
                             bootstrap=True)
##################################
Reg.fit(train[columns],train['count'])
#######################
y_pred= Reg.predict(test[columns])
############################
r2_score(test['count'],y_pred)
########################
np.sqrt(mean_squared_error(test['count'],y_pred))
#########################
importance= Reg.feature_importances_
#####################
importance = pd.DataFrame({'importance':importance,'feature':columns})
##################
importance = importance.sort_values(by ='importance',ascending=False)
######################
sns.barplot(importance['importance'],importance['feature'])
plt.title('RF feature imp')
####################
importance
#####################
plt.figure(figsize=(10,3))
plt.plot(test['day'],test['count'],label='test',color='blue')
plt.plot(test['day'],y_pred,label='predicted',color='red')
plt.legend()
plt.xlabel('Days')
plt.ylabel('Count')
plt.title('(RF) Actual vs Predicted')
plt.show()
########################
from sklearn.ensemble import ExtraTreesRegressor
###################
Reg = ExtraTreesRegressor(n_jobs=-1,n_estimators=30,max_depth=12,min_impurity_decrease=3,min_samples_leaf=1)
##########################
param_grid2 = {#"n_estimators": [10,30,50,100,150,200,250,280,300,350,400,450,500],            
#"max_depth": [None,3,5,6,8,9,10,12,15,17,18,20],            
#"min_samples_split": [2,3,4,5,6,7,8,10,15,20],            
#"min_samples_leaf": [1,2,3,4,5,10,30],            
    "max_leaf_nodes": [None,5,10,20,30, 40],            
    #"max_features": ['auto',0.5,'log2']            
      }
grid_search = GridSearchCV(Reg, param_grid=param_grid2,cv=3)
best_model= grid_search.fit(train[columns],train['count'])
#################################
best_model.best_params_
###################
best_model.best_params_
################
Reg = ExtraTreesRegressor(n_jobs=-1,n_estimators=50,max_depth=12,min_impurity_decrease=3,min_samples_leaf=1,                        
                           bootstrap=True)
##################
Reg.fit(train[columns],train['count'])
y_pred= Reg.predict(test[columns])
#######################
r2_score(test['count'],y_pred)
##########################
np.sqrt(mean_squared_error(test['count'],y_pred))
#######################
importance= Reg.feature_importances_
importance = pd.DataFrame({'importance':importance,'feature':columns})
importance = importance.sort_values(by ='importance',ascending=False)
importance
####################
importance = importance.sort_values(by ='importance',ascending=False)
sns.barplot(importance['importance'],importance['feature'],color='orange')
plt.title('ETR feature importance')
plt.show()
###########################
plt.figure(figsize=(15,4))
plt.plot(test['day'],test['count'],label='test',color='red')
plt.plot(test['day'],y_pred,label='predicted',color='brown')
plt.legend()
plt.xlabel('Days')
plt.ylabel('Count')
plt.title('(Extre tree) Actual vs Predicted')
plt.show()
####################################
importance= Reg.feature_importances_
importance = pd.DataFrame({'importance':importance,'feature':columns})
importance = importance.sort_values(by ='importance',ascending=False)
sns.barplot(importance['importance'],importance['feature'],color='brown')
plt.show()
######################
plt.figure(figsize=(10,3))
plt.plot(test['day'],test['count'],label='test')
plt.plot(test['day'],y_pred,label='predicted')
plt.legend()
plt.xlabel('Days')
plt.ylabel('Count')