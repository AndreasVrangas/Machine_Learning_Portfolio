import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
'Import Dataset'
names=['Relative_Compactness','Surface_Area','Wall_Area','Roof_Area','Overall_Height','Orientation','Glazing_Area','Glazing_Area_Distribution','Heating_Load','Cooling_Load']
df = pd.read_excel('C:/Users/Riko/OneDrive/Portfolio stuff/3.Energy/ENB2012_data.xlsx',names=names)



##Scale features using StandardScaler (-1>=x>=1)
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
#Aplly scaling
df = pd.DataFrame(stdsc.fit_transform(df.values),index = df.index, columns = df.columns)

#Create target value (Heating_Load)
Y = np.array(df['Heating_Load'])
#Features
X = df.drop(df[['Heating_Load','Cooling_Load']],axis = 1)


#Check correlation of all variables against Heading Load
corr_cool_load = df.corr().Heating_Load.sort_values(ascending=False)

##All variables


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

#Function to fit, predict and return evaluation metrics. It also returns fetarue importance plots for the ensembel method
def pred_ensemble(X,Y,model):
    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
    model.fit(X_train,y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test,preds))*100
    r2 = r2_score(y_test,preds)
    importances = model.feature_importances_
    plt.figure()
    plt.title("Feature importances ")
    plt.bar(range(X_train.shape[1]), importances,
       color="g",align="center")
    plt.xticks(range(X_train.shape[1]), X_train.columns)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout
    plt.xticks(rotation = 90 )
    plt.show()

    return rmse, r2,
    
rmse_rf,r2_rf = pred_ensemble(X,Y,RandomForestRegressor(random_state = 22))

##Function to fit, predict and return evaluation metrics. It also returns coefficinet magnitude plots for the linear methods

def pred_linear (X,Y,model):
    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
    model.fit(X_train,y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test,preds))*100
    r2 = r2_score(y_test,preds)
    coefs = pd.Series(model.coef_, index = X_train.columns)
    coefs.plot(kind = "bar")
    plt.title("Model Coefficients ")
    plt.xticks(rotation = 270)
    plt.tight_layout()
    plt.show()
    return rmse, r2
LR = LinearRegression()

rmse_lr,r2_lr = pred_linear(X,Y,LR)
    
    
