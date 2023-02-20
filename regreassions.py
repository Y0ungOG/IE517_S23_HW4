import pandas as pd
df = pd.read_csv('//ad.uillinois.edu/engr-ews/chenyim2/Desktop/housing.csv')

#split dataset
from sklearn.model_selection import train_test_split
df.columns
x = df.iloc[:,:-1].values
y = df['MEDV'].values
x_train, x_test , y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

#EDA
#heat map
df.describe()
import matplotlib.pyplot as plt
import seaborn as sns
corr_matrix = df.corr()
sns.heatmap(corr_matrix, cmap='coolwarm')
plt.title('correlation')
plt.show()

#create scatter plot matrix
cols = ['LSTAT', "INDUS", 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols],size=2.5)
plt.tight_layout()
plt.show()

#boxplot
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
sns.boxplot(x='RAD', y='MEDV', data=df, ax=ax[0])
sns.boxplot(x='CHAS', y='MEDV', data=df, ax=ax[1])
sns.boxplot(x='ZN', y='MEDV', data=df, ax=ax[2])
plt.tight_layout()
plt.show()



#create correlation matrix
import numpy as np
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm =sns.heatmap(cm,
                annot = True,
                square = True,
                fmt = '.2f',
                annot_kws = {'size':15},
                yticklabels = cols,
                xticklabels = cols)
plt.show()

# #linear regression model
# class LinearRegressionGD(object):
#     def __init__(self, eta = 0.001, n_iter = 20):
#         self.eta = eta
#         self.n_iter = n_iter
        
#     def fit(self, x, y):
#         self.w_ = np.zeros(1+x.shape[1])
#         self.cost_ = []
        
#         for i in range(self.n_iter):
#             output = self.net_input(x)
#             errors = (y-output)
#             self.w_[1:] += self.eta * x.T.dot(errors)
#             self.w_[0] += self.eta*errors.sum()
#             cost = (errors **2).sum()/2.0
#             self.cost_.append(cost)
#         return self
    
#     def net_input(self, x):
#         return np.dot(x, self.w_[1:] + self.w_[0])
    
#     def predict(self, x):
#         return self.net_input(x)
    
# x = df[['RM']].values
# y = df['MEDV'].values

# from sklearn.preprocessing import StandardScaler
# sc_x = StandardScaler()
# sc_y = StandardScaler()
# x_std = sc_x.fit_transform(x)
# y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()    
# lr = LinearRegressionGD()
# lr.fit(x_std, y_std)
# plt.plot(range(1, lr.n_iter+1), lr.cost_)
# plt.ylabel('SSE')
# plt.xlabel('Epoch')
# plt.show()    
    
# def lin_regplot(x, y, model):
#     plt.scatter(x, y, c='steelblue', edgecolor='white', s=70)
#     plt.plot(x, model.predict(x), color='black', lw=2)    
#     return     
    
# lin_regplot(x_std, y_std, lr)
# plt.xlabel('Average number of rooms [RM] (standardized)')
# plt.ylabel('Price in $1000s [MEDV] (standardized)')    
# plt.show()    


#another way of linear regression
from sklearn.linear_model import LinearRegression
#from different variables(features)
y = df['MEDV'].values
mse_list = []
r2_list = []
features = []

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
for feature in df.columns:
    # Skip the target variable
    if feature == 'MEDV':
        continue
    x = df[[feature]].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    slr = LinearRegression()
    slr.fit(x, y)
    y_pred = slr.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    features.append(feature)
    mse_list.append(mse)
    r2_list.append(r2)

# Create a scatter plot of MSE vs. R-squared for each feature
fig, ax = plt.subplots()
ax.scatter(mse_list, r2_list)

for i, txt in enumerate(features):
    ax.annotate(txt, (mse_list[i], r2_list[i]))

ax.set_xlabel('Mean Squared Error')
ax.set_ylabel('R-squared')
ax.set_title('Performance')
plt.show()

#using RM
x = df[['RM']].values
y = df['MEDV'].values
slr = LinearRegression()
slr.fit(x, y)
y_pred = slr.predict(x)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)


sns.regplot(x, y, slr)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')
plt.show()

#detect outliers   
from sklearn.linear_model import RANSACRegressor
ransac = RANSACRegressor(LinearRegression(), 
                         max_trials=100, 
                         min_samples=50, 
                         loss='absolute_loss', 
                         residual_threshold=5.0, 
                         random_state=0)


ransac.fit(x, y)
    
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(x[inlier_mask], y[inlier_mask],
            c='steelblue', edgecolor='white', 
            marker='o', label='Inliers')
plt.scatter(x[outlier_mask], y[outlier_mask],
            c='limegreen', edgecolor='white', 
            marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='black', lw=2)   
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper left')
plt.show()  

print('Slope: %.3f' % ransac.estimator_.coef_[0])
print('Intercept: %.3f' % ransac.estimator_.intercept_)    

#Evaluate the performance           
slr.fit(x_train, y_train)
y_train_pred = slr.predict(x_train)
y_test_pred = slr.predict(x_test)
    
plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.tight_layout()
plt.show()    
    
#MSE
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))    
    
# Ridge regression:
from sklearn.linear_model import Ridge

#find the best r2 and mse correspond to alpha
mse_list = []
r2_list = []
alpha_list  = np.arange(0,51)
x = df[['RM']].values
y = df['MEDV'].values
x_train, x_test , y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

for i in alpha_list:
    model = Ridge(alpha = i)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mse_list.append(mse)
    r2_list.append(r2)
    
best_mse = np.min(mse_list)
best_r2 = np.max(r2_list)
best_alpha_mse = alpha_list[mse_list.index(best_mse)]
print(best_alpha_mse)
best_r2 = alpha_list[r2_list.index(best_r2)]
print(best_r2)

plt.plot(range(51),r2_list)
plt.title('r2')
plt.xlabel('alpha')
plt.ylabel('score')
plt.show()

ridge = Ridge(alpha=33.0)
ridge.fit(x, y)

# print the intercept and coefficients
print("Intercept:", ridge.intercept_)
print("Coefficients:", ridge.coef_)

#from different variables(features)
y = df['MEDV'].values
mse_list = []
r2_list = []
features = []


for feature in df.columns:
    # Skip the target variable
    if feature == 'MEDV':
        continue
    x = df[[feature]].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    model = Ridge(alpha=1.0)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    features.append(feature)
    mse_list.append(mse)
    r2_list.append(r2)

# Create a scatter plot of MSE vs. R-squared for each feature
fig, ax = plt.subplots()
ax.scatter(mse_list, r2_list)

for i, txt in enumerate(features):
    ax.annotate(txt, (mse_list[i], r2_list[i]))

ax.set_xlabel('Mean Squared Error')
ax.set_ylabel('R-squared')
ax.set_title('Performance')

plt.show()


# LASSO regression:
from sklearn.linear_model import Lasso

mse_list = []
r2_list = []
alpha_list  = np.arange(0,51)
x = df[['RM']].values
y = df['MEDV'].values
x_train, x_test , y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

for i in alpha_list:
    model = Lasso(alpha = i)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mse_list.append(mse)
    r2_list.append(r2)
    
best_mse = np.min(mse_list)
best_r2 = np.max(r2_list)
best_alpha_mse = alpha_list[mse_list.index(best_mse)]
print(best_alpha_mse)
best_r2 = alpha_list[r2_list.index(best_r2)]
print(best_r2)

plt.plot(range(51),r2_list)
plt.title('r2')
plt.xlabel('alpha')
plt.ylabel('score')
plt.show()

lasso = Lasso(alpha=1.0)
lasso.fit(x, y)

# print the intercept and coefficients
print("Intercept:", lasso.intercept_)
print("Coefficients:", lasso.coef_)

#from different variables(features)
y = df['MEDV'].values
mse_list = []
r2_list = []
features = []


for feature in df.columns:
    # Skip the target variable
    if feature == 'MEDV':
        continue
    x = df[[feature]].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    model = Lasso(alpha=1.0)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    features.append(feature)
    mse_list.append(mse)
    r2_list.append(r2)

# Create a scatter plot of MSE vs. R-squared for each feature
fig, ax = plt.subplots()
ax.scatter(mse_list, r2_list)

for i, txt in enumerate(features):
    ax.annotate(txt, (mse_list[i], r2_list[i]))

ax.set_xlabel('Mean Squared Error')
ax.set_ylabel('R-squared')
ax.set_title('Performance')

plt.show()

# Elastic Net regression:
from sklearn.linear_model import ElasticNet
elanet = ElasticNet(alpha=1.0, l1_ratio=0.5)

    
#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
X = np.array([258.0, 270.0, 294.0, 
              320.0, 342.0, 368.0, 
              396.0, 446.0, 480.0, 586.0])\
             [:, np.newaxis]

y = np.array([236.4, 234.4, 252.8, 
              298.6, 314.2, 342.2, 
              360.8, 368.0, 391.2,
              390.8])    
lr = LinearRegression()
pr = LinearRegression()
quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X)

    
lr.fit(X, y)
X_fit = np.arange(250, 600, 10)[:, np.newaxis]
y_lin_fit = lr.predict(X_fit)

# fit quadratic features
pr.fit(X_quad, y)
y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))

plt.scatter(X, y, label='training points')
plt.plot(X_fit, y_lin_fit, label='linear fit', linestyle='--')
plt.plot(X_fit, y_quad_fit, label='quadratic fit')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()    
    
y_lin_pred = lr.predict(X)
y_quad_pred = pr.predict(X_quad)


print('Training MSE linear: %.3f, quadratic: %.3f' % (
        mean_squared_error(y, y_lin_pred),
        mean_squared_error(y, y_quad_pred)))
print('Training R^2 linear: %.3f, quadratic: %.3f' % (
        r2_score(y, y_lin_pred),
        r2_score(y, y_quad_pred)))    
    
print("My name is {Chenyi Mao}")
print("My NetID is: {chenyim2}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
    
    