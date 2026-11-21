#opt/anaconda3/bin/python3.7
"""
    compute reflectivity index across different lithologies
    based on seismic velocities and rock density
    1) Train a Neural Network based on labeled well-logs (Vp,Vs, rho, class label = reflectivity)
    2) Cross-validation testing
    3) Compare performance to base model (random and majority class label)
    4) Blind test of trained model

    input: X_train.npy - np.vstack([vp1, vs1, rho1, vp2, vs2, rho2, theta]).T
        with:
            # Make 'Upper' layers.
            vp1 = vp[:-1]
            vs1 = vs[:-1]
            rho1 = rho[:-1]

            # Make 'Lower' layers.
            vp2 = vp[1:]
            vs2 = vs[1:]
            rho2 = rho[1:]
"""
import matplotlib.pyplot as plt
import numpy as np
#-----scikit--------------------------------
from sklearn.model_selection import ShuffleSplit# for cross-validatio
from sklearn.model_selection import learning_curve
#from sklearn.model_selection import validation_curve
# classifiers/estimators
from sklearn.neural_network import MLPRegressor
#from sklearn.svm import SVR
from sklearn import linear_model
# Performance evaluation
import sklearn.metrics
#=====================0=====================
#            params and files
#===========================================
np.random.seed( 12345)
l_head    = ['Vp', 'Vs', 'Rho']
data_dir  = 'data'
file_train= 'well_train.npy'
file_test = 'well_test.npy'

# estimator = MLPRegressor( hidden_layer_sizes=(100,10), 
#                           activation='relu',) #SVR()
estimator = linear_model.LinearRegression()
#estimator  = linear_model.SGDRegressor()



#=====================1=====================
#            load data
#===========================================
m_data  = np.load( f"{data_dir}/{file_train}")
m_test  = np.load( f"{data_dir}/{file_test}")


X_train = m_data[:, :-1] 
a_y_train = m_data[:, -1]  

X_test = m_test[:, :-1]
a_y_test = m_test[:, -1]


print( ' Data (nSample, nAttrib): ', X_train.shape, a_y_train.shape)

#=====================2=====================
#              train ML algorithm
#===========================================
estimator.fit( X_train, a_y_train)
a_y_train_hat = estimator.predict( X_train)
a_y_test_hat  = estimator.predict( X_test)
#=====================3=====================
#               performance eval
#===========================================

R2_train = round(sklearn.metrics.r2_score( a_y_train, a_y_train_hat),2)
print( 'R2: train', R2_train)
R2_test  = round( sklearn.metrics.r2_score( a_y_test, a_y_test_hat), 2)
print( 'R2: test', R2_test)
# compare perfomance to baseline: i.e. randomly drawn targets
a_y_base = a_y_train[np.random.randint(0,len(a_y_train)-1, len(a_y_test))]
print( 'R2: base-model', round( sklearn.metrics.r2_score( a_y_test, a_y_base), 2))

MSE_train = round(sklearn.metrics.mean_squared_error(  a_y_train, a_y_train_hat),4)
print( "MSE: train", MSE_train)
MSE_test  = round(sklearn.metrics.mean_squared_error(  a_y_test, a_y_test_hat),4)
print( "MSE: test", MSE_test)

# cross-validation to get more accurate learning curve results
print('max train data size: ', len(a_y_train) *.9)# N_valid = N_tot - N_train - N_test
# cross validation with 10% test data
cv = ShuffleSplit( n_splits=10, test_size=0.1, random_state=123)
a_nTrain, m_train_scores, m_valid_scores =learning_curve(estimator=estimator,
                                                       X=X_train,y=a_y_train,
                                                       scoring= 'r2', #'mean_squared_error' or 'r2',
													   train_sizes=np.linspace(0.1, 1.0, 10),
													   cv=cv,n_jobs=-1)
#=====================4=====================
#                figures
#===========================================
# note that here we plot results as a function of sample not depth
# for simplicity. We could include depth in the training and testing data
# but it should only be used for plotting purposes
plt.figure(1)
ax = plt.subplot( 131)
ax.set_title(f"Train, R2={R2_train}")
ax.plot( a_y_train,     np.arange( len(a_y_train)),  'k-', lw = 3, alpha = .3, label = 'Obs')
ax.plot( a_y_train_hat, np.arange( len(a_y_train)),  'r-', lw = 1, alpha = 1, label = 'Model')

ax.invert_yaxis()
ax.legend()
ax.set_xlabel( 'Reflectivity')
ax.set_ylabel( 'Depth (samples)')

ax = plt.subplot( 132)
ax.set_title(f"Test, R2={R2_test}")
ax.plot( a_y_test, np.arange( len(a_y_test)),  'k-', lw = 3, alpha = .3, label = 'Obs')
ax.plot( a_y_test_hat, np.arange( len(a_y_test)),  'r-', lw = 1, alpha = 1, label = 'Model')
ax.plot( a_y_base, np.arange( len(a_y_base)),  'g-', lw = 1, alpha = 1, label = 'Base')
ax.invert_yaxis()
ax.legend()
ax.set_xlabel( 'Reflectivity')
ax.set_ylabel( '')

ax2= plt.subplot( 133)


residuals_test = a_y_test_hat - a_y_test
p5, p95 = np.percentile(residuals_test,[5,95])
print(f"Residuals 5th percentile: {p5:.4f}, 95th percentile: {p95:.4f}")
ax2 = plt.subplot(133)
ax2.hist(residuals_test, color="b", label="Residuals")
ax2.axvline(p5, color="b", linestyle="-.", label=f"5th: {p5:.3f}")
ax2.axvline(p95, color="b", linestyle="--", label=f"95th: {p95:.3f}")
ax2.set_xlabel("Residuals (misfit)")
ax2.legend()
plt.savefig("obs_train.png")

#-----------learning curve--------------------
plt.figure(2) 
ax = plt.subplot(111)
ax.plot( a_nTrain, m_train_scores.mean(axis=1), 'g-', lw = 2, label = 'Training')
ax.plot( a_nTrain, m_valid_scores.mean(axis=1), 'r-', lw = 2, label = 'Validation')
ax.legend( loc = 'upper right')
ax.set_xlabel( 'Size of Training Data')
ax.set_ylabel( 'Score (R2)')
plt.savefig( 'well_log_learning_curve.png')
plt.show()
