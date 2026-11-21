#opt/anaconda3/bin/python3.7
"""
    compute reflectivity index across different lithologies
    based on seismic velocities and rock density
        First step in machine learning workflow
        - data screening and pre-processing
        - standardizing, train/test/valid data split
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os

#            files and params
#===========================================
data_dir      = 'data'
well_file     = 'well_log-1.npy'
test_size     = 10 #10 to 30 percent
# addBott_layer = False
# standardize   = False

### for part 3
addBott_layer = True
standardize   = True
#=====================1=====================
#                 load data
#===========================================
m_data    = np.load(os.path.join(data_dir, well_file))

rows, cols = m_data.shape
print(f"Rows: {rows}, Columns: {cols}")

data_headers = ["depth", "Vp", "Vs", "rho","Seismic Ref."]

# Columns: 0=depth, 1=Vp, 2=VS, 3=rho, 4=Seismic Ref.
X = m_data[:, 1:4]*1e-3
a_z = m_data[:,0]
a_y = m_data[:, 4]


for i in range(cols):
  header = data_headers[i]
  col_data = m_data[:, i]
  col_len = len(col_data)
  col_min = np.min(col_data)
  col_max = np.max(col_data)
  print(
      f"Column {i} {header}: Length = {col_len}, Range = [{col_min:.4f},"
      f" {col_max:.4f}]"
  )

df = pd.DataFrame(m_data, columns=data_headers[:cols])


plot_cols = [col for col in df.columns if col != "depth"]

fig, axes = plt.subplots(1, len(plot_cols), figsize=(12, 8), sharey=True)

i=1
for ax, col in zip(axes.flat, plot_cols):
  sb.lineplot(data=df, x=col, y="depth", ax=ax, c=f"C{i}", orient="y")
  ax.set_xlabel(col)
  i=i+1

plt.gca().invert_yaxis()
axes[0].set_ylabel("Depth")
plt.tight_layout()
plt.savefig("well_logs_attribute.png", dpi=300)
plt.show()
plt.close()

# #=====================4=====================
# #          correlation matrix with pandas
# #===========================================
#Removing depth column from df
df_corr = df.drop(columns=["depth"])
#plotting the scatter_matrix
pd.plotting.scatter_matrix(df_corr, figsize=(10, 10), diagonal="hist", alpha=0.7)
plt.suptitle("Well Log Attribute Scatter Matrix", y=0.92)
plt.savefig("well_log_scatter_matrix.png", dpi=300)
plt.show()

#Compute and print the correlation matrix
corr_matrix = df.corr()
print(corr_matrix)


#=====================2=====================
#        train/test split, standardize
#===========================================
# For part 3 Add ensuing (bottom) layer Vp, Vs, and rho as 3 new columns
if addBott_layer == True:
  X = np.vstack((X[0:-1, :].T, X[1::, :].T)).T
  a_y = a_y[0:-1]
  a_z = a_z[0:-1]
  
X_train, X_test, y_train, y_test = train_test_split(X, a_y,
                                                    test_size=0.2,
                                                    random_state=12345)
print('train size: ', X_train.shape, 'test size', X_test.shape)

# standardize training and testing using same mean and sigma
if standardize == True:
    scaler = preprocessing.StandardScaler().fit( X_train)
    X_train= scaler.transform( X_train)
    X_test = scaler.transform( X_test)
    

#=====================3=====================
#         save data files as npy binary
#===========================================
print( 'shape of training data in well_train.npy')
print( np.vstack(( X_train.T, y_train)).T.shape)
np.save( f"{data_dir}/well_train.npy", np.vstack(( X_train.T, y_train)).T)
np.save( f"{data_dir}/well_test.npy",  np.vstack(( X_test.T,  y_test)).T)



plt.figure(3)
axCM = plt.subplot( 111)
plot3 = axCM.pcolor( corr_matrix, cmap = plt.cm.hot_r)
plt.colorbar( plot3, label = 'CC')
axCM.set_xticks( axCM.get_xticks()[0:-1]+.5)
axCM.set_yticks( axCM.get_yticks()[0:-1]+.5)
axCM.set_xticklabels( data_headers)
axCM.set_yticklabels( data_headers)
plt.savefig("well_corr_matrix.png")
plt.show()