import plotly as pt
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline 

# df = pd.read_csv('group/train_text_metric_v1.csv')
# df2 = pd.read_csv('group/train_text_metric.csv')

# df_train = df.val
# df2_train = df2.val

# x1 = np.arange(len(df_train))
# y1 = df_train

# model1 = make_interp_spline(x1, y1)
# x1 = np.linspace(1,len(df_train),4)
# y1 = model1(x1)

# x2 = np.arange(len(df2_train))
# y2 = df2_train

# model2 = make_interp_spline(x2, y2)
# x2 = np.linspace(1,len(df2_train),4)
# y2 = model2(x2)

# x1_txt = round(list(df_train)[-1],2)
# x2_txt = round(list(df2_train)[-1],2)

# plt.plot(x1, y1, color='royalblue', marker='o', lw=2, label='Open-I-S')
# plt.plot(x2, y2,  ls=':', color='orange', marker='*', ms=10, lw=2, label='Open-I')



# df = pd.read_csv('group/train_text_loss_v1.csv')
# df2 = pd.read_csv('group/train_text_loss.csv')

# df_train = df.train
# df2_train = df2.train

# x1 = np.arange(len(df_train))[:60]
# y1 = df_train[:60]

# x2 = np.arange(len(df2_train))[:15]
# y2 = df2_train[:15]


# plt.plot(x1, y1, color='royalblue', lw=2, label='Open-I-S')
# plt.plot(x2, y2,  ls='-', color='orange', ms=10, lw=2, label='Open-I')

# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend()
# plt.show()

df = pd.read_csv('group/ClsGen_2s_train_loss.csv')

df_train = df.train
df_val = df.valid

x = np.arange(len(df_train))


plt.plot(x, df_train, color='royalblue', lw=2, label='Train')
plt.plot(x, df_val,  ls='-', color='orange', ms=10, lw=2, label='Validation')

plt.title('MV and MV + T')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()





# plt.annotate(x1_txt, xy = (x1[-1], y1[-1]), xytext = (x1[-1]-5, y1[-1]-0.03))
# plt.annotate(x2_txt, xy = (x2[-1], y2[-1]), xytext = (x2[-1]-3, y2[-1]-0.03))

# plt.ylabel('Area Under Curve (AUC)')
# plt.xlabel('Epoch')
# plt.legend()
# plt.show()

# x1 = [1, 1, 2]
# y1 = [1, 2, 1]

# x2 = [0, 1, 0]
# y2 = [0, 0, 1]

# plt.scatter(x1, y1, color='r', marker='o')
# plt.scatter(x2, y2, color='orange', marker='*')
# plt.show()