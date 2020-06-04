import pandas as pd 
import numpy as np 
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 60)
from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt 

plt.rcParams['font.size']=24
import seaborn as sns
sns.set(font_scale=2)

from sklearn.model_selection import train_test_split

data = pd.read_csv(r'C:\Users\wangrg\Downloads\machine-learning-project-walkthrough-master\machine-learning-project-walkthrough-master\data\Energy_and_Water_Data_Disclosure_for_Local_Law_84_2017__Data_for_Calendar_Year_2016_.csv')

# print(data.head())
# print(data.info())

data=data.replace({'Not Available':np.nan})
for col in list(data.columns):
    if ('ft²' in col or 'kBtu' in col or 'Metric Tons CO2e' in col or 'kWh' in 
        col or 'therms' in col or 'gal' in col or 'Score' in col):
        data[col]=data[col].astype(float)
# print(data.describe())
# print(data.shape[0])
def missing_values_table(df):
    mis_val=df.isnull().sum()
    mis_val_percent=100*mis_val/len(df)
    mis_val_table=pd.concat([mis_val,mis_val_percent],axis=1)
    mis_val_table_ren_columns=mis_val_table.rename(columns={0:'Missing Values',1:r'% of Total Values'})
    mis_val_table_ren_columns=mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0 ].sort_values(r'% of Total Values',ascending=False).round(1)
    print("Your selected dataframe has" + str(df.shape[1])+"columns.\n"
    "There are"+str(mis_val_table_ren_columns.shape[0])+"columns that have missing values")
    return mis_val_table_ren_columns
# missing_values_talbe(data)
missing_df=missing_values_table(data)
missing_columns=list(missing_df[missing_df['% of Total Values']>50].index)
# print('We will remove %d columns' %len(missing_columns))

data=data.drop(columns=list(missing_columns))

# print(data.info())
#分析单变量因素
figsize(8, 8)
data=data.rename(columns={'ENERGY STAR Score':'Score'})

plt.style.use('fivethirtyeight')
plt.hist(data['Score'].dropna(),bins=100,edgecolor='k')
plt.xlabel('Score')
plt.ylabel('Number of Buildings')
plt.title('Enery Star Score Distribution')
plt.show()
#查看变量相关性
types = data.dropna(subset=['score'])
types = types['Largest Property Use Type'].value_counts()
types = list(types[types.values > 100].index)

figsize(12, 10)

for b_type in types:
  
    subset = data[data['Largest Property Use Type'] == b_type]
    sns.kdeplot(subset['score'].dropna(),
               label = b_type, shade = False, alpha = 0.8)
    
plt.xlabel('Energy Star Score', size = 20); plt.ylabel('Density', size = 20)
plt.title('Density Plot of Energy Star Scores by Building Type', size = 28)
plt.show()

correlations_data = data.corr()['score'].sort_values()

print(correlations_data.head(15), '\n')

print(correlations_data.tail(15))

numeric_subset = data.select_dtypes('number')

for col in numeric_subset.columns:
    if col == 'score':
        next
    else:
        numeric_subset['sqrt_' + col] = np.sqrt(numeric_subset[col])
        numeric_subset['log_' + col] = np.log(numeric_subset[col])

categorical_subset = data[['Borough', 'Largest Property Use Type']]
categorical_subset = pd.get_dummies(categorical_subset)
features = pd.concat([numeric_subset, categorical_subset], axis = 1)
features = features.dropna(subset = ['score'])
correlations = features.corr()['score'].dropna().sort_values()

figsize(12, 10)

features['Largest Property Use Type'] = data.dropna(subset = ['score'])['Largest Property Use Type']

features = features[features['Largest Property Use Type'].isin(types)]

sns.lmplot('Site EUI (kBtu/ft²)', 'score', 
          hue = 'Largest Property Use Type', data = features,
          scatter_kws = {'alpha': 0.8, 's': 60}, fit_reg = False,
          size = 12, aspect = 1.2)
plt.xlabel("Site EUI", size = 28)
plt.ylabel('Energy Star Score', size = 28)
plt.title('Energy Star Score vs Site EUI', size = 36)
plt.show()
#分离出测试集与训练集
no_score = features[features['score'].isna()]
score = features[features['score'].notnull()]

# print(no_score.shape)
# print(score.shape)
features = score.drop(columns='score')
targets = pd.DataFrame(score['score'])

features = features.replace({np.inf: np.nan, -np.inf: np.nan})
X, X_test, y, y_test = train_test_split(features, targets, test_size = 0.3, random_state = 42)
# print(X.shape)
# print(X_test.shape)
# print(y.shape)
# print(y_test.shape)
#定义一个baseline 平均绝对误差
def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))
baseline_guess = np.median(y)

# print('The baseline guess is a score of %0.2f' % baseline_guess)
# print("Baseline Performance on the test set: MAE = %0.4f" % mae(y_test, baseline_guess))
