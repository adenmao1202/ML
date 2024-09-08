""" 
Total : 14 cols 
Features description : 
checkin_acc - status of exixsting checking account
duration - duration of the credit given in months
credit_history - credit history
amount - amount of credit/loan 貸款金額
savings_acc - balance in savings acc
present_emp_since - employment in years
inst_rate - installment rate 分期付款利率
personal_Status - marital status of applicant
residing_since - residing since in years 居住時間
age - age in years
inst_plans - other installement plans of the applicant
checkin_acc (bal) - balance in checking account 支票帳戶餘額
job - types of job of an applicant
status (Credit) - credit status

 #   Column             Non-Null Count  Dtype 
---  ------             --------------  ----- 
 0   checkin_acc        1000 non-null   object
 1   duration           1000 non-null   int64 
 2   credit_history     1000 non-null   object
 3   amount             1000 non-null   int64 
 4   savings_acc        1000 non-null   object
 5   present_emp_since  1000 non-null   object
 6   inst_rate          1000 non-null   int64 
 7   personal_status    1000 non-null   object
 8   residing_since     1000 non-null   int64 
 9   age                1000 non-null   int64 
 10  inst_plans         1000 non-null   object
 11  num_credits        1000 non-null   int64 
 12  job                1000 non-null   object
 13  status             1000 non-null   int64 
""" 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from math import floor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score

# matplotlib defaults
plt.style.use("fivethirtyeight")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#######################################################################

df = pd.read_csv("new_credit_data.csv")
df.head()
df.info()
df.columns

############################################################

status_df=pd.DataFrame(data=np.array(df['status'].value_counts()), 
                   index=['good credit','bad credit'], 
                   columns=['status'])
print(status_df)

df.describe().T   ## transpose 比較好看而已


##############################################
### Understanding the data

""" length of cat_col:  7
categorical cols are :  ['checkin_acc', 'credit_history', 'savings_acc', 'present_emp_since', 'personal_status', 'inst_plans', 'job']
##########################################################
length of num_col:  7
numerical cols are:  ['duration', 'amount', 'inst_rate', 'residing_since', 'age', 'num_credits', 'status']
##########################################################
length of num but cat cols:  4
Num but categorical cols are:  ['inst_rate', 'residing_since', 'num_credits', 'status']
##########################################################
length of continuous cols:  3
continus features:  ['duration', 'amount', 'age']
##########################################################
lenght of high_Cat is:  0
""" 



def dataframe_columns(df, car=10):
    # cat columns
    cat_col = [col for col in df.columns if df[col].dtypes == 'object']
    print("length of cat_col: ", len(cat_col))
    print("categorical cols are : ", cat_col)
    print("##########################################################")
    
    # num columns
    num_col = [col for col in df.columns if df[col].dtypes == 'int64']
    print("length of num_col: ", len(num_col))
    print("numerical cols are: ", num_col)
    print("##########################################################")
    # cardinality of columns # 基數概念 
    num_but_cat = [col for col in df.columns if df[col].dtypes == 'int64' and df[col].nunique() < car]
    print("length of num but cat cols: ", len(num_but_cat))
    print("Num but categorical cols are: ", num_but_cat)
    print("##########################################################")
    # continuous featues
    cont_num_col = [col for col in num_col if col not in num_but_cat]
    print("length of continuos cols: ", len(cont_num_col))
    print("continus features: ", cont_num_col)
    print("##########################################################")
      # cat cardinality cols 
    # low cardinality cols
    low_cat = [col for col in df.columns if df[col].dtypes == 'object' and df[col].nunique() < car]
    print("length of cat_col: ", len(low_cat))
    print("Low categorical features: ", low_cat)
    print("##########################################################")
    # high cardinality cols
    high_Cat = [col for col in cat_col if col not in low_cat]
    print("lenght of high_Cat is: ", len(high_Cat))
    
    return cat_col, num_col, num_but_cat, cont_num_col

cat_col, num_col, num_but_cat, cont_num_col = dataframe_columns(df)

################################################
cat_col
num_col
cont_num_col  # continuous numerical columns
num_but_cat


df[df['status'] == 1].value_counts().sum()  # bad 
df[df['status'] == 0].value_counts().sum()  # good 



fig, ax = plt.subplots(3, figsize=(10,15))
for idx, col in enumerate(cont_num_col):  #idx = index 
    sns.distplot(x=df[df['status'] == 0][col],
                 kde=False, 
                 label='Good credit', 
                 ax=ax[idx])
    sns.distplot(x=df[df['status'] == 1][col],
                 kde=False, 
                 label='Bad credit', 
                 ax=ax[idx])
    ax[idx].legend(title='Status')
    ax[idx].set_xlabel(col)
    ax[idx].set_ylabel("Frequency")
    ax[idx].grid(visible=True, 
                 color='wheat', 
                 linestyle='--')
    ax[idx].set_title("Distribution by {}".format(col))
    
plt.suptitle("Distribution of Duration,Age and amount by credit status", 
             fontsize=18, 
             fontweight='bold')
fig.tight_layout()
plt.show()



#######################################
""" 
> WORKFLOW

count plots by cat and num_but_cat col
point plot by cat and num_Cat with status cols
box plot by status vs num col
ouliers detection
correlation matrix
scaling data
model building:

logistic regression
SVM
k-NN
Naive bayes
RF XGB
"""

## count plots by cat and num_but_cat_col

def count_plot(var):
    # printing values of variable
    new = df[var].value_counts()
    
    # plot detail 
    plt.figure(figsize=(8,5))
    sns.countplot(x=var, 
                  data=df)
    plt.grid(visible=True, 
                 linestyle='--')
    plt.title(var)
    plt.show()
    print("{}:\n{}".format(var,new))



## cat col
for var in cat_col: 
    count_plot(var)
    
############################################

## num_but_cat col 
for var in num_but_cat:
    count_plot(var)
    



############################################
new_cat_col = cat_col + num_but_cat

fig, ax = plt.subplots(6,2, figsize=(15,27))
for idx, var in enumerate(new_cat_col):
    sns.pointplot(x=var, y='status', data=df, ax=ax[floor(idx/2), idx%2])
    ax[floor(idx/2), idx%2].grid(visible=True,
                                linestyle='--',
                                color='wheat')
    ax[floor(idx/2), idx%2].set_title(var)
    
plt.suptitle("Relation between cat-cols and status of credit", 
             fontsize=18, 
             fontweight='bold')
fig.tight_layout()
plt.show()


## How do we get the conclusion below ? 
""" 
Conclusion : 
job, num_credits, and residing_since do not affects status of credits, 
these columns can be removed

checking acc and creidt history affect the credit rating

判斷：
均值变化不大：这些特征不同类别的 status 均值变化较小。
误差线重叠（垂直）：不同类别的均值误差线大部分重叠，说明这些类别之间没有显著差异。
--> 可用卡方或ANOVA 進階分析
"""

#########################################
## box plot of dist --> see outliers

cols = list([col for col in num_col if col != 'status'])

for idx, col in enumerate(cols):
    plt.figure(idx, figsize=(7,4))
    sns.boxplot(x=col, data=df)
    plt.grid(visible=True,
             color='wheat',
             linestyle='--')
    plt.title("Distribution of {}".format(col))
    plt.show()
    
""" 
outliers are there in : 
age, duration, amount and num_credits 

"""
    
######################################

# correlation matrix of numerical cols ( can only see corr in num col)
numerical_df = df.select_dtypes(include=[np.number])
corr_matrix = numerical_df.corr()

# Generate a mask for the upper triangle ( 只需要知道一半就可)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(10, 10))

# Draw the heatmap
heatmap = sns.heatmap(corr_matrix, mask=mask, annot=True, linewidth=2, cmap='coolwarm')

# Customize the heatmap title and display
heatmap.set_title('Correlation of Numerical Features', 
                  fontdict={'fontsize': 30, 'fontfamily': 'serif', 'fontweight': 'bold'},
                  pad=16)
plt.show()

#############################

## Logistic Reg
df
# features to be removed 
remove_col = ['job', 'residing_since','num_credits']
feature_encode = ['checkin_acc', 'credit_history', 'savings_acc', 'present_emp_since', 'personal_status', 'inst_plans']

# creat new copy of df
new_df = df.copy()
new_df.drop(remove_col, inplace=True, axis=1)
new_df
#encode features
# One Hot Encoding
# drop_first=True 用來避免虛擬變量陷阱，這樣可以減少多重共線性問題。

encode_df = pd.get_dummies(new_df, columns=feature_encode, drop_first=True)
encode_df

# 將布林值轉換為數值型
encode_df = encode_df.applymap(lambda x: 1 if x is True else (0 if x is False else x))


# divide X and y columns
# statsmodels.add_constant: 加入 const 可避免ML model 認為沒有截距項
X = sm.add_constant(encode_df.drop('status', axis=1))
y = encode_df['status']

# split the train and test
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.75, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

X_train
X_test
# builing logit model using statsmodels api
logit = sm.Logit(y_train, X_train)
logit_model = logit.fit()

""" 
Optimization terminated successfully.
         Current function value: 0.491087
         Iterations 6

這表示在優化過程結束時，目標函數的值是 0.491087。
對於邏輯回歸模型，這個目標函數是負對數似然函數（Negative Log-Likelihood），它度量了模型的擬合優度。
這個值越小，模型對訓練數據的擬合程度越好。
需要注意的是，這個值本身沒有單獨的意義，主要用來與其他模型的對數似然值進行比較。 """ 

# call the summary of logit model
logit_model.summary2()   # p-value is the key

# FIND significant features
significant_fea = pd.DataFrame(logit_model.pvalues)
significant_fea[significant_fea[0] < 0.05].reset_index()


# validate with test data
# prob > 0.5 then we predict to be 1 
pred_df = pd.DataFrame({"actual": y_test,
                        "Predicted_prob": logit_model.predict(X_test)
                       })

pred_df['Predicted'] = pred_df['Predicted_prob'].map(lambda x: 1 if x>0.5 else 0)
pred_df

##############################################

# confusion matrix
def confusion_matri(actual, predicted):
    cm = confusion_matrix(actual, predicted, labels=[1,0])
    tp, fp, fn, tn = cm.ravel()
    #plot confusion matrix
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='.2f', 
                xticklabels =['Bad credit','Good credit'],
                yticklabels =['Bad credit','Good credit'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted lable')
    plt.title("Confusion Matrix")
    plt.show()
    return tn,fp,fn,tp

# confusion matrix of logit model
confusion_matri(pred_df['actual'], pred_df['Predicted'])

""" 
here 0 = good and 1 = bad, 
TP = bad credit correctky classifed as bad
""" 

report = classification_report(pred_df['actual'], pred_df['Predicted'])
print(report)
print("#############################################################")
print("Accuracy score: ", accuracy_score(pred_df['actual'], pred_df['Predicted']))
print("Precision score: ", precision_score(pred_df['actual'], pred_df['Predicted']))

""" 
accuracy score is nearly 79% while precision is certainly low compared to accuracy.

here classes are unbalanced and we are taking decision boundry as a 0.5 
which is not resulting in right metrix.

--> our cut-off probability shoud be changed.
""" 

# let's plot actual value of class and predicted probability 
# 看 overlap 
plt.figure(figsize=(6,6))
sns.distplot(pred_df[pred_df['actual']==1]['Predicted_prob'], kde=False,
            label='Bad Credit', color='r')
sns.distplot(pred_df[pred_df['actual']==0]['Predicted_prob'], kde=False,
            label='Good Credit')

plt.title('Distribution of actual bad and credit with its predicted prob')
plt.grid(visible=True, color='wheat', linestyle='--')
plt.legend()
plt.show()

###############################################

# plotting roc-auc curve
def draw_roc(actual, proba):
    fpr, tpr, thresholds = roc_curve(actual, proba,drop_intermediate=False)  # 保留中間值
    auc_score = roc_auc_score(actual, proba)
    # plot the curve
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label="ROC curve(area={})".format(round(auc_score,2)))
    plt.plot([0,1],[0,1], 'k--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(visible=True, color='wheat', linestyle='--')
    plt.legend(loc="lower right")
    plt.title("ROC-AUC Curve")
    plt.show()
    return fpr, tpr, thresholds

# roc-auc curve of logit model
fpr, tpr, thresholds = draw_roc(pred_df['actual'],pred_df['Predicted_prob'])

# AUC score: area under the curve
auc_score = round(roc_auc_score(pred_df['actual'],pred_df['Predicted_prob']),2)
print("AUC score of the model is: ", auc_score)

# Youden's index based approch to find cut-off probability 
tpr_fpr = pd.DataFrame({'tpr': tpr,
                       'fpr': fpr,
                       'thresholds': thresholds}
                      )

tpr_fpr['Diff'] = tpr_fpr.tpr - tpr_fpr.fpr
tpr_fpr.sort_values(by='Diff', ascending=False).head()

""" 
Diff: youden's index ( TPR - FPR ) --> choose max 

thresholds: cut-off probability --> we get 0.370720
tpr: true positive rate
fpr: false positive rate
"""

### Change cut-off probability
pred_df['Predicted_new'] = pred_df['Predicted_prob'].map(lambda x: 1 if x>0.37 else 0)

# new confution matrix
confusion_matri(pred_df['actual'], pred_df['Predicted_new'])

new_report = classification_report(pred_df['actual'], pred_df['Predicted_new'])
print(new_report)

print("#######################################################################")

print("Accuracy score: ", accuracy_score(pred_df['actual'], pred_df['Predicted_new']))
print("Precision score: ", precision_score(pred_df['actual'], pred_df['Predicted_new']))

""" 
Accuracy score:  0.768
Precision score:  0.5777777777777777
# prediction still bad 
"""

##################################
### Other Models 
X_n = encode_df.drop('status', axis=1)
y_n = encode_df['status']
X_n.value_counts
y_n.value_counts


over_sampling = SMOTE(sampling_strategy='minority', random_state=42)
# Synthetic Minority Over-sampling Technique
# 它通過合成新的少數類別樣本來平衡類別分佈，而不是簡單地重複已有的少數類別樣本。
X_n, y_n = over_sampling.fit_resample(X_n,y_n)
print("New balanced classes")
y_n.value_counts()
""" 
status
0    700
1    700
""" 

""" 
對原始資料進行了平衡處理，使得每個類別的樣本數量相同。這通常是通過以下方法之一來實現的：

欠抽樣(Under-sampling)：從較多樣本的類別中隨機抽取與較少樣本的類別數量相同的樣本。
過抽樣(Over-sampling)：從較少樣本的類別中進行隨機重複抽樣，使其數量與較多樣本的類別相同。
合成方法(例如SMOTE)：通過生成新的樣本來平衡類別數量。
"""


# standardize 4 continous cols
# split the train and test
X_train, X_test, y_train, y_test = train_test_split(X_n, y_n, 
                                              train_size=0.75, 
                                             random_state=42)



################################################
### PREPROCESSING

# Min Max scaler : 將資料壓縮到一定範圍
# 為什麼一定是使用這個transformer ? 
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train= scaler.transform(X_train)
X_test= scaler.transform(X_test)
X_test.shape, X_train.shape, y_train.shape, y_test.shape


#####################################################

models_dict = {'RFC': RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42),
               'DT': DecisionTreeClassifier(criterion='entropy', random_state=42,max_depth=20),
               'knn': KNeighborsClassifier(n_neighbors=4),
               'GNB': GaussianNB(),
               'SVC': SVC(C=2.0),
               'GBC': GradientBoostingClassifier(max_depth=2, random_state=42),
               'XGB': XGBClassifier(n_estimators=150, max_leaves=4)}

# fitting models and score
for k,model in models_dict.items():
    model.fit(X_train,y_train)
    score = model.score(X_test,y_test)   # 可以直接計算 model score ? 
    predict_y = model.predict(X_test)
    classif_report = classification_report(y_test,predict_y)
    print("Metrics of Model {}: \n".format(k))
    print("Score of model {} is {}: \n".format(k,round(score,2)))
    print("Classification report of model {} is: \n{}".format(k,classif_report))
    print("###################################################################")


# confusion matrix
# confusion matrix of all six models
for k,model in models_dict.items():
    score = round(model.score(X_test,y_test),2)
    predict_y = model.predict(X_test)
    cm = confusion_matri(y_test,predict_y)
    print(f"For model {k}", cm)
    print("##########################################################")


# Accuracy of all the models
for k,model in models_dict.items():
    score = round(model.score(X_test,y_test),2)
    predict_y = model.predict(X_test)
    print(f"Accuracy score of {k}: ", round(accuracy_score(y_test,predict_y),2))