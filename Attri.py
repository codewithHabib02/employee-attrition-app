import os 
os.environ["LOKY_MAX_CPU_COUN"]="4"

import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from scipy.stats import chi2_contingency
import seaborn as sns  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn. svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler
from sklearn.metrics import accuracy_score,classification_report,roc_auc_score,precision_score,f1_score
df=pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

print("\n" + "=" *60)
print("Head of the Dataset")
print(df.head())
print("\n" + "=" *60)

print("\n" + "=" *60)
print("Dtypes in the Dataset")
print(df.dtypes)
print("\n" + "=" *60)

print("\n" + "=" *60)
print("Missing_values in of the Dataset")
print(df.isna().sum())
print("\n" + "=" *60)

print("\n" + "=" *60)
print("Npercentage of Missing_values in of the Dataset")
print(df.isna().sum()/len(df)*100)
print("\n" + "=" *60)

print("\n" + "=" *60)
print("Duplicates in the Dataset")
print(df.duplicated().sum())
print("\n" + "=" *60)

print("\n" + "=" *60)
print("Description of the Dataset")
print(df.describe())
print("\n" + "=" *60)

categorical_cols=df.select_dtypes(include=['object','category']).columns.tolist()
if 'Attrition' in categorical_cols:
    categorical_cols.remove('Attrition')
    for c in categorical_cols:
        print(f"\n ============={c} vs Attrition============")
        crosstb=pd.crosstab(df[c],df['Attrition'])
        print(f"table:{crosstb}")
        chi2,p,dof,expected=chi2_contingency(crosstb)
        print(
            f"chi: {chi2}\n"
            f"p_signal: {p: .4f}"
        )

numeric_cols=df.select_dtypes(include=['int','float']).columns.tolist()
corr=df[numeric_cols].corr()
strong_crr=(
    corr.where(abs(corr) >=0.77)
    .stack()
    .reindex()
)
print(strong_crr)


label=df['Attrition'].value_counts(dropna=False)
label.plot(kind='pie',autopct='%1.1f%%')
plt.close()

sns.barplot(data=df,x='MaritalStatus',y='MonthlyIncome')
plt.title("Average MonthlyIncome by OverTime")
plt.xlabel("OverTime")
plt.ylabel("MonthlyIncome")
plt.close()


plt.figure(figsize=(8,5))
sns.boxplot(data=df,x='JobRole',y='TotalWorkingYears')
plt.title("Distribution TotalWorkingYears by JobRole")
plt.xlabel("JobRole")
plt.ylabel(" TotalWorkingYears")
plt.xticks(rotation=40,ha='right')
plt.tight_layout()
plt.close()


sns.scatterplot(data=df,x='JobLevel',y='MonthlyIncome')
plt.title("")
plt.xlabel("PerformanceRating")
plt.ylabel("MonthlyIncome")
plt.close()

plt.figure(figsize=(9,6))
corr=df[numeric_cols].corr()
sns.heatmap(corr,annot=True,fmt='.2f')
plt.xticks(rotation=40,ha='right')
plt.tight_layout()
plt.close()

# Preprocessing and Enginering##

df['Attrition']=df['Attrition'].map({'Yes': 1, 'No': 0})


x=df.drop('Attrition',axis=1,errors='ignore')
y=df["Attrition"]

cat_low=[c for c in categorical_cols if x[c].nunique()<=10]
cat_high=[c for c in categorical_cols if x[c].nunique()>10]

preprocessor=ColumnTransformer(transformers=[
    ("Num",StandardScaler(),numeric_cols),
    ("low",OneHotEncoder(handle_unknown='ignore'),cat_low),
    ("high",OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1),cat_high)

])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)



models={
    "svm": SVC(),
    "RandomForest": RandomForestClassifier(random_state=42),
    "Logistic": LogisticRegression(random_state=42),
    "Decision": DecisionTreeClassifier()
}
for name,model in models.items():

    
    pip=Pipeline(steps=[
    ("prepr",preprocessor),
    ("smot",SMOTE()),
    ("model",model)
    ])

    pip.fit(x_train,y_train)
    prediction=pip.predict(x_test)
    acc=accuracy_score(y_test,prediction)
    print(
        f"model_name :{name}\n"
        f"Accuracy :{acc}"
    )


best_model=Pipeline(steps=[
    ("prepr",preprocessor),
    ("smot",SMOTE()),
    ("model",RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ))
    ])

cv_score=cross_val_score(pip,x_train,y_train,cv=5,scoring='f1')
print(
    f"cv f1 : {cv_score}\n"
    f"mean cv: {cv_score.mean()}"
)


best_model.fit(x_train,y_train)
## Variance/Bias
trained_pred=best_model.predict(x_train)
test_pred=best_model.predict(x_test)
trained_acc=accuracy_score(y_train,trained_pred)
test_acc=accuracy_score(y_test,test_pred)
print(
    f"TRAINED: {trained_acc: .2f}\n"
    f"Tested : {test_acc: .2f}\n"
)


prediction=best_model.predict(x_test)
acc=accuracy_score(y_test,prediction)
precision=precision_score(y_test,prediction)
recal=precision_score(y_test,prediction)
f1_score=precision_score(y_test,prediction)
clt=precision_score(y_test,prediction)
roc=roc_auc_score(y_test,prediction)
print(
    f"Accuracy: {acc:.2f}\n"
    f"precision1:{precision: .2f}\n"
    f"recall: {recal:.2f}\n"
    f"f1 : {f1_score:.2f}\n"
    f"creport: {clt: .2f}\n"
    f"roc_ :{roc:.2f}"
)


# Feature_importance
features=best_model.named_steps["prepr"].get_feature_names_out()
coefficion=best_model.named_steps['model'].feature_importances_
importance=pd.DataFrame({
    "features": features,
    "importance":abs(coefficion)
}).sort_values(by="importance",ascending=False)

top_10=importance.head(10)

print(top_10)

#Retraining the top_10 features

selected_cols = [
    "OverTime",
    "JobLevel",
    "StockOptionLevel",
    "MaritalStatus",
    "YearsWithCurrManager",
    "YearsAtCompany",
    "MonthlyIncome",
    "EEAge",
    "JobRole"
]

print(df[selected_cols].nunique())
x_selected=df[selected_cols]
y=df['Attrition']

x_train,x_test,y_train,y_test=train_test_split(
    x_selected,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

num_cols=x_selected.select_dtypes(include=['int','float']).columns.tolist()
cat_cols=x_selected.select_dtypes(include=['object','category']).columns.tolist()

prop=ColumnTransformer(transformers=[
    ("num", StandardScaler(),num_cols),
    ("cat",OneHotEncoder(handle_unknown='ignore'),cat_cols)
])

pipeline=Pipeline(steps=[
    ("prob",prop),
    ("smot",SMOTE()),
    ("model",RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42))
])

pipeline.fit(x_train,y_train)
joblib.dump(pipeline,"Attrition_model.pkl")


## Variance/Bias
trained_pred=pipeline.predict(x_train)
test_pred=pipeline.predict(x_test)
trained_acc=accuracy_score(y_train,trained_pred)
test_acc=accuracy_score(y_test,test_pred)
print(
    f"TRAINED: {trained_acc: .2f}\n"
    f"Tested : {test_acc: .2f}\n"
)

prediction=pipeline.predict(x_test)
acc=accuracy_score(y_test,prediction)
precision=precision_score(y_test,prediction)
recal=precision_score(y_test,prediction)
f1_score=precision_score(y_test,prediction)
clt=precision_score(y_test,prediction)
roc=roc_auc_score(y_test,prediction)
print(
    f"Accuracy: {acc:.2f}\n"
    f"precision1:{precision: .2f}\n"
    f"recall: {recal:.2f}\n"
    f"f1 : {f1_score:.2f}\n"
    f"creport: {clt: .2f}\n"
    f"roc_ :{roc:.2f}"
)