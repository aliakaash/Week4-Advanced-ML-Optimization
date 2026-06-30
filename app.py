"""
app.py – Customer Churn Prediction Dashboard
Week 4 – Advanced ML | Intern: Ali Akaash
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, json, os, warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              roc_curve, auc)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model   import LogisticRegression
from sklearn.tree           import DecisionTreeClassifier
from sklearn.ensemble       import RandomForestClassifier
from sklearn.neighbors      import KNeighborsClassifier
from sklearn.svm            import SVC

# ── Page config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="🎯", layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH     = os.path.join(BASE_DIR, "churn.csv")
MODEL_PATH    = os.path.join(BASE_DIR, "best_model.pkl")
SCALER_PATH   = os.path.join(BASE_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_names.json")

# ── CSS ──────────────────────────────────────────────────────────────────
st.markdown("""<style>
.main-header{font-size:2.2rem;font-weight:700;color:#1f77b4;text-align:center}
.sub-header{text-align:center;color:#555;margin-bottom:1.5rem}
.metric-box{background:#f0f4ff;border-radius:10px;padding:1rem;
            text-align:center;border-left:4px solid #1f77b4}
.metric-val{font-size:1.8rem;font-weight:700;color:#1f77b4}
.metric-lbl{font-size:.85rem;color:#666}
.churn-yes{background:#ffe8e8;border-left:4px solid #e74c3c;
           padding:1rem;border-radius:8px;text-align:center}
.churn-no{background:#e8f5e9;border-left:4px solid #27ae60;
          padding:1rem;border-radius:8px;text-align:center}
</style>""", unsafe_allow_html=True)

# ── Data & model loading ─────────────────────────────────────────────────
@st.cache_data
def load_raw():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def preprocess():
    df = pd.read_csv(DATA_PATH)
    df.drop('customerID', axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    df.drop_duplicates(inplace=True)
    df['Churn'] = df['Churn'].map({'Yes':1,'No':0})
    for col in ['gender','Partner','Dependents','PhoneService','PaperlessBilling']:
        df[col] = LabelEncoder().fit_transform(df[col])
    multi = ['MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
             'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
             'Contract','PaymentMethod']
    df = pd.get_dummies(df, columns=multi, drop_first=True)
    df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['TenureGroup']     = pd.cut(df['tenure'],bins=[-1,12,24,48,72],
                                    labels=[0,1,2,3]).astype(int)
    X_all = df.drop('Churn', axis=1)
    y_all = df['Churn']
    sel   = VarianceThreshold(threshold=0.01)
    X_s   = sel.fit_transform(X_all)
    cols  = X_all.columns[sel.get_support()]
    X     = pd.DataFrame(X_s, columns=cols)
    y     = y_all.reset_index(drop=True)
    sc    = StandardScaler()
    nc    = [c for c in ['tenure','MonthlyCharges','TotalCharges',
                           'AvgMonthlySpend','TenureGroup'] if c in X.columns]
    X[nc] = sc.fit_transform(X[nc])
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    return X, y, Xtr, Xte, ytr, yte, sc, list(X.columns), nc

@st.cache_resource
def train_all(_Xtr, _ytr):
    ms = {
        'Logistic Regression': LogisticRegression(max_iter=1000,random_state=42),
        'Decision Tree'      : DecisionTreeClassifier(random_state=42),
        'Random Forest'      : RandomForestClassifier(n_estimators=100,random_state=42),
        'KNN'                : KNeighborsClassifier(n_neighbors=5),
        'SVM'                : SVC(kernel='rbf',probability=True,random_state=42)
    }
    for m in ms.values(): m.fit(_Xtr, _ytr)
    return ms

def evaluate(model, Xte, yte):
    yp  = model.predict(Xte)
    ypr = model.predict_proba(Xte)[:,1]
    return {'Accuracy' :round(accuracy_score(yte,yp),4),
            'Precision':round(precision_score(yte,yp,zero_division=0),4),
            'Recall'   :round(recall_score(yte,yp,zero_division=0),4),
            'F1-Score' :round(f1_score(yte,yp,zero_division=0),4),
            'ROC-AUC'  :round(roc_auc_score(yte,ypr),4)}

raw_df = load_raw()
X,y,Xtr,Xte,ytr,yte,scaler_live,feat_cols,num_cols = preprocess()
models  = train_all(Xtr.values, ytr.values)

model_saved  = joblib.load(MODEL_PATH)  if os.path.exists(MODEL_PATH)  else models['Random Forest']
scaler_saved = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else scaler_live

# ── Sidebar ──────────────────────────────────────────────────────────────
st.sidebar.title("🎯 Churn Dashboard")
st.sidebar.markdown("**Week 4 – Advanced ML**")
st.sidebar.markdown("Intern: Ali Akaash")
st.sidebar.divider()
page = st.sidebar.radio("Navigate",[
    "🏠 Home","📊 Dataset Overview","📈 EDA",
    "🤖 Model Comparison","🎯 Prediction",
    "⭐ Feature Importance","ℹ About"])
st.sidebar.divider()
st.sidebar.caption("Supervisor: Qamar Naveed | Due: 3 July 2026")

COLORS = ['#4C72B0','#DD8452','#55A868','#C44E52','#8172B2']

# ════════════════════════════════════════════════════════════════
# HOME
# ════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<div class="main-header">🎯 Customer Churn Prediction</div>',unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced ML Dashboard – Week 4</div>',unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    churn_pct = round(raw_df['Churn'].value_counts(normalize=True).get('Yes',0)*100,1)
    for col,val,lbl in [(c1,f"{raw_df.shape[0]:,}","Total Customers"),
                         (c2,f"{churn_pct}%","Churn Rate"),
                         (c3,str(raw_df.shape[1]),"Features"),
                         (c4,"5","ML Models")]:
        with col:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{val}</div>'
                        f'<div class="metric-lbl">{lbl}</div></div>',unsafe_allow_html=True)
    st.markdown("---")
    a,b = st.columns(2)
    with a:
        st.markdown("""**Goal:** Predict customer churn to allow preventive action.

**Dataset:** Telco Customer Churn (7,043 rows, 21 features)

**Models Trained:**
- Logistic Regression
- Decision Tree
- Random Forest ⭐ Best
- KNN
- SVM""")
    with b:
        st.markdown("""**Techniques Used:**
- Feature Engineering & Variance Threshold Selection
- GridSearchCV & RandomizedSearchCV
- Stratified K-Fold Cross Validation
- Model Persistence with Joblib

**Dashboard Pages:**
- Dataset Overview → EDA → Model Comparison
- Interactive Prediction → CSV Download
- Feature Importance""")
    st.info("👈 Use the sidebar to navigate.")

# ════════════════════════════════════════════════════════════════
# DATASET OVERVIEW
# ════════════════════════════════════════════════════════════════
elif page == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")
    t1,t2,t3 = st.tabs(["Preview","Statistics","Missing Values"])
    with t1:
        n = st.slider("Rows",5,50,10)
        st.dataframe(raw_df.head(n),use_container_width=True)
        st.caption(f"Shape: {raw_df.shape[0]} rows × {raw_df.shape[1]} columns")
    with t2:
        st.dataframe(raw_df.describe().round(2),use_container_width=True)
        dtype_df = pd.DataFrame({'Column':raw_df.dtypes.index,'Type':raw_df.dtypes.values,
                                  'Unique':[raw_df[c].nunique() for c in raw_df.columns]})
        st.dataframe(dtype_df,use_container_width=True)
    with t3:
        mv = raw_df.isnull().sum().reset_index()
        mv.columns=['Column','Missing']
        mv['Missing %']=(mv['Missing']/len(raw_df)*100).round(2)
        st.dataframe(mv,use_container_width=True)
        st.metric("Duplicate Rows",raw_df.duplicated().sum())

# ════════════════════════════════════════════════════════════════
# EDA
# ════════════════════════════════════════════════════════════════
elif page == "📈 EDA":
    st.title("📈 Exploratory Data Analysis")
    t1,t2,t3 = st.tabs(["Class Distribution","Numerical Features","Correlation Heatmap"])

    with t1:
        counts = raw_df['Churn'].value_counts()
        fig,axes = plt.subplots(1,2,figsize=(10,4))
        axes[0].bar(['No Churn','Churn'],counts,color=['steelblue','tomato'],edgecolor='white')
        axes[0].set_title('Class Distribution')
        [axes[0].text(i,v+20,str(v),ha='center',fontweight='bold') for i,v in enumerate(counts)]
        axes[1].pie(counts,labels=['No Churn','Churn'],autopct='%1.1f%%',
                    colors=['steelblue','tomato'],startangle=90,wedgeprops={'edgecolor':'white'})
        axes[1].set_title('Churn Ratio')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with t2:
        feat = st.selectbox("Feature",['tenure','MonthlyCharges','TotalCharges'])
        rc   = raw_df.copy()
        rc['TotalCharges'] = pd.to_numeric(rc['TotalCharges'],errors='coerce')
        fig,ax = plt.subplots(figsize=(8,4))
        for v,c,l in [('No','steelblue','No Churn'),('Yes','tomato','Churn')]:
            ax.hist(rc.loc[rc['Churn']==v,feat].dropna(),bins=30,alpha=0.6,color=c,label=l)
        ax.set_title(f'{feat} by Churn'); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()
        c1,c2=st.columns(2)
        for v,col,lbl in [('No',c1,'No Churn'),('Yes',c2,'Churn')]:
            with col:
                st.metric(f"{lbl} Mean",f"{rc.loc[rc['Churn']==v,feat].mean():.1f}")

    with t3:
        fig,ax = plt.subplots(figsize=(14,10))
        corr = X.corr()
        sns.heatmap(corr,mask=np.triu(np.ones_like(corr,dtype=bool)),
                    cmap='coolwarm',annot=False,linewidths=0.3,vmin=-1,vmax=1,ax=ax)
        ax.set_title('Feature Correlation Heatmap',fontsize=14,fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════════
# MODEL COMPARISON
# ════════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":
    st.title("🤖 Model Comparison")
    rows = []
    for name,m in models.items():
        r = evaluate(m,Xte,yte); r['Model']=name; rows.append(r)
    res = pd.DataFrame(rows)[['Model','Accuracy','Precision','Recall','F1-Score','ROC-AUC']]

    st.subheader("Performance Metrics")
    st.dataframe(res.style.highlight_max(
        subset=['Accuracy','Precision','Recall','F1-Score','ROC-AUC'],
        color='#d4edda'),use_container_width=True)

    t1,t2,t3 = st.tabs(["Accuracy","All Metrics","ROC Curves"])
    with t1:
        fig,ax=plt.subplots(figsize=(9,5))
        bars=ax.bar(res['Model'],res['Accuracy'],color=COLORS,edgecolor='white',width=0.5)
        ax.set_ylim(0,1.1); ax.set_title('Accuracy Comparison',fontsize=14,fontweight='bold')
        [ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.01,f'{v:.4f}',
                 ha='center',fontweight='bold') for b,v in zip(bars,res['Accuracy'])]
        plt.xticks(rotation=15); plt.tight_layout(); st.pyplot(fig); plt.close()

    with t2:
        metrics=['Accuracy','Precision','Recall','F1-Score','ROC-AUC']
        x=np.arange(len(res)); w=0.15
        fig,ax=plt.subplots(figsize=(12,5))
        for i,(m,c) in enumerate(zip(metrics,COLORS)):
            ax.bar(x+i*w,res[m],w,label=m,color=c)
        ax.set_xticks(x+w*2); ax.set_xticklabels(res['Model'],rotation=15)
        ax.set_ylim(0,1.15); ax.set_title('All Metrics',fontsize=14,fontweight='bold'); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with t3:
        fig,ax=plt.subplots(figsize=(8,6))
        for (name,m),c in zip(models.items(),COLORS):
            fpr,tpr,_=roc_curve(yte,m.predict_proba(Xte)[:,1])
            ax.plot(fpr,tpr,color=c,lw=2,label=f'{name} (AUC={auc(fpr,tpr):.3f})')
        ax.plot([0,1],[0,1],'k--',lw=1)
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
        ax.set_title('ROC Curves',fontsize=14,fontweight='bold'); ax.legend(loc='lower right')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("Confusion Matrix")
    sel = st.selectbox("Model",list(models.keys()))
    cm  = confusion_matrix(yte,models[sel].predict(Xte))
    fig,ax=plt.subplots(figsize=(4,3))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',ax=ax,
                xticklabels=['No','Yes'],yticklabels=['No','Yes'])
    ax.set_title(f'{sel}'); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════════
# PREDICTION
# ════════════════════════════════════════════════════════════════
elif page == "🎯 Prediction":
    st.title("🎯 Customer Churn Prediction")
    st.info("Fill in customer details to predict churn.")

    with st.form("pred_form"):
        st.subheader("Customer Details")
        c1,c2,c3 = st.columns(3)
        with c1:
            gender    = st.selectbox("Gender",["Male","Female"])
            senior    = st.selectbox("Senior Citizen",[0,1])
            partner   = st.selectbox("Partner",["Yes","No"])
            dependents= st.selectbox("Dependents",["Yes","No"])
            tenure    = st.slider("Tenure (months)",0,72,12)
        with c2:
            phone     = st.selectbox("Phone Service",["Yes","No"])
            multi     = st.selectbox("Multiple Lines",["Yes","No","No phone service"])
            internet  = st.selectbox("Internet Service",["DSL","Fiber optic","No"])
            sec       = st.selectbox("Online Security",["Yes","No","No internet service"])
            backup    = st.selectbox("Online Backup",["Yes","No","No internet service"])
        with c3:
            devpro    = st.selectbox("Device Protection",["Yes","No","No internet service"])
            techsup   = st.selectbox("Tech Support",["Yes","No","No internet service"])
            contract  = st.selectbox("Contract",["Month-to-month","One year","Two year"])
            paperless = st.selectbox("Paperless Billing",["Yes","No"])
            payment   = st.selectbox("Payment Method",
                ["Electronic check","Mailed check",
                 "Bank transfer (automatic)","Credit card (automatic)"])
            monthly   = st.number_input("Monthly Charges ($)",18.0,120.0,65.0,step=0.5)
            total     = st.number_input("Total Charges ($)",0.0,9000.0,
                                         monthly*tenure,step=1.0)
        submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

    if submitted:
        inp = {c:0 for c in feat_cols}
        inp.update({
            'gender'          :1 if gender=='Male' else 0,
            'SeniorCitizen'   :senior,
            'Partner'         :1 if partner=='Yes' else 0,
            'Dependents'      :1 if dependents=='Yes' else 0,
            'tenure'          :tenure,
            'PhoneService'    :1 if phone=='Yes' else 0,
            'MonthlyCharges'  :monthly,
            'TotalCharges'    :total,
            'PaperlessBilling':1 if paperless=='Yes' else 0,
            'MultipleLines_No phone service':1 if multi=='No phone service' else 0,
            'MultipleLines_Yes':1 if multi=='Yes' else 0,
            'InternetService_Fiber optic':1 if internet=='Fiber optic' else 0,
            'InternetService_No':1 if internet=='No' else 0,
            'OnlineSecurity_No internet service':1 if sec=='No internet service' else 0,
            'OnlineSecurity_Yes':1 if sec=='Yes' else 0,
            'OnlineBackup_No internet service':1 if backup=='No internet service' else 0,
            'OnlineBackup_Yes':1 if backup=='Yes' else 0,
            'DeviceProtection_No internet service':1 if devpro=='No internet service' else 0,
            'DeviceProtection_Yes':1 if devpro=='Yes' else 0,
            'TechSupport_No internet service':1 if techsup=='No internet service' else 0,
            'TechSupport_Yes':1 if techsup=='Yes' else 0,
            'Contract_One year':1 if contract=='One year' else 0,
            'Contract_Two year':1 if contract=='Two year' else 0,
            'PaymentMethod_Credit card (automatic)':1 if payment=='Credit card (automatic)' else 0,
            'PaymentMethod_Electronic check':1 if payment=='Electronic check' else 0,
            'PaymentMethod_Mailed check':1 if payment=='Mailed check' else 0,
            'AvgMonthlySpend':total/(tenure+1),
            'TenureGroup':min(int(tenure//12),3)
        })
        df_inp = pd.DataFrame([inp]).reindex(columns=feat_cols, fill_value=0)
        nc_here = [c for c in num_cols if c in df_inp.columns]
        df_inp[nc_here] = scaler_saved.transform(df_inp[nc_here])

        pred  = model_saved.predict(df_inp)[0]
        prob  = model_saved.predict_proba(df_inp)[0][1]

        st.markdown("---")
        st.subheader("🔮 Result")
        r1,r2 = st.columns(2)
        with r1:
            if pred==1:
                st.markdown(f'<div class="churn-yes"><h2>⚠️ WILL CHURN</h2>'
                            f'<p>Probability: <strong>{prob*100:.1f}%</strong></p></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="churn-no"><h2>✅ WILL NOT CHURN</h2>'
                            f'<p>Probability: <strong>{prob*100:.1f}%</strong></p></div>',
                            unsafe_allow_html=True)
        with r2:
            fig,ax=plt.subplots(figsize=(4,3))
            ax.barh(['No Churn','Churn'],[1-prob,prob],color=['steelblue','tomato'])
            ax.set_xlim(0,1); ax.set_title('Churn Probability')
            [ax.text(v+0.01,i,f'{v*100:.1f}%',va='center') for i,v in enumerate([1-prob,prob])]
            plt.tight_layout(); st.pyplot(fig); plt.close()

        csv = pd.DataFrame([{'Gender':gender,'Tenure':tenure,'Contract':contract,
                              'MonthlyCharges':monthly,'TotalCharges':total,
                              'Prediction':'Churn' if pred==1 else 'No Churn',
                              'Probability_%':round(prob*100,2)}]).to_csv(index=False).encode()
        st.download_button("⬇️ Download Result CSV", csv,
                           "prediction_result.csv","text/csv",use_container_width=True)

# ════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════════
elif page == "⭐ Feature Importance":
    st.title("⭐ Feature Importance")
    rf  = models['Random Forest']
    imp = pd.Series(rf.feature_importances_, index=feat_cols)
    n   = st.slider("Top N features",5,30,15)
    top = imp.nlargest(n).sort_values()
    fig,ax=plt.subplots(figsize=(10,max(5,n//2)))
    top.plot(kind='barh',ax=ax,
             color=plt.cm.Blues(np.linspace(0.4,0.9,len(top))),edgecolor='white')
    ax.set_title(f'Top {n} Feature Importances – Random Forest',fontsize=14,fontweight='bold')
    ax.set_xlabel('Importance Score')
    [ax.text(v+0.001,i,f'{v:.4f}',va='center',fontsize=9) for i,v in enumerate(top)]
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("Full Table")
    st.dataframe(pd.DataFrame({'Feature':imp.index,'Importance':imp.values})
                 .sort_values('Importance',ascending=False).reset_index(drop=True),
                 use_container_width=True)

# ════════════════════════════════════════════════════════════════
# ABOUT
# ════════════════════════════════════════════════════════════════
elif page == "ℹ About":
    st.title("ℹ About This Project")
    c1,c2=st.columns(2)
    with c1:
        st.markdown("""### Project Details
|  |  |
|---|---|
|**Project**|Customer Churn Prediction|
|**Week**|4 – Advanced ML|
|**Intern**|Ali Akaash|
|**Supervisor**|Qamar Naveed|
|**Due**|3 July 2026|

### Technologies
- Python 3.10+ | Scikit-learn 1.4
- Pandas & NumPy
- Matplotlib & Seaborn
- Streamlit 1.32 | Joblib""")
    with c2:
        st.markdown("""### Techniques Applied
- Feature Engineering & Variance Threshold Selection
- Label Encoding & One-Hot Encoding
- StandardScaler Normalization
- GridSearchCV & RandomizedSearchCV
- Stratified K-Fold Cross Validation
- Model Persistence (Joblib)

### Project Structure
```
Week4-Advanced-ML-Optimization/
├── data/churn.csv
├── notebooks/churn_analysis.ipynb
├── models/best_model.pkl
├── models/scaler.pkl
├── app.py
├── requirements.txt
└── README.md
```""")
    st.markdown("---")
    st.success("✅ All Week 4 deliverables completed.")
