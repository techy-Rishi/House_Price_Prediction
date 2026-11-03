import streamlit as st
import pandas as pd
import numpy as np
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Lasso, Ridge
import xgboost as xgb

st.set_page_config(page_title="House Price EDA & Prediction", layout="wide")

def filling_null(features):
    features['Functional'] = features['Functional'].fillna('Typ')
    features['Electrical'] = features['Electrical'].fillna(features['Electrical'].mode()[0])
    features['KitchenQual'] = features['KitchenQual'].fillna(features['KitchenQual'].mode()[0])
    features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
    features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
    features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])
    features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
    features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    
    cat_features = features.select_dtypes(include='object').columns
    num_features = features.select_dtypes(exclude='object').columns
    features[cat_features] = features[cat_features].fillna('None')
    features[num_features] = features[num_features].fillna(0)
    return features

def new_features(features):
    features['HasPool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    features['Has2ndFloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    features['HasGarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    features['HasBsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    features['HasFireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
    
    features['TotalSF'] = features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']
    features['Total_Bathrooms'] = (
        features['FullBath'] + (0.5 * features['HalfBath']) +
        features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath'])
    )
    features['Total_porch_sf'] = (
        features['OpenPorchSF'] + features['3SsnPorch'] +
        features['EnclosedPorch'] + features['ScreenPorch']
    )
    features['YearsSinceRemodel'] = features['YrSold'].astype(int) - features['YearRemodAdd'].astype(int)
    return features

@st.cache_resource
def run_pdf_pipeline():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    test_id = test['Id']
    
    train.drop('Id', axis=1, inplace=True)
    test.drop('Id', axis=1, inplace=True)
    
    # Outlier removal and log transformation from PDF
    train = train[train['GrLivArea'] < 4500].copy()
    raw_train_reference = train.drop('SalePrice', axis=1).copy()
    train['SalePrice'] = np.log1p(train['SalePrice'])
    
    y = train['SalePrice'].reset_index(drop=True)
    train_features = train.drop('SalePrice', axis=1)
    test_features = test.copy()
    
    features = pd.concat((train_features, test_features)).reset_index(drop=True)
    
    features['MSSubClass'] = features['MSSubClass'].apply(str)
    features['MoSold'] = features['MoSold'].astype(str)
    features['YrSold'] = features['YrSold'].astype(str)
    
    features = features.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1)
    features = filling_null(features)
    features = new_features(features)
    
    # Compute Box-Cox lambdas so single user inputs can be transformed identically
    numerical_columns = features.select_dtypes(exclude='object').columns
    skewed = features[numerical_columns].apply(lambda x: x.skew()).sort_values(ascending=False)
    high_skew = skewed[abs(skewed) > 0.5].index
    
    lambdas = {}
    for col in high_skew:
        lambdas[col] = boxcox_normmax(features[col] + 1)
        features[col] = boxcox1p(features[col], lambdas[col])
        
    features_encoded = pd.get_dummies(features).reset_index(drop=True)
    
    X = features_encoded.iloc[:len(y), :]
    X_submission = features_encoded.iloc[len(y):, :]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    
    # Train all 3 models from the PDF
    ls = Lasso(alpha=0.1)
    ls.fit(X_train, y_train)
    
    ridge = Ridge(alpha=0.1)
    ridge.fit(X_train, y_train)
    
    xgb_regress = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.1, random_state=0)
    xgb_regress.fit(X_train, y_train)
    
    models = {
        "Ridge Regression (alpha=0.1)": ridge,
        "Lasso Regression (alpha=0.1)": ls,
        "XGBoost Regressor": xgb_regress
    }
    
    metrics = []
    for name, m in models.items():
        tr_pred = m.predict(X_train)
        te_pred = m.predict(X_test)
        metrics.append({
            "Model": name,
            "Train RMSE": round(np.sqrt(mean_squared_error(y_train, tr_pred)), 5),
            "Test RMSE": round(np.sqrt(mean_squared_error(y_test, te_pred)), 5),
            "Test MSE": round(mean_squared_error(y_test, te_pred), 5),
            "Test MAE": round(mean_absolute_error(y_test, te_pred), 5)
        })
        
    # Generate submission dataframe from PDF
    test_pred = np.expm1(ridge.predict(X_submission))
    submission = pd.DataFrame({'Id': test_id, 'SalePrice': test_pred})
    
    return models, pd.DataFrame(metrics), submission, raw_train_reference, lambdas, X.columns

st.title("🏠 House Price EDA and Prediction (Ames Housing Dataset)")

try:
    models, metrics_df, submission_df, ref_df, lambdas, encoded_cols = run_pdf_pipeline()
except FileNotFoundError:
    st.error("Please make sure `train.csv` and `test.csv` from the `house-prices-advanced-regression-techniques` dataset are uploaded to the root of your GitHub repository.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🔮 Predict Custom House Price", "📊 Model Evaluation Metrics", "📥 Kaggle Submission (`result_ridge.csv`)"])

with tab1:
    st.subheader("Predict House Price Using Your Notebook's Exact 324-Feature Pipeline")
    selected_model = st.selectbox("Choose Model", list(models.keys()))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        overall_qual = st.slider("Overall Quality (OverallQual)", 1, 10, 7)
        overall_cond = st.slider("Overall Condition (OverallCond)", 1, 10, 5)
        gr_liv_area = st.number_input("Living Area Sq Ft (GrLivArea)", 300, 4499, 1500)
        lot_area = st.number_input("Lot Area Sq Ft (LotArea)", 1000, 100000, 9600)
    with col2:
        total_bsmt_sf = st.number_input("Total Basement Sq Ft (TotalBsmtSF)", 0, 3500, 1000)
        first_flr_sf = st.number_input("1st Floor Sq Ft (1stFlrSF)", 300, 3500, 1000)
        second_flr_sf = st.number_input("2nd Floor Sq Ft (2ndFlrSF)", 0, 2500, 500)
        full_bath = st.slider("Full Bathrooms (FullBath)", 0, 4, 2)
    with col3:
        garage_cars = st.slider("Garage Cars (GarageCars)", 0, 4, 2)
        garage_area = st.number_input("Garage Area Sq Ft (GarageArea)", 0, 1500, 500)
        year_remod = st.number_input("Remodel Year (YearRemodAdd)", 1950, 2010, 2005)
        neighborhood = st.selectbox("Neighborhood", sorted(ref_df['Neighborhood'].dropna().unique()))

    if st.button("Predict SalePrice", type="primary"):
        # Start with median/mode baseline row for the remaining 68 background features
        base_row = ref_df.iloc[[0]].copy()
        for c in ref_df.columns:
            if ref_df[c].dtype == 'object':
                base_row[c] = ref_df[c].mode()[0]
            else:
                base_row[c] = ref_df[c].median()
                
        # Override with user inputs
        base_row['OverallQual'] = overall_qual
        base_row['OverallCond'] = overall_cond
        base_row['GrLivArea'] = gr_liv_area
        base_row['LotArea'] = lot_area
        base_row['TotalBsmtSF'] = total_bsmt_sf
        base_row['1stFlrSF'] = first_flr_sf
        base_row['2ndFlrSF'] = second_flr_sf
        base_row['FullBath'] = full_bath
        base_row['GarageCars'] = garage_cars
        base_row['GarageArea'] = garage_area
        base_row['YearRemodAdd'] = year_remod
        base_row['Neighborhood'] = neighborhood
        
        base_row['MSSubClass'] = base_row['MSSubClass'].apply(str)
        base_row['MoSold'] = base_row['MoSold'].astype(str)
        base_row['YrSold'] = base_row['YrSold'].astype(str)
        base_row = base_row.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1)
        base_row = new_features(base_row)
        
        for col, lmbda in lambdas.items():
            if col in base_row.columns:
                base_row[col] = boxcox1p(base_row[col], lmbda)
                
        base_encoded = pd.get_dummies(base_row).reindex(columns=encoded_cols, fill_value=0)
        pred_log = models[selected_model].predict(base_encoded)[0]
        pred_price = np.expm1(pred_log)
        st.success(f"### Estimated Sale Price: ${pred_price:,.2f}")

with tab2:
    st.subheader("Lasso, Ridge, and XGBoost Results (From PDF)")
    st.dataframe(metrics_df, use_container_width=True)

with tab3:
    st.subheader("Test Set Predictions (`result_ridge.csv`)")
    st.dataframe(submission_df.head(15), use_container_width=True)
    csv_bytes = submission_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download result_ridge.csv", data=csv_bytes, file_name="result_ridge.csv", mime="text/csv")
