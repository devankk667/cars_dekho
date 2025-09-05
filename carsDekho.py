# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import numpy as np
# import pickle
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# # Load the dataset
# df = pd.read_csv('cardekho.csv')
# #drop unncessary columns
# df["Brand"] = df["name"].str.split(" ").str[0]
# df.drop(["name"], axis=1, inplace=True)
# df.rename(columns={"selling_price": "Price"}, inplace=True)
# df["Car_Age"] = 2025 - df["year"]
# df.drop(["year"], axis=1, inplace=True)
# #one hot encoding
# df=pd.get_dummies(df, drop_first=True)
# # print(df.head())

# #split the training and testing data
# X=df.drop("Price", axis=1)  #everything except price
# y=df["Price"]
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
# #feature scaling using scaler
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# #train the models 
# # models={
# #     "LinearRegression":LinearRegression(),
# #     "Ridge":Ridge(alpha=10),
# #     "Lasso":Lasso(alpha=0.01),
# #     "ElasticNet":ElasticNet(alpha=0.01, l1_ratio=0.5)
# # }
# #for each model, train it and make predictions using loop in models
# # for model_name, model in models.items():
# #     model.fit(X_train, y_train)
# #     y_pred=model.predict(X_test)
# #     #print the scores,rmse and mae for each model
# #     print(f"{model_name}: R²={r2_score(y_test, y_pred):.3f}, "
# #           f"RMSE={np.sqrt(mean_squared_error(y_test, y_pred)):.3f}, "
# #           f"MAE={mean_absolute_error(y_test, y_pred):.3f}")
          
# #every model has almost same r2 scorei.e 0.530),expect elastic net(i.e 0.531) so we will go with elastic net

# #tune the elastic net model using grid search
# # param_grid={
# #     "alpha":np.logspace(-4,4,20),
# #     "l1_ratio":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
# # }
# # grid_search=GridSearchCV(
# #     ElasticNet(),
# #     param_grid,
# #     scoring="neg_mean_squared_error",
# #     cv=5,
# #     n_jobs=-1,
# #     verbose=2
# #     )
# # grid_search.fit(X_train,y_train)
# # print(grid_search.best_params_)
# #best fits, alpha=0.0336, l1_ratio=0.6
# #train the model with best params so no retrain needed
# # model=ElasticNet(alpha=0.0336, l1_ratio=0.6)
# # model.fit(X_train,y_train)
# # y_pred=model.predict(X_test)
# # print(f"ElasticNet: R²={r2_score(y_test, y_pred):.3f}, "
# #           f"RMSE={np.sqrt(mean_squared_error(y_test, y_pred)):.3f}, "
# #           f"MAE={mean_absolute_error(y_test, y_pred):.3f}")


# #since r2 small trying random forest,xgboost,lightgbm and catboost
# from sklearn.ensemble import RandomForestRegressor

# # Boosting libraries
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor
# models = {
#     "ElasticNet": ElasticNet(alpha=0.0336, l1_ratio=0.6),
#     "RandomForest": RandomForestRegressor(
#         n_estimators=200, random_state=42, n_jobs=-1
#     ),
#     "XGBoost": XGBRegressor(
#         n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1
#     ),
#     "LightGBM": LGBMRegressor(
#         n_estimators=500, learning_rate=0.05, max_depth=-1, random_state=42, n_jobs=-1
#     ),
#     "CatBoost": CatBoostRegressor(
#         iterations=500, learning_rate=0.05, depth=6, random_state=42, verbose=0
#     )
# }

# #train the models
# results = {}
# y_preds = {}

# # for name, model in models.items():
# #     if name == "ElasticNet":
# #         model.fit(X_train_scaled, y_train)
# #         y_pred = model.predict(X_test_scaled)
# #     else:
# #         model.fit(X_train, y_train)
# #         y_pred = model.predict(X_test)

# #     results[name] = {
# #         "R²": r2_score(y_test, y_pred),
# #         "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
# #         "MAE": mean_absolute_error(y_test, y_pred)
# #     }
# #     y_preds[name] = y_pred

# # results_df = pd.DataFrame(results).T.sort_values("R²", ascending=False)
# # print("\n--- Model Comparison ---")
# # print(results_df)



# #using random forest best r2 score
# model=RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
# model.fit(X_train, y_train)
# y_pred=model.predict(X_test)
# print(f"RandomForest: R²={r2_score(y_test, y_pred):.3f}, "
#           f"RMSE={np.sqrt(mean_squared_error(y_test, y_pred)):.3f}, "
#           f"MAE={mean_absolute_error(y_test, y_pred):.3f}")

# #tuning randomforest using gridsearch cv
# # rf = RandomForestRegressor(random_state=42)
# # param_grid = {
# #     "n_estimators": [100, 200, 300],
# #     "max_depth": [10, 20, None],
# #     "min_samples_split": [2, 5, 10],
# #     "min_samples_leaf": [1, 2, 4],
# #     "max_features": ["auto", "sqrt"]
# # }

# # grid_search = GridSearchCV(
# #     rf,
# #     param_grid,
# #     cv=3,  # 3-fold cross validation
# #     scoring="r2",
# #     n_jobs=-1,
# #     verbose=2
# # )

# # grid_search.fit(X_train, y_train)

# # print("Best Params:", grid_search.best_params_)
# # print("Best CV R²:", grid_search.best_score_)

# # Best Params: {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
# # Best CV R²: 0.805989302714759

# #using best params
# # model=RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', random_state=42)
# # model.fit(X_train, y_train)
# # y_pred=model.predict(X_test)
# # print(f"RandomForest: R²={r2_score(y_test, y_pred):.3f}, "
# #           f"RMSE={np.sqrt(mean_squared_error(y_test, y_pred)):.3f}, "
# #           f"MAE={mean_absolute_error(y_test, y_pred):.3f}")

# #plot the actual vs predicted values
# # plt.scatter(y_test, y_pred)
# # plt.xlabel("Actual")
# # plt.ylabel("Predicted")
# # plt.show()

# #save the model
# with open("random_forest_model.pkl", "wb") as f:
#     pickle.dump(model, f)

# #load the model
# with open("random_forest_model.pkl", "rb") as f:
#     model = pickle.load(f)

# #make predictions
# # y_pred = model.predict(X_test)
# # print(f"RandomForest: R²={r2_score(y_test, y_pred):.3f}, "
# #           f"RMSE={np.sqrt(mean_squared_error(y_test, y_pred)):.3f}, "
# #           f"MAE={mean_absolute_error(y_test, y_pred):.3f}")

# #plot the actual vs predicted values
# plt.scatter(y_test, y_pred)
# plt.xlabel("Actual")
# plt.ylabel("Predicted")
# plt.show()



    



#deployment ready code
# Fixed and improved CarDekho model training code
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess(path="cardekho.csv"):
    """Load and preprocess the CarDekho dataset with proper data cleaning"""
    df = pd.read_csv(path)
    print(f"Original dataset shape: {df.shape}")
    print(f"Dataset info:\n{df.info()}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Handle missing values first
    df = df.dropna()
    
    # Remove outliers in selling_price (prices > 100 lakhs are likely errors)
    df = df[df['selling_price'] <= 10000000]  # 1 crore max
    df = df[df['selling_price'] > 25000]      # 25k min (realistic minimum)
    
    # Remove outliers in km_driven
    df = df[df['km_driven'] <= 500000]        # 5 lakh km max
    df = df[df['km_driven'] >= 0]             # No negative km
    
    # Extract brand from name
    df["Brand"] = df["name"].str.split(" ").str[0]
    df = df.drop(["name"], axis=1)
    
    # Rename target column
    df = df.rename(columns={"selling_price": "Price"})
    
    # Calculate car age
    df["Car_Age"] = 2025 - df["year"]
    df = df.drop(["year"], axis=1)
    
    # Remove cars older than 30 years (likely data errors)
    df = df[df["Car_Age"] <= 30]
    df = df[df["Car_Age"] >= 0]
    
    print(f"After cleaning shape: {df.shape}")
    
    # Feature Engineering (but don't create circular dependencies)
    # Don't use Price to create features that will predict Price!
    
    # Luxury brand indicator
    luxury_brands = ['Mercedes-Benz', 'BMW', 'Audi', 'Jaguar', 'Volvo', 
                    'Land', 'Rover', 'Porsche', 'Ferrari', 'Lamborghini']
    df["Is_Luxury"] = df["Brand"].apply(
        lambda x: 1 if any(luxury in str(x) for luxury in luxury_brands) else 0
    )
    
    # Mileage categories based on km_driven
    df["Mileage_Category"] = pd.cut(
        df["km_driven"], 
        bins=[0, 20000, 50000, 100000, 200000, float('inf')],
        labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'],
        include_lowest=True
    )
    
    # Age categories
    df["Age_Category"] = pd.cut(
        df["Car_Age"],
        bins=[0, 3, 7, 12, 20, float('inf')],
        labels=['New', 'Recent', 'Medium', 'Old', 'Very_Old'],
        include_lowest=True
    )
    
    # Handle categorical variables properly
    categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner', 
                       'Brand', 'Mileage_Category', 'Age_Category']
    
    # Check for high cardinality in Brand
    brand_counts = df['Brand'].value_counts()
    print(f"Number of unique brands: {len(brand_counts)}")
    
    # Keep only brands with at least 10 cars, group others as 'Other'
    common_brands = brand_counts[brand_counts >= 10].index
    df['Brand'] = df['Brand'].apply(lambda x: x if x in common_brands else 'Other')
    
    print(f"Brands after grouping: {df['Brand'].unique()}")
    
    return df

def encode_features(df, fit_encoder=True, encoder=None):
    """Properly encode categorical features"""
    categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner', 
                       'Brand', 'Mileage_Category', 'Age_Category']
    
    if fit_encoder:
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        encoded_data = encoder.fit_transform(df[categorical_cols])
        
        # Save encoder
        with open("encoder.pkl", "wb") as f:
            pickle.dump(encoder, f)
    else:
        if encoder is None:
            raise ValueError("Encoder must be provided when fit_encoder=False")
        encoded_data = encoder.transform(df[categorical_cols])
    
    # Create DataFrame with encoded features
    feature_names = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df.index)
    
    # Drop original categorical columns and add encoded features
    df_processed = df.drop(categorical_cols, axis=1).copy()
    df_processed = pd.concat([df_processed, encoded_df], axis=1)
    
    return df_processed, encoder

def split_and_scale_data(df):
    """Split data and apply scaling to numerical features"""
    X = df.drop("Price", axis=1)
    y = df["Price"]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    
    # Scale numerical features only (not the encoded categorical ones)
    numerical_features = ['km_driven', 'Car_Age', 'Is_Luxury']
    scaler = StandardScaler()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])
    
    # Save scaler
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_random_forest(X_train, y_train):
    """Train Random Forest with proper hyperparameter tuning"""
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_random = RandomizedSearchCV(
        estimator=rf, 
        param_distributions=param_grid,
        n_iter=20,  # Reduced for faster training
        cv=3, 
        scoring='r2',
        verbose=1, 
        random_state=42, 
        n_jobs=-1
    )
    
    rf_random.fit(X_train, y_train)
    print(f"Best RF params: {rf_random.best_params_}")
    print(f"Best RF CV score: {rf_random.best_score_:.3f}")
    
    return rf_random.best_estimator_

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"{model_name}:")
    print(f"  R² Score: {r2:.3f}")
    print(f"  RMSE: ₹{rmse:,.0f}")
    print(f"  MAE: ₹{mae:,.0f}")
    print("-" * 40)
    
    return r2, rmse, mae, y_pred

def save_model_artifacts(model, feature_names, encoder, scaler):
    """Save all model artifacts"""
    with open("best_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)
    print("✅ Model artifacts saved!")

def plot_results(y_test, y_pred, model_name):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.6, s=30)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Price (₹)')
    plt.ylabel('Predicted Price (₹)')
    plt.title(f'Actual vs Predicted Car Prices - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format axis in lakhs
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/100000:.1f}L'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/100000:.1f}L'))
    
    plt.tight_layout()
    plt.show()

def load_model_artifacts():
    """Load saved model artifacts"""
    try:
        with open("best_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)
        with open("encoder.pkl", "rb") as f:
            encoder = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, feature_names, encoder, scaler
    except FileNotFoundError as e:
        print(f"Error loading model artifacts: {e}")
        return None, None, None, None

def predict_price(car_data):
    """Predict price for new car data"""
    model, feature_names, encoder, scaler = load_model_artifacts()
    
    if model is None:
        print("Model not found. Please train the model first.")
        return None
    
    # Preprocess the new data same way as training
    df_new = pd.DataFrame([car_data])
    
    # Add derived features
    df_new["Car_Age"] = 2025 - df_new["year"]
    df_new = df_new.drop(["year"], axis=1)
    
    luxury_brands = ['Mercedes-Benz', 'BMW', 'Audi', 'Jaguar', 'Volvo', 
                    'Land', 'Rover', 'Porsche', 'Ferrari', 'Lamborghini']
    df_new["Is_Luxury"] = df_new["Brand"].apply(
        lambda x: 1 if any(luxury in str(x) for luxury in luxury_brands) else 0
    )
    
    df_new["Mileage_Category"] = pd.cut(
        df_new["km_driven"], 
        bins=[0, 20000, 50000, 100000, 200000, float('inf')],
        labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'],
        include_lowest=True
    )
    
    df_new["Age_Category"] = pd.cut(
        df_new["Car_Age"],
        bins=[0, 3, 7, 12, 20, float('inf')],
        labels=['New', 'Recent', 'Medium', 'Old', 'Very_Old'],
        include_lowest=True
    )
    
    # Encode categorical features
    df_processed, _ = encode_features(df_new, fit_encoder=False, encoder=encoder)
    
    # Scale numerical features
    numerical_features = ['km_driven', 'Car_Age', 'Is_Luxury']
    df_processed[numerical_features] = scaler.transform(df_processed[numerical_features])
    
    # Align with training features
    df_processed = df_processed.reindex(columns=feature_names, fill_value=0)
    
    # Predict
    predicted_price = model.predict(df_processed)[0]
    return predicted_price

# MAIN PIPELINE
if __name__ == "__main__":
    print(" CarDekho Price Prediction Model Training")
    print("=" * 50)
    
    try:
        # Load and preprocess data
        print(" Loading and preprocessing data...")
        df = load_and_preprocess("cardekho.csv")
        
        # Encode features
        print(" Encoding categorical features...")
        df_encoded, encoder = encode_features(df, fit_encoder=True)
        
        # Split and scale data
        print(" Splitting and scaling data...")
        X_train, X_test, y_train, y_test = split_and_scale_data(df_encoded)
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        # Train Random Forest model
        print(" Training Random Forest model...")
        rf_model = train_random_forest(X_train, y_train)
        
        # Evaluate model
        print("\n Model Evaluation:")
        r2, rmse, mae, y_pred = evaluate_model(rf_model, X_test, y_test, "Random Forest")
        
        # Save model artifacts
        print(" Saving model artifacts...")
        save_model_artifacts(rf_model, X_train.columns.tolist(), encoder, None)
        
        # Plot results
        print(" Plotting results...")
        plot_results(y_test, y_pred, "Random Forest")
        
        # Example prediction
        print("\n Example Prediction:")
        sample_car = {
            "Brand": "Maruti",
            "year": 2018,
            "km_driven": 40000,
            "fuel": "Petrol",
            "seller_type": "Individual", 
            "transmission": "Manual",
            "owner": "First Owner"
        }
        
        predicted_price = predict_price(sample_car)
        if predicted_price:
            print(f"Sample car details: {sample_car}")
            print(f"Predicted price: ₹{predicted_price:,.0f} (₹{predicted_price/100000:.1f}L)")
        
        print("\n✅ Model training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        import traceback
        traceback.print_exc()