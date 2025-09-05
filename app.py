import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load the trained model, feature columns, and encoder
@st.cache_resource
def load_model():
    """Load all model artifacts"""
    try:
        with open("best_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("feature_names.pkl", "rb") as f:  # Updated filename
            feature_names = pickle.load(f)
        with open("encoder.pkl", "rb") as f:
            encoder = pickle.load(f)
        with open("scaler.pkl", "rb") as f:  # New scaler
            scaler = pickle.load(f)
        return model, feature_names, encoder, scaler
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.error("Please run the training script first to generate model files.")
        return None, None, None, None

def get_popular_brands():
    """Get list of popular brands (same as used in training)"""
    return ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Tata', 'Mahindra', 
            'Ford', 'Chevrolet', 'Renault', 'Nissan', 'Volkswagen',
            'BMW', 'Mercedes-Benz', 'Audi', 'Skoda', 'Fiat', 'Datsun']

# Preprocess input data (FIXED VERSION)
def preprocess_input(input_data, feature_names, encoder, scaler):
    """Preprocess input data to match the fixed training pipeline"""
    try:
        # Create DataFrame from input
        df = pd.DataFrame([input_data])
        
        # Calculate car age
        df["Car_Age"] = 2025 - df["year"]
        df = df.drop(["year"], axis=1)
        
        # Add luxury brand indicator
        luxury_brands = ['Mercedes-Benz', 'BMW', 'Audi', 'Jaguar', 'Volvo', 
                        'Land', 'Rover', 'Porsche', 'Ferrari', 'Lamborghini']
        df["Is_Luxury"] = df["Brand"].apply(
            lambda x: 1 if any(luxury in str(x) for luxury in luxury_brands) else 0
        )
        
        # Add mileage category
        df["Mileage_Category"] = pd.cut(
            df["km_driven"], 
            bins=[0, 20000, 50000, 100000, 200000, float('inf')],
            labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'],
            include_lowest=True
        )
        
        # Add age category
        df["Age_Category"] = pd.cut(
            df["Car_Age"],
            bins=[0, 3, 7, 12, 20, float('inf')],
            labels=['New', 'Recent', 'Medium', 'Old', 'Very_Old'],
            include_lowest=True
        )
        
        # Handle brand grouping (same as training)
        popular_brands = get_popular_brands()
        df['Brand'] = df['Brand'].apply(
            lambda x: x if x in popular_brands else 'Other'
        )
        
        # Encode categorical features
        categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner', 
                           'Brand', 'Mileage_Category', 'Age_Category']
        
        encoded_data = encoder.transform(df[categorical_cols])
        feature_names_encoded = encoder.get_feature_names_out(categorical_cols)
        encoded_df = pd.DataFrame(encoded_data, columns=feature_names_encoded, index=df.index)
        
        # Drop original categorical columns and add encoded features
        df_processed = df.drop(categorical_cols, axis=1)
        df_processed = pd.concat([df_processed, encoded_df], axis=1)
        
        # Scale numerical features
        numerical_features = ['km_driven', 'Car_Age', 'Is_Luxury']
        if scaler is not None:
            df_processed[numerical_features] = scaler.transform(df_processed[numerical_features])
        
        # Align with training features
        df_processed = df_processed.reindex(columns=feature_names, fill_value=0)
        
        return df_processed
        
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        return None

def load_sample_data():
    """Load and display sample data safely"""
    try:
        df = pd.read_csv("cardekho.csv")
        return df
    except FileNotFoundError:
        st.warning("Dataset file 'cardekho.csv' not found. Sample data not available.")
        return None

# Main app
def main():
    st.set_page_config(
        page_title="Car Price Predictor", 
        page_icon="🚗",
        layout="wide"
    )
    
    st.title("Enhanced Car Price Prediction App")
    st.markdown("---")
    st.write("This app predicts the selling price of a used car using an improved machine learning model.")
    
    # Load model
    model, feature_names, encoder, scaler = load_model()
    if model is None:
        st.stop()
    
    # Load sample data for dropdowns
    df = load_sample_data()
    
    # Create two columns for better layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("🔧 Car Details")
        
        # Brand selection
        if df is not None:
            # Get unique brands from dataset
            unique_brands = sorted([name.split()[0] for name in df["name"].unique()])
            unique_brands = list(set(unique_brands))  # Remove duplicates
        else:
            unique_brands = get_popular_brands()
        
        brand = st.selectbox("Car Brand", unique_brands, index=0)
        
        # Year input
        current_year = 2025
        year = st.slider(
            "Year of Manufacture", 
            min_value=1990, 
            max_value=current_year, 
            value=2018,
            help="Select the manufacturing year of the car"
        )
        
        # Kilometers driven
        km_driven = st.number_input(
            "Kilometers Driven", 
            min_value=0, 
            max_value=500000,
            value=40000,
            step=1000,
            help="Total kilometers driven by the car"
        )
        
        # Fuel type
        fuel_options = ["Petrol", "Diesel", "CNG", "LPG"]
        fuel = st.selectbox("Fuel Type", fuel_options)
        
        # Seller type
        seller_options = ["Individual", "Dealer"]
        seller_type = st.selectbox("Seller Type", seller_options)
        
        # Transmission
        transmission_options = ["Manual", "Automatic"]
        transmission = st.selectbox("Transmission", transmission_options)
        
        # Owner
        owner_options = ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"]
        owner = st.selectbox("Owner", owner_options)
        
        # Predict button
        predict_button = st.button("Predict Price", type="primary", use_container_width=True)
    
    with col2:
        if predict_button:
            # Create input dictionary
            input_data = {
                "Brand": brand,
                "year": year,
                "km_driven": km_driven,
                "fuel": fuel,
                "seller_type": seller_type,
                "transmission": transmission,
                "owner": owner
            }
            
            # Show input summary
            st.subheader("📋 Input Summary")
            input_df = pd.DataFrame([input_data])
            input_df["Car_Age"] = 2025 - input_df["year"]
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**Car Details:**")
                st.write(f"• Brand: {brand}")
                st.write(f"• Year: {year} ({2025-year} years old)")
                st.write(f"• Kilometers: {km_driven:,} km")
                st.write(f"• Fuel: {fuel}")
            
            with col_b:
                st.write("**Ownership Details:**")
                st.write(f"• Seller: {seller_type}")
                st.write(f"• Transmission: {transmission}")
                st.write(f"• Owner: {owner}")
            
            # Preprocess input
            with st.spinner("Processing input data..."):
                processed_data = preprocess_input(input_data, feature_names, encoder, scaler)
            
            if processed_data is not None:
                # Make prediction
                with st.spinner("Predicting price..."):
                    predicted_price = model.predict(processed_data)[0]
                
                # Display prediction with styling
                st.subheader("Predicted Price")
                
                # Create metrics display
                col_price1, col_price2 = st.columns(2)
                with col_price1:
                    st.metric(
                        "Predicted Price", 
                        f"₹{predicted_price:,.0f}",
                        help="Predicted selling price based on the input features"
                    )
                
                with col_price2:
                    st.metric(
                        "In Lakhs", 
                        f"₹{predicted_price/100000:.2f}L",
                        help="Price in lakhs (1 lakh = 100,000)"
                    )
                
                # Price range estimation
                st.info(f"💡 **Price Range Estimate**: ₹{predicted_price*0.9:,.0f} - ₹{predicted_price*1.1:,.0f}")
                
                # Show feature importance if available
                if hasattr(model, 'feature_importances_'):
                    st.subheader("📊 Feature Importance")
                    
                    # Get top 10 important features
                    importances = model.feature_importances_
                    feature_importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False).head(10)
                    
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.barh(range(len(feature_importance_df)), feature_importance_df['Importance'], color='skyblue')
                    ax.set_yticks(range(len(feature_importance_df)))
                    ax.set_yticklabels(feature_importance_df['Feature'])
                    ax.set_xlabel("Relative Importance")
                    ax.set_title("Top 10 Most Important Features")
                    ax.grid(axis='x', alpha=0.3)
                    
                    # Add value labels on bars
                    for i, bar in enumerate(bars):
                        width = bar.get_width()
                        ax.text(width, bar.get_y() + bar.get_height()/2, 
                               f'{width:.3f}', ha='left', va='center', fontsize=9)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
            else:
                st.error("❌ Error processing input data. Please check your inputs and try again.")
    
    # Additional information sections
    st.markdown("---")
    
    # Model information
    with st.expander("ℹ️ About the Model", expanded=False):
        st.markdown("""
        **Model Details:**
        - **Algorithm**: Random Forest Regressor with optimized hyperparameters
        - **Features**: Uses car age, mileage, brand, fuel type, and other characteristics
        - **Training**: Trained on real car sales data with proper data cleaning and feature engineering
        
        **Key Improvements:**
        - ✅ Proper outlier handling and data cleaning
        - ✅ Safe feature engineering (no target leakage)
        - ✅ Consistent preprocessing pipeline
        - ✅ Hyperparameter optimization using RandomizedSearchCV
        
        **Accuracy Metrics:**
        - The model achieves good R² scores on test data
        - Predictions are based on market trends and car characteristics
        """)
    
    # Sample data display
    if df is not None:
        with st.expander("📊Sample Dataset", expanded=False):
            st.subheader("Dataset Overview")
            
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.metric("Total Cars", f"{len(df):,}")
            with col_stat2:
                st.metric("Unique Brands", len(df["name"].str.split().str[0].unique()))
            with col_stat3:
                st.metric("Year Range", f"{df['year'].min()}-{df['year'].max()}")
            with col_stat4:
                st.metric("Avg Price", f"₹{df['selling_price'].mean()/100000:.1f}L")
            
            st.subheader("Sample Data")
            st.dataframe(df.head(10), use_container_width=True)
    
    # Tips section
    with st.expander("Tips for Better Predictions", expanded=False):
        st.markdown("""
        **For More Accurate Predictions:**
        1. **Be Accurate**: Enter the exact year and mileage
        2. **Choose Correct Brand**: Select the exact brand name
        3. **Owner History**: First owner cars typically have higher values
        4. **Fuel Type**: Diesel cars often have different valuations than petrol
        5. **Transmission**: Automatic transmission can affect the price significantly
        
        **Market Factors Not Considered:**
        - Current market conditions
        - Regional price variations  
        - Car condition and maintenance history
        - Accident history
        - Modifications or upgrades
        """)

if __name__ == "__main__":
    main()