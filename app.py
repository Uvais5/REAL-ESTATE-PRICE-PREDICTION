import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page configuration
st.set_page_config(page_title="Apartment Price Predictor", layout="wide")


# 1. LOAD DATA & MODEL BUNDLE

@st.cache_resource
def load_bundle():
    model_path = "real_estate_model_bundle.pkl"
    if not os.path.exists(model_path):
        st.error(f"❌ Model file '{model_path}' not found!")
        st.stop()
    return joblib.load(model_path)

@st.cache_data
def get_districts_from_csv():
    csv_path = "case_data.csv" 
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if 'District' in df.columns:
                return sorted(df['District'].dropna().unique().tolist())
        except: pass
    return []

bundle = load_bundle()
districts_list = get_districts_from_csv()

model = bundle['model']
feature_names = bundle['feature_names']
target_maps = bundle['target_maps']
medians = bundle['medians']
global_mean = bundle['global_mean']
global_median = bundle['global_median']


handover_options = {
    "Ready Now (Built)": 0,
    "In 1 Year": 1,
    "In 2 Years": 2,
    "In 3 Years": 3,
    "In 4 Years": 4,
    "5+ Years": 5
}


# 2. LINK PARSING LOGIC (Samolet.ru Example)

def parse_listing_link(url):
    try:
        if "samolet.ru" in url:
            # Mock data representing what the scraper finds
            return {
                "total_area": 55.4,
                "living_area": 30.2,
                "kitchen_area": 12.5,
                "balcony_area": 4.5,
                "bathroom_area": 5.5,
                "floor": 14,
                "floors_total": 25,
                "rooms": 2,
                "ceiling_height": 2.8,
                "years_to_wait": "In 1 Year" # Match the key in handover_options
            }
    except: pass
    return None


# 3. UI TABS

st.title("🏠 Smart Apartment Price Predictor")
tab1, tab2 = st.tabs(["🔗 Link-Based Input", "⌨️ Manual Adjustment"])

if 'input_data' not in st.session_state:
    st.session_state['input_data'] = {
        'total_area': 55, 'living_area': 30, 'kitchen_area': 10, 'balcony_area': 5, 'bathroom_area': 6,
        'floor': 5, 'floors_total': 12, 'ceiling_height': 2.8, 'rooms': 2, 'years_to_wait': "Ready Now (Built)"
    }

# TAB 1: LINK INPUT
with tab1:
    st.subheader("Extract Data from Listing")
    listing_url = st.text_input("Paste Samolet.ru URL", placeholder="https://samolet.ru/...")
    if st.button("Extract Data"):
        scraped = parse_listing_link(listing_url)
        if scraped:
            st.session_state['input_data'].update(scraped)
            st.success("Data extracted! Switch to 'Manual Adjustment' to review.")
        else:
            st.error("Could not parse this link.")

# TAB 2: MANUAL INPUTS 
with tab2:
    col1, col2, col3 = st.columns(3)
    d = st.session_state['input_data']

    with col1:
        st.subheader("📐 Dimensions")
        total_area = st.number_input("Total Area (sqm)", 20, 300, int(d['total_area']))
        living_area = st.number_input("Living Area (sqm)", 10, 200, int(d['living_area']))
        kitchen_area = st.number_input("Kitchen Area (sqm)", 5, 50, int(d['kitchen_area']))
        balcony_area = st.number_input("Balcony Area (sqm)", 0, 50, int(d['balcony_area']))
        bathroom_area = st.number_input("Bathroom Area (sqm)", 2, 20, int(d['bathroom_area']))

    with col2:
        st.subheader("🏢 Building Info")
        floor = st.number_input("Floor", 1, 50, int(d['floor']))
        floors_total = st.number_input("Total Floors", 1, 60, int(d['floors_total']))
        ceiling_height = st.number_input("Ceiling Height (m)", 2.2, 5.5, float(d['ceiling_height']))
        rooms = st.number_input("Number of Rooms", 1, 6, int(d['rooms']))
        
        # FIXED: Selectbox instead of Slider
        handover_selection = st.selectbox(
            "Years to Handover", 
            options=list(handover_options.keys()),
            index=list(handover_options.keys()).index(d['years_to_wait']) if d['years_to_wait'] in handover_options else 0
        )
        # Convert selected text back to numeric for the model
        years_to_wait = handover_options[handover_selection]

    with col3:
        st.subheader("📍 Details")
        district = st.selectbox("District", districts_list) if districts_list else st.text_input("District", "Central")
        property_type = st.selectbox("Property Type", ["Студия", "1-комн", "2-комн", "3-комн", "4-комн"])
        class_type = st.selectbox("Class", ["Эконом", "Комфорт", "Бизнес", "Премиум"])
        building_type = st.selectbox("Building Type", ["Монолит", "Панель", "Кирпич"])
        finishing = st.selectbox("Finishing", ["Без отделки", "Чистовая", "Предчистовая"])
        mortgage = st.selectbox("Mortgage", ["Да", "Нет"])
        subsidies = st.selectbox("Subsidies", ["Да", "Нет"])


# 4. PREDICTION ENGINE

if st.button("Calculate Price", type="primary", use_container_width=True):
    data = {
        'TotalArea': total_area, 'LivingArea': living_area, 'KitchenArea': kitchen_area,
        'BalconyArea_Sum': balcony_area, 'BathroomArea_Sum': bathroom_area,
        'Floor': floor, 'FloorsTotal': floors_total, 'CeilingHeight': ceiling_height,
        'YearsToWait': years_to_wait, 'Rooms': rooms, 'District': district,
        'Class': class_type, 'BuildingType': building_type, 'Finishing': finishing,
        'Mortgage': mortgage, 'Subsidies': subsidies, 'PropertyType': property_type
    }
    
    # Feature Engineering logic
    data['Is_First_Floor'] = int(floor <= 1)
    data['Is_Top_Floor'] = int(floor == floors_total)
    data['Floor_Position'] = floor / floors_total if floors_total != 0 else 0
    data['Is_Middle_Floor'] = int((floor > 1) and (floor < floors_total))
    data['Living_to_Total'] = living_area / total_area if total_area != 0 else 0
    data['Kitchen_to_Total'] = kitchen_area / total_area if total_area != 0 else 0
    data['Balcony_to_Total'] = balcony_area / total_area if total_area != 0 else 0
    data['Bathroom_to_Total'] = bathroom_area / total_area if total_area != 0 else 0
    data['Is_Ready'] = int(years_to_wait == 0)
    data['Has_Finishing'] = int(finishing != "Без отделки")
    data['Has_Mortgage'] = int(mortgage == "Да")
    data['Has_Subsidies'] = int(subsidies == "Да")
    data['TotalArea_Squared'] = total_area ** 2
    data['Floor_Squared'] = floor ** 2
    data['TotalArea_Log'] = np.log1p(total_area)
    data['CeilingHeight_Squared'] = ceiling_height ** 2

    input_df = pd.DataFrame([data])
    
    # Encoding & Alignment
    for col, enc in target_maps.items():
        if col in input_df.columns:
            val = input_df[col].iloc[0]
            m_map = enc.get('mean_map', enc.get('mean', {}))
            input_df[f'{col}_Price_Mean'] = m_map.get(val, global_mean)
            input_df[f'{col}_Price_Median'] = enc.get('median_map', enc.get('median', {})).get(val, global_median)
            input_df[f'{col}_Price_Std'] = enc.get('std_map', enc.get('std', {})).get(val, 0)
            input_df[f'{col}_Price_Count'] = enc.get('count_map', enc.get('count', {})).get(val, 0)

    final_features = input_df.reindex(columns=feature_names)
    for col in feature_names:
        if pd.isna(final_features[col].iloc[0]):
            final_features[col] = medians.get(col, 0)
    
    try:
        prediction = model.predict(final_features)[0]
        st.markdown("---")
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Price per m²", f"{int(prediction):,} ₽")
        res_col2.metric("Estimated Total Price", f"{int(prediction * total_area):,} ₽")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
