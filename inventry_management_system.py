import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import streamlit as st


# Configure Streamlit page layout
st.set_page_config(page_title="Scanner Data Analysis", layout="wide")

# Load the dataset
file_path = r'C:\Users\labha\Desktop\Gate 2025\scanner_data.csv'
scanner_data = pd.read_csv(file_path)

# Display initial dataset
st.title("Scanner Data Analysis")
st.subheader("Initial Dataset Preview")
st.write(scanner_data.head())
st.write(scanner_data.info())

# Handle and format date columns
if 'Custom…' in scanner_data.columns:
    scanner_data = scanner_data.rename(columns={'Custom…': 'Date'})

scanner_data['Date'] = pd.to_datetime(scanner_data['Date'], errors='coerce')
scanner_data = scanner_data.sort_values('Date')

# Generate synthetic demand data
np.random.seed(42)
num_points = len(scanner_data)
seasonal_component = 50 + 15 * np.sin(np.linspace(0, 12 * np.pi, num_points))
trend_component = np.linspace(20, 80, num_points)
random_noise = np.random.normal(0, 5, num_points)
synthetic_demand = seasonal_component + trend_component + random_noise
scanner_data['demand'] = synthetic_demand

# Plot demand data
st.subheader("Synthetic Demand Data")
st.line_chart(scanner_data[['Date', 'demand']].set_index('Date'))

# User input for demand levels
st.subheader("Filter Demand Data by Level")
selected_level = st.radio(
    "Select demand level to filter:",
    ('Low', 'Medium', 'High')
)

# Bin and filter demand based on user input
bins = [0, 40, 80, 120]
labels = ['Low', 'Medium', 'High']
scanner_data['demand_category'] = pd.cut(scanner_data['demand'], bins=bins, labels=labels)

filtered_data = scanner_data[scanner_data['demand_category'] == selected_level]
st.write(f"Data filtered by demand level '{selected_level}':")
st.write(filtered_data)

# Confusion matrix for categorized demand
y_true_binned = pd.cut(scanner_data['demand'], bins=bins, labels=labels).astype(str)
y_pred_binned = pd.cut(scanner_data['demand'], bins=bins, labels=labels).astype(str)
cm = confusion_matrix(y_true_binned, y_pred_binned, labels=labels)

fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues, ax=ax)
st.subheader("Confusion Matrix for Binned Demand Predictions")
st.pyplot(fig)

# Date filtering function
def get_items_sold_on_date(date: str, data: pd.DataFrame) -> pd.DataFrame:
    input_date = pd.to_datetime(date, format='%m/%d/%Y', errors='coerce')
    return data[data['Date'] == input_date]

# User input for date filtering
st.subheader("Check Items Sold on a Specific Date")
date_to_search = st.text_input("Enter the date (MM/DD/YYYY) to check items sold:")

if date_to_search:
    items_sold = get_items_sold_on_date(date_to_search, scanner_data)
    st.subheader(f"Items sold on {date_to_search}:")
    st.write(items_sold if not items_sold.empty else "No items were sold on this date.")

# Inventory optimization calculations
inventory_data = pd.read_csv(file_path)
inventory_data['Date'] = pd.to_datetime(inventory_data['Date'], format='%m/%d/%Y', errors='coerce')
daily_demand = inventory_data.groupby(['SKU', 'Date'])['Quantity'].sum().reset_index()
sku_stats = daily_demand.groupby('SKU')['Quantity'].agg(['mean', 'std']).fillna(0)
sku_stats.columns = ['average_daily_demand', 'std_daily_demand']

holding_cost_per_unit_per_year = 1.5
ordering_cost_per_order = 50
lead_time_days = 7
sku_stats['annual_demand'] = sku_stats['average_daily_demand'] * 365

def calculate_eoq(annual_demand, holding_cost, ordering_cost):
    return ((2 * annual_demand * ordering_cost) / holding_cost) ** 0.5

sku_stats['EOQ'] = calculate_eoq(sku_stats['annual_demand'], holding_cost_per_unit_per_year, ordering_cost_per_order)
service_level_z = 1.65
sku_stats['safety_stock'] = service_level_z * sku_stats['std_daily_demand'] * np.sqrt(lead_time_days)
sku_stats['reorder_point'] = (sku_stats['average_daily_demand'] * lead_time_days) + sku_stats['safety_stock']

st.subheader("Inventory Optimization Recommendations")
st.write(sku_stats[['average_daily_demand', 'std_daily_demand', 'EOQ', 'safety_stock', 'reorder_point']].head())

# Sample data for initial stock (can be replaced with user input in the future)
initial_stock = { 
    '0EM7L': 100,  # Example initial stock for SKU '0EM7L'
    '68BRQ': 150,
    'CZUZX': 200,
    '549KK': 250,
    'K8EHH': 300
}

# Convert initial stock data to DataFrame for merging
initial_stock_df = pd.DataFrame(list(initial_stock.items()), columns=['SKU', 'Initial_Stock'])

# Sample scanner data for demonstration purposes (replace with actual data loading as necessary)
scanner_data = {
    'SKU': ['0EM7L', '68BRQ', 'CZUZX', '549KK', 'K8EHH', '68BRQ'],
    'Quantity': [20, 35, 50, 10, 5, 40]
}
scanner_data_df = pd.DataFrame(scanner_data)

# Streamlit app interface
st.title("Inventory Management Dashboard")

# Step 3: Summarize total quantities sold per SKU
inventory_summary = scanner_data_df.groupby('SKU')['Quantity'].sum().reset_index()
inventory_summary.columns = ['SKU', 'Total_Quantity_Sold']

# Merge initial stock data with inventory summary
current_inventory = pd.merge(initial_stock_df, inventory_summary, on='SKU', how='left')
current_inventory['Current_Inventory_Level'] = (
    current_inventory['Initial_Stock'] - current_inventory['Total_Quantity_Sold'].fillna(0)
)

# Display the final inventory levels
st.subheader("Current Inventory Levels")
st.dataframe(current_inventory)

# Optional: Add a chart for visualization (e.g., bar chart)
st.bar_chart(current_inventory.set_index('SKU')['Current_Inventory_Level'])




