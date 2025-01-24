import traceback  # For error handling and stack trace
import csv  # For CSV file operations
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting graphs
import tensorflow as tf  # For machine learning and deep learning
from matplotlib.backends.backend_pdf import PdfPages  # For saving plots to PDF
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder  # For data preprocessing
from sklearn.model_selection import train_test_split  # For splitting datasets
from tensorflow.keras.models import Sequential  # For building sequential models
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input  # For LSTM model layers
from sklearn.metrics import mean_squared_error  # For evaluating model performance
import tkinter as tk  # For GUI application
from tkinter import messagebox, ttk, filedialog  # For GUI components
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # For embedding plots in Tkinter
from statsmodels.tsa.stattools import adfuller  # For statistical tests
import logging  # For logging events
import math  # For mathematical operations


# Configure logging to record events and errors

logging.basicConfig(filename='forecast_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define global file paths for inventory and sales data

inventory_file = 'inventory.csv'
inventory_log_file = 'inventory_log.csv'
sales_log_file = 'sales_data.csv'

def load_inventory():
    """Load inventory data from CSV."""

    """Load inventory data from CSV."""
    try:
        inventory_data = pd.read_csv(inventory_file)
        inventory_data['product_id'] = inventory_data['product_id'].astype(str)
        return inventory_data.set_index('product_id')
    except FileNotFoundError:
        empty_df = pd.DataFrame(columns=['product_id', 'product_name', 'quantity'])
        empty_df.to_csv(inventory_file, index=False)
        return empty_df

def save_inventory(inventory_data):
    """Save inventory data to CSV."""

    """Save inventory data to CSV."""
    inventory_data.to_csv(inventory_file)

def log_event(event_type, product_id, product_name, quantity):
    """Log actions to a CSV file with a formatted timestamp."""

    """Log actions to a CSV file with a formatted timestamp."""
    current_timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = pd.DataFrame({
        'event_type': [event_type],
        'product_id': [product_id],
        'product_name': [product_name],
        'quantity': [quantity],
        'timestamp': [current_timestamp]
    })
    log_entry.to_csv(inventory_log_file, mode='a', header=not pd.io.common.file_exists(inventory_log_file), index=False)

def load_and_prepare_data():
    """Load and prepare sales data for analysis."""

    try:
        sales_data = pd.read_csv('sales_data.csv')
        sales_data['product_id'] = sales_data['product_id'].astype(str).str.strip()
        sales_data['date'] = pd.to_datetime(sales_data['date'])
        sales_data = sales_data.sort_values(by=['product_id', 'date'])

        required_cols = ['product_id', 'quantity_sold']
        if not all(col in sales_data.columns for col in required_cols):
            raise ValueError(f"Sales data must contain columns: {required_cols}")

        sales_data = sales_data.set_index(['product_id', 'date'])
        min_date = sales_data.index.get_level_values('date').min()
        max_date = sales_data.index.get_level_values('date').max()
        all_dates = pd.date_range(start=min_date, end=max_date, freq='MS')

        # Use pivot_table to handle aggregation and missing values
        sales_pivot = sales_data.reset_index().pivot_table(values='quantity_sold', index='date', columns='product_id', aggfunc='sum', fill_value=0)

        # Reindex with all_dates - forward fill ensures data is carried over
        sales_pivot = sales_pivot.reindex(all_dates, method='ffill')

        monthly_data = sales_pivot.stack().reset_index()
        monthly_data = monthly_data.rename(columns={'level_0': 'date', 'level_1': 'product_id', 0: 'quantity_sold'})

        print("\n--- monthly_data after cleaning ---")
        print(monthly_data)
        return monthly_data

    except FileNotFoundError:
        print("Sales data file not found.")
        logging.error("Sales data file not found.")
        return pd.DataFrame(columns=['date', 'product_id', 'quantity_sold'])
    except ValueError as ve:
        print(f"Value Error in load_and_prepare_data: {ve}")
        logging.error(f"Value Error in load_and_prepare_data: {ve}")
        return None
    except KeyError as ke:
        print(f"KeyError in load_and_prepare_data: {ke}")
        logging.error(f"KeyError in load_and_prepare_data: {ke}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred in load_and_prepare_data: {e}")
        traceback.print_exc()
        logging.exception(f"Unexpected error in load_and_prepare_data: {e}")
        return None

def prepare_lstm_data(product_data, time_step=1):
    """Prepare data for LSTM model training."""

    print(f"\n---Entering prepare_lstm_data for product {product_data['product_id'].unique()}---")
    if len(product_data) < time_step:
        print(f"Insufficient data points ({len(product_data)}) for product. Returning None.")
        return None, None, None, None

    try:
        scaler = MinMaxScaler(feature_range=(0, 1))
        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        numerical_data = product_data[['quantity_sold']]
        categorical_data = product_data[['product_id']]

        # Check for empty dataframes before proceeding.
        if numerical_data.empty or categorical_data.empty:
            print("Warning: Empty numerical or categorical data. Returning None.")
            return None, None, None, None

        categorical_data['product_id'] = categorical_data['product_id'].astype(str).str.strip()

        scaled_numerical = scaler.fit_transform(numerical_data)
        encoded_categorical = onehot_encoder.fit_transform(categorical_data)

        #Check shapes before proceeding.
        if scaled_numerical.shape[0] != encoded_categorical.shape[0]:
            print(f"Error: Mismatched data shapes: {scaled_numerical.shape}, {encoded_categorical.shape}. Returning None.")
            return None, None, None, None

        scaled_data = np.column_stack((scaled_numerical, encoded_categorical))
        num_rows = len(scaled_data) - time_step

        if num_rows <= 0:
            print(f"Insufficient data points after scaling and encoding (num_rows: {num_rows}). Returning None")
            return None, None, None, None

        X = np.array([scaled_data[i:i + time_step] for i in range(num_rows)])
        y = scaled_data[time_step:time_step + num_rows, 0]
        print(f"X shape: {X.shape}, dtype: {X.dtype}")
        print(f"y shape: {y.shape}, dtype: {y.dtype}")

        if X.shape[0] != y.shape[0]:
            print(f"Error: Length of X and y do not match. X.shape: {X.shape}, y.shape: {y.shape}")
            return None, None, None, None

        return X, y, scaler, onehot_encoder
    except ValueError as ve:
        print(f"Error during OneHotEncoder or scaling: {ve}")
        return None, None, None, None
    except Exception as e:
        print(f"Unexpected error during OneHotEncoder or scaling: {e}")
        return None, None, None, None

def create_lstm_model(input_shape):
    """Create and compile LSTM model."""

    """Create and compile LSTM model."""
    model = Sequential()
    model.add(Input(shape=(input_shape[0], input_shape[1])))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_sales_trend_plot(monthly_data, inventory_data):
    """Create a plot of monthly sales trends by product."""

    plt.figure(figsize=(10, 5))
    products = monthly_data['product_id'].unique()
    all_dates = pd.date_range(start=monthly_data['date'].min(), end=monthly_data['date'].max(), freq='MS')

    for product in products:
        product_str = str(product)
        product_data = monthly_data[monthly_data['product_id'] == product]

        try:
            product_name = inventory_data.loc[product_str, 'product_name']
        except KeyError:
            print(f"Warning: Product ID {product_str} not found in inventory data. Skipping.")
            continue

        aligned_data = product_data.set_index('date')
        aligned_data = aligned_data.reindex(all_dates, method='ffill', fill_value=0) #Forward fill for missing values


        plt.plot(aligned_data.index, aligned_data['quantity_sold'], label=f'Product {product_name}')

    plt.title('Monthly Sales Trends by Product')
    plt.xlabel('Date')
    plt.ylabel('Quantity Sold')
    plt.legend()
    plt.tight_layout()

def generate_report_with_plots(monthly_data, inventory_data, forecasts):
    """Generates a PDF report, prompting user for save location."""
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.asksaveasfilename(
        defaultextension=".pdf",
        filetypes=[("PDF files", "*.pdf")],
        title="Save Forecast Report",
        initialdir="."  # Set initial directory to the current directory
    )

    if file_path:
        with PdfPages(file_path) as pdf:
            # Add title page
            plt.figure(figsize=(8.5, 11))  # Letter paper size
            plt.text(0.5, 0.5, "Inventory and Sales Forecast Report", ha='center', va='center', fontsize=20)
            pdf.savefig()
            plt.close()

            # Add sales trend plot
            create_sales_trend_plot(monthly_data, inventory_data)  # Pass inventory_data
            pdf.savefig()
            plt.close()

            # Add actual vs. predicted plot
            for product, info in forecasts.items():
                product_data = monthly_data[monthly_data['product_id'] == product].set_index('date')
                predictions = info['forecast']
                product_name = info['name']  # Access product name from info dictionary
                plot_forecast_vs_actual(product_data, predictions, product_name)  # Pass product_name
                pdf.savefig()
                plt.close()

            # Add inventory levels plot
            plot_inventory_levels(inventory_data)
            pdf.savefig()
            plt.close()
        return file_path  # Return the file path
    else:
        return None  # Return None if the user cancelled

def plot_forecast_vs_actual(original_data, predictions, product_name):
    """Plot actual vs. predicted sales for a product."""

    plt.figure(figsize=(10, 5))
    plt.plot(original_data.index, original_data['quantity_sold'], label='Actual Sales')
    plt.plot(original_data.index[-len(predictions):], predictions, label='Predicted Sales', color='red')
    plt.title(f'Actual vs. Predicted Sales for {product_name}')
    plt.xlabel('Date')
    plt.ylabel('Quantity Sold')
    plt.legend()
    plt.tight_layout()

def plot_inventory_levels(inventory_data):
    """Plot current inventory levels by product."""

    plt.figure(figsize=(10, 5))
    plt.bar(inventory_data['product_name'], inventory_data['quantity'], label='Inventory Quantity')
    plt.title('Current Inventory Levels')
    plt.xlabel('Product Name')
    plt.ylabel('Quantity')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()

def forecast_demand(monthly_data, inventory_data, min_data_points=12):
    """Forecast demand with improved data handling and detailed error logging."""

    """Forecast demand with improved data handling and detailed error logging."""
    forecasts = {}
    products = inventory_data.index.astype(str)  # Get products from inventory data
    time_step = 6

    for product in products:
        product_data = monthly_data[monthly_data['product_id'] == product].copy()

        if product_data.empty:
            product_name = inventory_data.loc[str(product), 'product_name']
            print(f"Skipping forecast for product {product_name} (ID: {product}): No sales data available.")
            logging.warning(f"Product {product}: No sales data available for forecasting.")
            forecasts[product] = {'name': product_name, 'forecast': []}
            continue

        if 'date' not in product_data.columns or 'quantity_sold' not in product_data.columns:
            product_name = inventory_data.loc[product, 'product_name']
            print(f"Skipping forecast for product {product_name} (ID: {product}): Missing 'date' or 'quantity_sold' column.")
            logging.warning(f"Product {product}: Missing 'date' or 'quantity_sold' column.")
            forecasts[product] = {'name': product_name, 'forecast': ["Missing columns"]}
            continue

        product_data = product_data.set_index('date')

        if product_data.empty:
            product_name = inventory_data.loc[str(product), 'product_name']
            print(f"Skipping forecast for product {product_name} (ID: {product}): Empty DataFrame after setting index.")
            logging.warning(f"Product {product}: Empty DataFrame after setting index.")
            forecasts[product] = {'name': product_name, 'forecast': ["Empty DataFrame"]}
            continue

        product_data = product_data.asfreq('MS').ffill()

        original_length = len(product_data)
        product_data = product_data.dropna(subset=['quantity_sold'])

        if len(product_data) == 0:  # Check if data cleaning resulted in an empty DataFrame
            product_name = inventory_data.loc[str(product), 'product_name']
            print(f"Skipping forecast for product {product_name} (ID: {product}): No data left after cleaning.")
            logging.warning(f"Product {product}: No data left after cleaning.")
            forecasts[product] = {'name': product_name, 'forecast': ["No data after cleaning"]}
            continue

        num_data_points = len(product_data)

        if num_data_points < min_data_points:
            product_name = inventory_data.loc[str(product), 'product_name']
            print(f"Skipping forecast for product {product_name} (ID: {product}): Insufficient data points ({num_data_points} < {min_data_points}).")
            logging.warning(f"Product {product}: Insufficient data for forecasting.")
            forecasts[product] = {'name': product_name, 'forecast': ["Insufficient data"]}
            continue

        print(f"\n--- Raw Data for Product {product} ---")
        print(product_data)

        print(f"\n--- Data types for Product {product} ---")
        print(product_data.dtypes)

        X, y, scaler, onehot_encoder = prepare_lstm_data(product_data, time_step)


        # Handle cases with insufficient data
        if X is None or y is None or X.shape[0] == 0 or y.shape[0] == 0:
            product_name = inventory_data.loc[str(product), 'product_name']
            print(f"Skipping forecast for product {product_name} (ID: {product}): Not enough data for forecasting.")
            logging.warning(f"Product {product}: Insufficient data for forecasting.")
            forecasts[product] = {'name': product_name, 'forecast': []}
            continue

        try:
            model = create_lstm_model((X.shape[1], X.shape[2]))
            model.fit(X, y, epochs=50, batch_size=32, verbose=0)
            predictions = model.predict(X)
            print(f"predictions shape before inverse transform: {predictions.shape}, dtype: {predictions.dtype}")
            predictions = scaler.inverse_transform(predictions)
            print(f"predictions shape after inverse transform: {predictions.shape}, dtype: {predictions.dtype}")
            predictions = predictions.flatten().astype(float)
            print(f"predictions shape after flatten and astype: {predictions.shape}, dtype: {predictions.dtype}")
            rmse = math.sqrt(mean_squared_error(y, predictions))
            print(f"RMSE for Product ID {product}: {rmse}")
            logging.info(f"Forecast complete for product {product}. RMSE: {rmse}")
            forecasts[product] = {'name': inventory_data.loc[str(product)]['product_name'], 'forecast': predictions}

        except Exception as e:
            product_name = inventory_data.loc[str(product), 'product_name']
            print(f"Error forecasting product {product_name} (ID: {product}): {e}")
            logging.error(f"Error forecasting product {product}: {e}")
            forecasts[product] = {'name': product_name, 'forecast': []}

    return forecasts

def check_stationarity(timeseries):
    """Check whether the time series is stationary."""

    """Check whether the time series is stationary."""
    result = adfuller(timeseries)
    return result[1]

def show_loading_indicator():
    """Show a loading indicator during long calculations."""

    loading_label = ttk.Label(forecast_frame, text="Calculating forecast...")
    loading_label.grid(row=3, column=0, sticky="w")
    loading_bar = ttk.Progressbar(forecast_frame, orient="horizontal", mode="indeterminate")
    loading_bar.grid(row=4, column=0, sticky="ew")
    loading_bar.start()
    root.update()
    return loading_label, loading_bar

def hide_loading_indicator(loading_label, loading_bar):
    """Hides the loading indicator."""

    """Hides the loading indicator."""
    loading_label.destroy()
    loading_bar.destroy()

def on_calculate():
    """Handle the calculate button press for sales forecasting."""

    """Handle the calculate button press for sales forecasting."""
    try:
        loading_label, loading_bar = show_loading_indicator()
        results_text.delete("1.0", tk.END)
        monthly_data = load_and_prepare_data()
        inventory_data = load_inventory()

        if monthly_data is None:
            messagebox.showerror("Error", "Could not load sales data. Check your sales_data.csv file.")
            return

        create_sales_trend_plot(monthly_data, inventory_data)  # Pass inventory_data

        for widget in frame.winfo_children():
            widget.destroy()

        # Forecast the demand
        forecasts = forecast_demand(monthly_data, inventory_data, min_data_points=12)

        results_summary = "Sales Forecast for the Next 12 Months:\n"
        month_names = ["January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"]

        for product, info in forecasts.items():
            if isinstance(info['forecast'], list) and len(info['forecast']) == 0:
                results_summary += f"\nForecast for {info['name']} (Product ID {product}):\nNo Forecast Data Available.\n"
                continue  # Skip products with no forecast data
            product_data = monthly_data[monthly_data['product_id'] == product].set_index('date')
            predictions = info['forecast']
            product_name = info['name']

            plot_forecast_vs_actual(product_data, predictions, product_name)

            results_summary += f"\nForecast for {info['name']} (Product ID {product}):\n"
            forecast_len = len(predictions)

            # Check if the forecasts are actually numbers before attempting operations
            if all(isinstance(pred, (int, float)) for pred in predictions):
                for month in range(12):
                    if month < forecast_len and not np.isnan(predictions[month]):
                        results_summary += f"{month_names[month]}: {predictions[month]:.2f}\n"
                    else:
                        results_summary += f"{month_names[month]}: Not available\n"
            else:
                results_summary += "Forecast data is not numeric. Cannot generate report.\n"

        results_text.insert(tk.END, results_summary)

        # Generate PDF report
        try:
            report_filename = generate_report_with_plots(monthly_data, inventory_data, forecasts)
            if report_filename:
                messagebox.showinfo("Report Generated", f"Inventory and sales forecast report generated successfully in {report_filename}")
            else:
                messagebox.showinfo("Report Cancelled", "Report generation cancelled by user.")

        except Exception as pdf_error:
            messagebox.showerror("Error Generating Report", f"An error occurred while generating the PDF report: {pdf_error}")

        hide_loading_indicator(loading_label, loading_bar)

    except ValueError as ve:
        hide_loading_indicator(loading_label, loading_bar)
        messagebox.showerror("ValueError", f"Value error occurred: {ve}")
    except Exception as e:
        hide_loading_indicator(loading_label, loading_bar)
        error_message = f"An error occurred: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        messagebox.showerror("Error", error_message)

def add_item_to_inventory():
    """Add or update an item in the inventory."""

    product_id = product_id_entry.get().strip()
    product_name = product_name_entry.get().strip()
    quantity_str = quantity_entry.get().strip()

    if not product_id.isalnum():
        messagebox.showwarning("Input Error", "Product ID must be alphanumeric.")
        return
    if not quantity_str.isdigit() or int(quantity_str) <= 0:
        messagebox.showwarning("Input Error", "Please enter a valid positive integer quantity.")
        return

    quantity = int(quantity_str)
    inventory_data = load_inventory()

    if product_id in inventory_data.index.astype(str):
        inventory_data.loc[product_id, 'quantity'] += quantity
        log_event("Update", product_id, product_name, quantity)
    else:
        new_item = pd.DataFrame({'product_id': [product_id], 'product_name': [product_name], 'quantity': [quantity]})
        inventory_data = pd.concat([inventory_data, new_item.set_index('product_id')], ignore_index=False)
        log_event("Add", product_id, product_name, quantity)

    inventory_data.to_csv(inventory_file, index=True)
    messagebox.showinfo("Success", f"Added/Updated {product_name} (ID: {product_id}) to inventory.")

def input_sale_or_recall():
    """Input a sale or recall for a specific item."""

    """Input a sale or recall for a specific item."""
    product_id = sale_product_id_entry.get().strip()
    quantity_str = sale_quantity_entry.get().strip()

    if not product_id.isalnum() or not quantity_str.isdigit() or int(quantity_str) <= 0:
        messagebox.showwarning("Input Error", "Invalid input. Product ID must be alphanumeric and quantity must be a positive integer")
        return

    quantity = int(quantity_str)
    inventory_data = load_inventory()

    if product_id not in inventory_data.index.astype(str):
        messagebox.showwarning("Warning", f"Product ID {product_id} not found in inventory.")
        return

    cur_quantity = inventory_data.loc[product_id, 'quantity']
    if cur_quantity >= quantity:
        inventory_data.loc[product_id, 'quantity'] -= quantity
        save_inventory(inventory_data)
        log_event("Sale", product_id, inventory_data.loc[product_id, 'product_name'], quantity)

        if pd.io.common.file_exists(sales_log_file):
            sales_data = pd.read_csv(sales_log_file, parse_dates=['date'])
            sales_data['date'] = sales_data['date'].dt.strftime('%Y-%m-%d')
        else:
            sales_data = pd.DataFrame(columns=['date', 'product_id', 'quantity_sold'])

        sale_date = pd.Timestamp.now().strftime('%Y-%m-%d')

        new_sale = pd.DataFrame({'date': [sale_date], 'product_id': [product_id], 'quantity_sold': [quantity]})

        sales_data = pd.concat([sales_data, new_sale], ignore_index=True)
        sales_data.to_csv(sales_log_file, index=False)

        messagebox.showinfo("Success", f"Processed sale of {quantity} for Product ID {product_id}.")
    else:
        messagebox.showerror("Error", "Not enough inventory to process the sale.")

def generate_report():
    """Generate a report of items returned vs deployed."""

    """Generate a report of items returned vs deployed."""
    inventory_data = load_inventory()

    if inventory_data.empty:
        report_text.delete("1.0", tk.END)
        report_text.insert(tk.END, "No inventory data available.")
        return

    report_summary = "Inventory Report:\n"
    for index, row in inventory_data.iterrows():
        report_summary += f"Product ID: {index}, Name: {row['product_name']}, Available Quantity: {row['quantity']}\n"

    report_text.delete("1.0", tk.END)
    report_text.insert(tk.END, report_summary)

root = tk.Tk()
root.title("Inventory Management Dashboard")

# Configure grid weights for responsiveness (important for layout)
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# Top-left quadrant: Inventory input
inventory_frame = ttk.Frame(root)
inventory_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

ttk.Label(inventory_frame, text="Product ID:").grid(row=0, column=0, sticky="w")
product_id_entry = ttk.Entry(inventory_frame)
product_id_entry.grid(row=0, column=1, sticky="ew")

ttk.Label(inventory_frame, text="Product Name:").grid(row=1, column=0, sticky="w")
product_name_entry = ttk.Entry(inventory_frame)
product_name_entry.grid(row=1, column=1, sticky="ew")

ttk.Label(inventory_frame, text="Quantity to Add:").grid(row=2, column=0, sticky="w")
quantity_entry = ttk.Entry(inventory_frame)
quantity_entry.grid(row=2, column=1, sticky="ew")

add_inventory_button = ttk.Button(inventory_frame, text="Add/Update Quantity", command=add_item_to_inventory)
add_inventory_button.grid(row=3, column=0, columnspan=2, pady=5)

warning_label = ttk.Label(inventory_frame, text="Warning: Entered product names will be overwritten by existing product names for each Product ID adjusted", foreground="red")
warning_label.grid(row=4, column=0, columnspan=2)

# Top-right quadrant: Sales input
sale_frame = ttk.Frame(root)
sale_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

ttk.Label(sale_frame, text="Sale Product ID:").grid(row=0, column=0, sticky="w")
sale_product_id_entry = ttk.Entry(sale_frame)
sale_product_id_entry.grid(row=0, column=1, sticky="ew")

ttk.Label(sale_frame, text="Sale Quantity:").grid(row=1, column=0, sticky="w")
sale_quantity_entry = ttk.Entry(sale_frame)
sale_quantity_entry.grid(row=1, column=1, sticky="ew")

sale_button = ttk.Button(sale_frame, text="Process Sale", command=input_sale_or_recall)
sale_button.grid(row=2, column=0, columnspan=2, pady=5)

# Bottom-left quadrant: Report generation
report_frame = ttk.Frame(root)
report_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

report_button = ttk.Button(report_frame, text="Generate Report", command=generate_report)
report_button.grid(row=0, column=0, pady=5)

ttk.Label(report_frame, text="Inventory Report:").grid(row=1, column=0, sticky="w")
report_text = tk.Text(report_frame, width=40, height=15, wrap=tk.WORD)
report_text.grid(row=2, column=0, sticky="nsew")

# Bottom-right quadrant: Forecast
forecast_frame = ttk.Frame(root)
forecast_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

calculate_forecast_button = ttk.Button(forecast_frame, text="Calculate Forecast", command=on_calculate)
calculate_forecast_button.grid(row=0, column=0, pady=5)

ttk.Label(forecast_frame, text="Sales Forecast:").grid(row=1, column=0, sticky="w")
results_text = tk.Text(forecast_frame, width=40, height=15, wrap=tk.WORD)
results_text.grid(row=2, column=0, sticky="nsew")

# Frame for plots
frame = ttk.Frame(root)
frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")


root.mainloop()