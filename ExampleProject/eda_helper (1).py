
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

car_df = pd.read_csv('https://raw.githubusercontent.com/jamesh9595/excel/main/Project1_car_data.csv') #load csv

data_field_descriptions = {
    'Company Name': {
        'Data Type': 'object',
        'Description': 'Company name for each car brand',
        'Valid Values/Ranges': 'Non-negative objects',
        'Missing Values': 'none',
        'Source': 'Car_data.csv',
        'Additional Notes': 'There are 31 company names'
    },

    'Model Name': {
        'Data Type': 'object',
        'Description': 'Model Name of each car',
        'Valid Values/Ranges': 'Non-negative objects',
        'Missing Values': 'None',
        'Source': 'Car_data.csv',
        'Additional Notes': 'Multiple model names belong to one company'
    },

    'Price in PKR': {
        'Data Type': 'Integer',
        'Description': 'Price of each car in the dataset',
        'Valid Values/Ranges': 'Non-negative integers',
        'Missing Values': 'None',
        'Source': 'Car_data.csv',
        'Additional Notes': 'The price in this column has the unit of Pakistan Rupees'
    },

    'Model Year': {
        'Data Type': 'Integer',
        'Description': 'Model year for each car',
        'Valid Values/Ranges': 'Non-negative integers',
        'Missing Values': 'none',
        'Source': 'Car_data.csv',
        'Additional Notes': 'There are 196 model names'
    },

    'Location': {
        'Data Type': 'object',
        'Description': 'The location of each car in India',
        'Valid Values/Ranges': 'Non-negative objects',
        'Missing Values': 'None',
        'Source': 'Car_data.csv',
        'Additional Notes': 'Multiple cars can be located in the same location or spread out in different locations'
    },

    'Mileage': {
        'Data Type': 'Integer',
        'Description': 'The mileage for each car in the dataset',
        'Valid Values/Ranges': 'Non-negative integers',
        'Missing Values': 'none',
        'Source': 'Car_data.csv',
        'Additional Notes': 'The mileage ranges from 1 to 999999 miles'
    },

    'Engine Type': {
        'Data Type': 'object',
        'Description': 'Engine type for each car',
        'Valid Values/Ranges': 'Non-negative objects',
        'Missing Values': 'None',
        'Source': 'Car_data.csv',
        'Additional Notes': 'There are a total of 3 engine types: Patrol, Diesel, Hybrid'
    },

    'Engine Capacity': {
        'Data Type': 'Integer',
        'Description': 'The total volume of the cylinders in the engine',
        'Valid Values/Ranges': 'Non-negative integers',
        'Missing Values': 'None',
        'Source': 'Car_data.csv',
        'Additional Notes': 'There are 75 types of engine capacity'
    },

    'Color': {
        'Data Type': 'Object',
        'Description': 'The color of each car',
        'Valid Values/Ranges': 'Non-negative objects',
        'Missing Values': 'None',
        'Source': 'Car_data.csv',
        'Additional Notes': 'There are 24 different colors in this dataset'
    },

    'Assembly': {
        'Data Type': 'object',
        'Description': 'Where was the car assembled',
        'Valid Values/Ranges': 'Non-negative objects',
        'Missing Values': 'none',
        'Source': 'Car_data.csv',
        'Additional Notes': 'The car can be assembled locally or imported'
    },

    'Body Type': {
        'Data Type': 'object',
        'Description': 'Body type of the car',
        'Valid Values/Ranges': 'Non-negative objects',
        'Missing Values': 'None',
        'Source': 'Car_data.csv',
        'Additional Notes': 'There are 6 different body types: Hatchback, Sedan, SUV, Crossover, Van, Mini Van'
    },

    'Transmission Type': {
        'Data Type': 'Object',
        'Description': 'The transmission type for each car',
        'Valid Values/Ranges': 'Non-negative objects',
        'Missing Values': 'None',
        'Source': 'Car_data.csv',
        'Additional Notes': 'There are two types of transmissions: Automatic and Manual'
    },

    'Registration Status': {
        'Data Type': 'object',
        'Description': 'The status of the car',
        'Valid Values/Ranges': 'Non-negative objects',
        'Missing Values': 'None',
        'Source': 'Car_data.csv',
        'Additional Notes': 'The status of the car can be Un-Registered or Registered'
    },

    'Price USD': {
        'Data Type': 'integer',
        'Description': 'Price of the car converted to U.S. Dollars',
        'Valid Values/Ranges': 'Non-negative integers',
        'Missing Values': 'None',
        'Source': 'Car_data.csv',
    }}

def list_data_descriptors(car_df, data_field_descriptions):
    for column in car_df.columns:
        if column in data_field_descriptions:
            print(f"Column: {column}")
            description = data_field_descriptions[column]
            for key, value in description.items():
                print(f"{key}: {value}")
            print()
        else:
            print(f"No description available for column: {column}\n")
            
cat_cols=car_df.select_dtypes(include=['object']).columns
num_cols = car_df.select_dtypes(include=np.number).columns.tolist()

def plot_numeric_histograms():
    for i, col in enumerate(num_cols):
        plt.figure(figsize=(14, 7))  # Create a new figure for each column
        sns.histplot(car_df, x=col, color='#5573ad', kde=True)  # Histogram with KDE
        plt.xlabel(col, fontweight='bold',fontsize=12)  # Set the x-axis label`1
        plt.ylabel('Frequency', fontweight='bold',fontsize=12)
        plt.tick_params(axis='both', labelsize=11) #set tick size
        plt.title(f'Distribution of {col}', size=15)  # Title for each histogram
        plt.savefig(f"Distribution_{col}.png", dpi=150, bbox_inches="tight")  # Save each plot
        plt.show()  # Display the plot

def plot_category_barplots():
    for col in cat_cols:
        # Select the top 15 most frequent categories in the column
        top_categories = car_df[col].value_counts().head(15).index

        # Filter the DataFrame to include only the top categories
        filtered_df = car_df[car_df[col].isin(top_categories)]

        # Create the bar plot
        plt.figure(figsize=(18, 7)) 
        sns.countplot(x=col, data=filtered_df, order=top_categories, color='#5573ad')
        plt.xlabel(col, fontweight='bold', fontsize=14)
        plt.ylabel('Frequency', fontweight='bold', fontsize=14)
        plt.tick_params(axis='both', labelsize=12,)
        plt.xticks(rotation=0)  # Set tick size
        plt.title(f'Top 15 in {col}', size=18)  
        plt.savefig(f"Top15_{col}.png", dpi=150, bbox_inches="tight")  # Save each plot
        plt.show()  # Display 

def plot_price_per_company_scatter():
    plt.figure(figsize=(18, 7))  
    sns.scatterplot(x='Company Name', y='Price USD', data=car_df, palette='muted')  # ci='sd' shows the std as error bar
    plt.xticks(rotation=45)  
    plt.tick_params(labelsize=12)
    plt.xlabel('Brand', fontweight='bold', fontsize=14)
    plt.ylabel('Price USD', fontweight='bold',fontsize=14)
    plt.title('Price per Brand',fontsize=18)
    plt.savefig("Price Per Brand.png", dpi=150, bbox_inches="tight")
    plt.show()

def plot_price_per_model_bar():
    plt.figure(figsize=(18, 7))  
    sns.barplot(x='Model Name', y='Price USD', data=car_df[:40], ci='sd', palette='muted')  # ci='sd' will show the standard deviation as the error bar
    plt.xticks(rotation=45)  
    plt.tick_params(labelsize=12)
    plt.xlabel('Model Name', fontweight='bold',fontsize=14)
    plt.ylabel('Average Price', fontweight='bold',fontsize=14)
    plt.title('Average Price per Model for the Top 40 Models with Standard Deviation',fontsize=18)
    plt.savefig("Price Per Model.png", dpi=150, bbox_inches="tight")
    plt.show()
def plot_price_per_mile_scatter():  
    plt.figure(figsize=(18, 7))  # Adjust the figure size as needed
    sns.scatterplot(x='Mileage', y='Price USD', data=car_df, palette='muted')
    plt.xticks(rotation=45)  
    plt.tick_params(labelsize=12)
    plt.xlabel('Mileage (Hundred Thousand Miles)', fontweight='bold', fontsize=14)
    plt.ylabel('Price USD', fontweight='bold', fontsize=14)
    plt.title('Price per Mileage', fontweight='bold', fontsize=18)

    # Create a custom tick locator for x-axis
    x_locator = ticker.MultipleLocator(base=100000)  # Set the base to 100,000 for hundred thousand increments

    # Apply the custom tick locator and format to the x-axis
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_locator)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1000):,}K'))
    
    plt.savefig("Price Per Mile.png", dpi=150, bbox_inches="tight")
    plt.show() #plot.
    
def plot_price_per_modelyear_bar():   
    plt.figure(figsize=(18, 7))  # Adjust the figure size as needed
    sns.barplot(x='Model Year', y='Price USD', data=car_df, palette='muted')  # ci='sd' will show the standard deviation as the error bar
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tick_params(labelsize=12)
    plt.xlabel('Model Year', fontweight='bold', fontsize=14)
    plt.ylabel('Price USD', fontweight='bold',fontsize=14)
    plt.title('Price per Model Year',fontsize=18)
    plt.savefig("Price Per Model Year.png", dpi=150, bbox_inches="tight")
    plt.show()
