import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
import requests
from pathlib import Path

# Function to load the dataset from a URL or local file with customizable settings
def load_data(file_path_or_url, sep, header, encoding, is_url=False):
    try:
        if is_url:
            data = requests.get(file_path_or_url).content
            df = pd.read_csv(StringIO(data.decode(encoding)), sep=sep, header=header)
        else:
            df = pd.read_csv(file_path_or_url, sep=sep, header=header, encoding=encoding)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None
    return df

# Analyze categorical columns
def analyze_categorical_columns(df):
    categorical_df = df.select_dtypes(include=['object', 'category'])
    if not categorical_df.empty:
        stats_dict = {
            'Unique Values': categorical_df.nunique(),
            'Most Common': categorical_df.apply(lambda x: x.value_counts().index[0] if not x.value_counts().empty else None),
            'Most Common Count': categorical_df.apply(lambda x: x.value_counts().iloc[0] if not x.value_counts().empty else 0),
            'Least Common': categorical_df.apply(lambda x: x.value_counts().index[-1] if not x.value_counts().empty else None),
            'Least Common Count': categorical_df.apply(lambda x: x.value_counts().iloc[-1] if not x.value_counts().empty else 0)
        }
        return pd.DataFrame(stats_dict)
    return None

# Analyze numeric columns
def analyze_numeric_columns(df):
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        stats_df = pd.DataFrame({
            'Mean': numeric_df.mean(),
            'Median': numeric_df.median(),
            'Std': numeric_df.std(),
            'Min': numeric_df.min(),
            'Max': numeric_df.max(),
            'Skewness': numeric_df.skew(),
            'Kurtosis': numeric_df.kurtosis()
        })
        return stats_df
    return None

# Plot correlation matrix
def plot_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        corr = numeric_df.corr()
        fig = px.imshow(corr,
                        labels=dict(color="Correlation"),
                        x=corr.columns,
                        y=corr.columns,
                        aspect="auto",
                        color_continuous_scale="RdBu",
                        text_auto=True)  # Add correlation numbers
        return fig
    return None

def main():
    st.set_page_config(page_title="ML Dataset Explorer", layout="wide")

    st.title("EDA Essentials")
    st.write("Quickly explore your datasets by loading them from your local machine or from a URL.")

    # Dataset source selection
    dataset_source = st.radio(
        "Select how you'd like to load the dataset:",
        ("Local file", "From URL (e.g., GitHub)")
    )

    sep = st.text_input("Separator", value=',')  # Separator input with default value
    header = st.selectbox("Header", [None, 'infer'], index=1)  # Choose whether the dataset has a header
    encoding = st.selectbox("Encoding", ['utf-8', 'latin1', 'iso-8859-1', 'cp1252'], index=0)  # Encoding selection

    if dataset_source == "Local file":
        uploaded_file = st.file_uploader("Choose a file (must be a CSV file):")
        if uploaded_file is not None:
            df = load_data(uploaded_file, sep, header, encoding, is_url=False)
        else:
            df = None
    else:
        dataset_url = st.text_input("Enter the dataset URL (must be a CSV file):")
        if dataset_url:
            df = load_data(dataset_url, sep, header, encoding, is_url=True)
        else:
            df = None

    # Proceed if dataset is loaded
    if df is not None:
        st.subheader("Dataset Overview")
        st.write(f"**Shape of dataset:** {df.shape[0]} rows, {df.shape[1]} columns")
        st.dataframe(df.head())

        # Numeric column analysis
        st.subheader("Numeric Column Analysis")
        numeric_stats = analyze_numeric_columns(df)
        if numeric_stats is not None:
            st.dataframe(numeric_stats)
        else:
            st.write("No numeric columns found in the dataset.")

        # Categorical column analysis
        st.subheader("Categorical Column Analysis")
        categorical_stats = analyze_categorical_columns(df)
        if categorical_stats is not None:
            st.dataframe(categorical_stats)
        else:
            st.write("No categorical columns found in the dataset.")

        # Correlation Matrix
        st.subheader("Correlation Matrix")
        corr_matrix = plot_correlation_matrix(df)
        if corr_matrix:
            st.plotly_chart(corr_matrix)
        else:
            st.write("No numeric columns found for correlation matrix.")

        # Download button for cleaned data
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Cleaned Data as CSV",
            data=csv,
            file_name='cleaned_data.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    main()