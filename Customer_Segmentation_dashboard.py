# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:19:39 2024

@author: Administrator
"""

import numpy as np
import pandas as pd
import os
os.environ['THREADPOOLCTL_ENABLED'] = '0'
from sklearn.cluster import KMeans
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

# Load the customer segmentation data for clustering
pca_data = pd.read_csv("pca_data.csv")

# Load the marketing campaign data for dashboard
df = pd.read_excel('marketing_campaign1 (1).xlsx')

# Feature engineering for marketing data
df['Age'] = 2024 - df['Year_Birth']

# Train the KMeans model on PCA data
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(pca_data[['PC1', 'PC2', 'PC3', 'PC4']])

# Save the KMeans model
with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

# Define the prediction function
def predict_clusters(X_new):
    # Load the saved KMeans model
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    
    # Assign new data points to clusters
    cluster_labels = kmeans.predict(X_new)
    
    return cluster_labels

# Page title
st.title("Marketing Campaign and K-Means Clustering Dashboard")

# Sidebar for navigation
st.sidebar.header("Navigation")
pages = st.sidebar.selectbox("Choose a section", ["Overview", "Demographics", "Spending Patterns", "Campaign Analysis", "Customer Segmentation", "K-Means Clustering"])

# Overview Section
if pages == "Overview":
    st.header("Dataset Overview")
    st.write(df.head())
    st.write("### Basic Information")
    st.write(df.describe())
    st.write(f"Total records: {df.shape[0]}, Total features: {df.shape[1]}")

# Demographics Section
if pages == "Demographics":
    st.header("Customer Demographics Analysis")
    
    # Age Distribution
    st.subheader("Age Distribution")
    age_hist = px.histogram(df, x='Age', nbins=20, title="Age Distribution")
    st.plotly_chart(age_hist)

    # Education Distribution
    st.subheader("Education Distribution")
    edu_bar = px.bar(df['Education'].value_counts(), title="Education Levels")
    st.plotly_chart(edu_bar)

    # Marital Status Distribution
    st.subheader("Marital Status Distribution")
    marital_bar = px.pie(df, names='Marital_Status', title="Marital Status Breakdown")
    st.plotly_chart(marital_bar)

# Spending Patterns Section
if pages == "Spending Patterns":
    st.header("Customer Spending Patterns")
    
    # Spending on different products
    products = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    
    st.subheader("Spending on Product Categories")
    spending = df[products].sum()
    spending_bar = px.bar(spending, labels={'index':'Product', 'value':'Amount Spent'}, title="Total Spending on Products")
    st.plotly_chart(spending_bar)
    
    st.subheader("Spending by Age Group")
    age_spending = df.groupby('Age')[products].mean().reset_index()
    age_spending_line = px.line(age_spending, x='Age', y=products, title="Average Spending by Age Group")
    st.plotly_chart(age_spending_line)

# Campaign Analysis Section
if pages == "Campaign Analysis":
    st.header("Marketing Campaign Analysis")
    
    # Campaign acceptance
    st.subheader("Campaign Acceptance Rates")
    campaigns = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
    campaign_sum = df[campaigns].sum()
    campaign_bar = px.bar(campaign_sum, labels={'index':'Campaign', 'value':'Accepted Count'}, title="Accepted Campaigns")
    st.plotly_chart(campaign_bar)

    # Response to the last campaign
    st.subheader("Response to the Last Campaign")
    response_pie = px.pie(df, names='Response', title="Response to the Last Campaign (1: Yes, 0: No)")
    st.plotly_chart(response_pie)

# Customer Segmentation Section
if pages == "Customer Segmentation":
    st.header("Customer Segmentation with K-Means")

    # Select features for clustering
    st.subheader("Choose features for clustering")
    selected_features = st.multiselect("Select features", options=df.columns, default=['Age', 'Income', 'MntWines', 'NumWebPurchases'])
    
    if selected_features:
        st.write("### Selected Features")
        st.write(df[selected_features].head())

        # Drop rows with missing values in selected features
        df_cluster = df[selected_features].dropna()

        # Perform clustering
        n_clusters = st.slider("Select number of clusters", 2, 10, 4)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df_cluster['Cluster'] = kmeans.fit_predict(df_cluster)

        # Merge the clusters back into the original dataframe
        df['Cluster'] = pd.NA
        df.loc[df_cluster.index, 'Cluster'] = df_cluster['Cluster']

        # Cluster visualization
        st.subheader(f"Clusters Visualization ({n_clusters} clusters)")
        cluster_scatter = px.scatter(df_cluster, x=selected_features[0], y=selected_features[1], color='Cluster', title="Cluster Scatter Plot")
        st.plotly_chart(cluster_scatter)

        st.write(f"Cluster Centers: {kmeans.cluster_centers_}")

# K-Means Clustering Section (3D Clustering App)
if pages == "K-Means Clustering":
    st.header("K-Means Clustering App")

    # Load the model when the button is clicked
    load_model = st.button("Load Model")

    if load_model:
        # Plot the clusters in 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(pca_data['PC1'], pca_data['PC2'], pca_data['PC3'], c=kmeans.labels_, cmap='tab20', s=50)
        ax.set_title("K-Means Clustering Results")
        
        # Create a legend for the clusters
        legend_labels = [f"Cluster {i}" for i in range(4)]
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=legend_labels[i], markerfacecolor=plt.cm.tab20(i / 4)) for i in range(4)]
        ax.legend(handles=legend_handles)
        
        st.pyplot(fig)

    # Text input for new data
    new_data_input = st.text_input("Enter new data (PC1, PC2, PC3, PC4):")
    predict_cluster = st.button("Predict Cluster")

    if predict_cluster:
        if new_data_input:  
            if new_data_input.strip() != "":  
                try:
                    new_data = [float(x) for x in new_data_input.split(",")]
                    new_data = pd.DataFrame([new_data], columns=['PC1', 'PC2', 'PC3', 'PC4'])
                    cluster_label = predict_clusters(new_data)[0]
                    st.write(f"Cluster label: {cluster_label}")
                except ValueError:
                    st.write("Please enter valid numeric data separated by commas.")
            else:
                st.write("Please enter new data first!")
        else:
            st.write("Please enter new data first!")
