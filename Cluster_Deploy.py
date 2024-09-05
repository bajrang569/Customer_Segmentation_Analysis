# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 23:09:51 2024

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

# Load the data
pca_data = pd.read_csv("pca_data.csv")

# Train the KMeans model
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

# Create a Streamlit app
st.title("K-Means Clustering App")

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