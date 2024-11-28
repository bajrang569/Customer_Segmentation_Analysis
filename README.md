# Customer_Segmentation_Analysis
Marketing Campaign Data Analysis 🎯

📌 Project Overview
This project analyzes a comprehensive dataset of marketing campaigns to uncover patterns in customer behavior, optimize targeting strategies, and improve campaign success rates. It includes data cleaning, visualization, segmentation, and predictive modeling steps.

🗂 Dataset Overview
The dataset contains 2240 entries with 29 attributes representing customer demographics, purchasing behavior, campaign responses, and more.

**Key Columns:
ID: Unique customer identifier
Year_Birth: Year of birth of the customer
Education: Education level (e.g., Graduation, PhD)
Income: Annual income in monetary units
Kidhome & Teenhome: Number of children and teenagers at home
Recency: Number of days since the last purchase
MntWines, MntMeatProducts, etc.: Amount spent on various product categories
NumWebPurchases, NumStorePurchases: Purchase channel data
Response: Indicator of campaign response (1 = Yes, 0 = No)**

🎯 Objectives
Clean and preprocess data to handle missing values and inconsistencies.
Perform Exploratory Data Analysis (EDA) to identify key trends and patterns.
Use customer segmentation techniques to group customers based on behavior and demographics.
Build predictive models to determine the likelihood of campaign responses.
Provide actionable insights to optimize marketing efforts.

🛠️ Tech Stack
Python Libraries:

Pandas, NumPy for data manipulation
Matplotlib, Seaborn for visualizations
Scikit-learn for machine learning models
Tools:

Jupyter Notebook for analysis
Streamlit for potential deployment

🧹 Data Cleaning Steps
Removed duplicate entries and filled missing values in the Income column.
Standardized columns to remove inconsistencies in data types.
Outlier detection performed on spending-related columns.

🔍 Exploratory Data Analysis (EDA) Highlights
Customer Segments: Found distinct groups based on age, spending habits, and campaign responses.
Top Spending Categories: Wines and Meat Products dominate spending.
Campaign Analysis: Acceptance rates varied significantly by customer demographics.

🧠 Machine Learning Approach
Clustering (Segmentation): Used K-means to group customers into meaningful segments.
Predictive Modeling: Applied classification models to predict campaign response rates

📈 Results and Insights
High-Value Customers: Identified based on spending and engagement metrics.
Campaign Optimization: Suggested targeting strategies for underperforming campaigns.
Behavioral Patterns: Customers with children had distinct spending trends.
