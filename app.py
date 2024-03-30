import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kmeans import KmeansClustering
from distances import euclidean_distance, manhattan_distance, cosine_distance
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


st.title('Data Analysis App')

# Let the user choose the method of data input - either by uploading a file or manual input
data_input_method = st.radio("Choose how to input data:", ('Upload a file', 'Manual input'))

# Initialize an empty DataFrame in the session state if it doesn't exist to store the data
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame()

# File upload option
if data_input_method == 'Upload a file':
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'txt'])
    if uploaded_file is not None:
        st.session_state.data = pd.read_csv(uploaded_file)

# Manual input option
elif data_input_method == 'Manual input':
    st.write("### Step 1: Define Columns")
    col1, col2 = st.columns(2)
    
    with col1:
        num_columns = st.number_input("Number of columns", min_value=1, value=2, step=1)
        column_names = []
        for i in range(num_columns):
            column_name = st.text_input(f"Column {i+1} name:", key=f'col_name_{i}')
            if column_name:  # Ensure column name is not empty
                column_names.append(column_name)

    with col2:
        st.write("### Step 2: Generate Random Data")
        st.write("Specify the number of rows and value range for random data generation.")
        num_rows = st.number_input("Number of rows to generate", min_value=1, value=5, step=1)
        min_val = st.number_input("Minimum value", value=0)
        max_val = st.number_input("Maximum value", value=100)
        
        if st.button("Generate Data"):
            random_data = np.random.randint(min_val, max_val+1, size=(num_rows, len(column_names)))
            temp_df = pd.DataFrame(random_data, columns=column_names)
            st.session_state.data = temp_df  # Update the session state DataFrame

    # Allow user to directly edit the DataFrame if columns have been defined
    if column_names:
        st.write("### Step 3: Edit Data (Optional)")
        st.session_state.data = st.data_editor(st.session_state.data, num_rows="dynamic")

# Display the data to be analyzed
if not st.session_state.data.empty:
    st.write("### Data Preview:")
    st.write(st.session_state.data)
    
    # Let the user choose the clustering method: K-Means or Hierarchical
    method = st.radio("Choose a clustering method:", ('K-Means Clustering', 'Hierarchical Clustering'))

    if method == 'K-Means Clustering':
        # User defines the number of clusters for K-Means
        n_clusters = st.number_input("Number of clusters", min_value=2, value=3, step=1, key='kmeans_clusters')
        
        # User selects the distance metric
        distance_metric = st.selectbox("Select distance metric:", ('Euclidean', 'Manhattan', 'Cosine'), key='distance_metric')
        
        # Map user selection to actual distance function
        distance_functions = {
            'Euclidean': euclidean_distance,
            'Manhattan': manhattan_distance,
            'Cosine': cosine_distance
        }
        selected_distance_function = distance_functions[distance_metric]
        
        if st.button('Run K-Means', key='run_kmeans'):
            # Convert all columns of the DataFrame to numeric, coercing errors to NaN
            numeric_data = st.session_state.data.apply(pd.to_numeric, errors='coerce').dropna()

            # Ensure the DataFrame has enough numeric columns
            if numeric_data.empty or numeric_data.shape[1] < 2:
                st.error("The dataset must contain at least two numeric columns for clustering.")
            else:
                # Proceed with clustering using numeric_data
                kmeans = KmeansClustering(k=n_clusters, distance=selected_distance_function)
                labels = kmeans.fit(numeric_data.values)  # Use values for clustering
                st.session_state.data.loc[numeric_data.index, 'Cluster'] = labels  # Assign labels to the original data
                
                # Visualization
                fig = kmeans.plot_clusters(numeric_data.values, labels)
                st.pyplot(fig)

            
    elif method == 'Hierarchical Clustering':
        # User defines the number of clusters for cutting the dendrogram
        n_clusters_hier = st.number_input("Number of clusters (for dendrogram cut)", min_value=2, value=3, step=1, key='hier_clusters')
        if st.button('Run Hierarchical Clustering', key='run_hierarchical'):
            # Perform Hierarchical Clustering and display dendrogram
            linked = linkage(st.session_state.data, method='ward')
            plt.figure(figsize=(10, 7))
            dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
            plt.title('Hierarchical Clustering Dendrogram')
            plt.xlabel('Sample Index')
            plt.ylabel('Distance')
            st.pyplot(plt)
            plt.close()  # Clear the plot to avoid conflicts in subsequent plots

            # If a specific number of clusters was chosen, use AgglomerativeClustering to assign cluster labels
            if n_clusters_hier:
                model = AgglomerativeClustering(n_clusters=n_clusters_hier, linkage='ward')
                labels = model.fit_predict(st.session_state.data)
                st.session_state.data['Cluster'] = labels

                # Plot the clustered data points
                fig, ax = plt.subplots()
                scatter = ax.scatter(st.session_state.data.iloc[:, 0], st.session_state.data.iloc[:, 1], c=labels, cmap='viridis', label='Data Points')
                plt.title('Hierarchical Clustering Results')
                plt.xlabel('Feature 1')
                plt.ylabel('Feature 2')
                ax.legend()
                st.pyplot(fig)
