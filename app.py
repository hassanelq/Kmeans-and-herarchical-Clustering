import streamlit as st
import pandas as pd
import numpy as np

from kmeans_Clustering import KmeansClustering
from hierarchical_Clustering import hierarchicalClustering


def run_kmeans_clustering(data):
    st.sidebar.title("Parameters")
    K = st.sidebar.slider('Number of clusters', value=3, min_value=2, max_value=50, step=1, key='kmeans_K')
    distance = st.sidebar.selectbox('Distance metric', ('Euclidean', 'Manhattan', 'Cosine'), key='kmeans_distance')
    
    if data.empty or data.shape[1] < 2:
        st.error("The dataset must contain at least two numeric columns for clustering.")
        return
    
    kmeans = KmeansClustering(k=K, distance=distance)
    labels = kmeans.fit(data.values)
    fig = kmeans.plot_clusters(data.values, labels)
    st.pyplot(fig)

def run_hierarchical_clustering(data):
    st.sidebar.title("Parameters")
    K = st.sidebar.slider('Number of clusters', value=3, min_value=2, max_value=50, step=1, key='hier_K')
    
    if data.empty or data.shape[1] < 2:
        st.error("The dataset must contain at least two numeric columns for clustering.")
        return
    
    hierarchical_clustering = hierarchicalClustering(n_clusters=K, linkage='ward')
    labels = hierarchical_clustering.fit_predict(data.values)
    
    # Plot dendrogram
    fig_dendrogram = hierarchical_clustering.plot_dendrogram()
    st.pyplot(fig_dendrogram)
    
    # Plot clustered data points
    fig_clusters = hierarchical_clustering.plot_clusters(data.values, labels)
    st.pyplot(fig_clusters)


st.title('kmeans clustering and hierarchical clustering visualization')
st.write("by [Hassan EL QADI](https://www.linkedin.com/in/el-qadi/)")

# data input

st.header("1. Presenting the Data")
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
    st.write("#### Step 1: Define Columns")
    col1, col2 = st.columns(2)
    
    with col1:
        num_columns = st.number_input("Number of columns", min_value=1, value=2, step=1)
        column_names = []
        for i in range(num_columns):
            column_name = st.text_input(f"Column {i+1} name:", key=f'col_name_{i}')
            if column_name:  # Ensure column name is not empty
                column_names.append(column_name)

    with col2:
        st.write("#### Step 2: Generate Random Data")
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
        st.write("#### Step 3: Edit Data (Optional)")
        st.session_state.data = st.data_editor(st.session_state.data, num_rows="dynamic")

# Display the data to be analyzed
if not st.session_state.data.empty:
    st.write("#### Data Preview:")
    st.write(st.session_state.data)
    
    method = st.radio("Choose a clustering method:", ('K-Means Clustering', 'Hierarchical Clustering'), key='clustering_method')

    # Convert all columns of the DataFrame to numeric, coercing errors to NaN, and drop rows with NaN values
    numeric_data = st.session_state.data.apply(pd.to_numeric, errors='coerce').dropna()

    if method == 'K-Means Clustering':
        run_kmeans_clustering(numeric_data)
    elif method == 'Hierarchical Clustering':
        run_hierarchical_clustering(numeric_data)