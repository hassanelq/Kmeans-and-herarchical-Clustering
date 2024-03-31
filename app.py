
import streamlit as st
import pandas as pd
import numpy as np

# Assuming KmeansClustering and hierarchicalClustering have the necessary methods
from kmeans_Clustering import KmeansClustering
from hierarchical_Clustering import hierarchicalClustering

# Function Definitions
def run_kmeans_clustering(data, n_clusters, distance_metric):
    if data.shape[1] < 2:
        st.error("The dataset must contain at least two numeric columns for clustering.")
        return
    kmeans = KmeansClustering(k=n_clusters, distance=distance_metric)
    labels = kmeans.fit(data.values)
    fig = kmeans.plot_clusters(data.values, labels)
    st.pyplot(fig)

def run_hierarchical_clustering(data, n_clusters_hier):
    if data.shape[1] < 2:
        st.error("The dataset must contain at least two numeric columns for clustering.")
        return
    hierarchical_clustering = hierarchicalClustering(n_clusters=n_clusters_hier, linkage='ward')
    labels = hierarchical_clustering.fit_predict(data.values)
    fig_dendrogram = hierarchical_clustering.plot_dendrogram()
    st.pyplot(fig_dendrogram)
    fig_clusters = hierarchical_clustering.plot_clusters(data.values, labels)
    st.pyplot(fig_clusters)

# Streamlit App Layout
st.title('Clustering Visualization')
st.write("by [Hassan EL QADI](https://www.linkedin.com/in/el-qadi/)")

# Sidebar for clustering options
method = st.sidebar.radio("Choose a clustering method:", ('K-Means Clustering', 'Hierarchical Clustering'), key='clustering_method')

n_clusters = None
distance_metric = None
if method == 'K-Means Clustering':
    n_clusters = st.sidebar.slider('Number of clusters for K-Means', value=3, min_value=2, max_value=50, step=1, key='kmeans_n_clusters')
    distance_metric = st.sidebar.selectbox('Distance metric for K-Means', ('Euclidean', 'Manhattan', 'Cosine'), key='kmeans_distance_metric')

n_clusters_hier = None
if method == 'Hierarchical Clustering':
    n_clusters_hier = st.sidebar.slider('Number of clusters for Hierarchical Clustering', value=3, min_value=2, max_value=50, step=1, key='hier_n_clusters')

# Data input section
data_input_method = st.radio("Choose how to input data:", ('Upload a file', 'Manual input'), key='data_input_method')

if data_input_method == 'Upload a file':
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'txt'])
    if uploaded_file is not None:
        st.session_state.data = pd.read_csv(uploaded_file)
elif data_input_method == 'Manual input':
    num_columns = st.number_input("Number of columns", min_value=1, value=2, step=1)
    column_names = [st.text_input(f"Column {i+1} name:", key=f'manual_col_name_{i}') for i in range(num_columns)]
    num_rows = st.number_input("Number of rows to generate", min_value=1, value=5, step=1)
    min_val, max_val = st.slider("Value range for random data", 0, 100, (0, 100))
    
    if st.button("Generate Data"):
        random_data = np.random.randint(min_val, max_val+1, size=(num_rows, len(column_names)))
        st.session_state.data = pd.DataFrame(random_data, columns=column_names)

# Ensure data preview and clustering execution only if data is available
if 'data' in st.session_state and not st.session_state.data.empty:
    st.write("#### Data Preview:")
    st.write(st.session_state.data)

    # Preprocess data
    numeric_data = st.session_state.data.apply(pd.to_numeric, errors='coerce').dropna()
    
    # Clustering execution
    if method == 'K-Means Clustering' and n_clusters and distance_metric:
        run_kmeans_clustering(numeric_data, n_clusters, distance_metric)
    elif method == 'Hierarchical Clustering' and n_clusters_hier:
        run_hierarchical_clustering(numeric_data, n_clusters_hier)
