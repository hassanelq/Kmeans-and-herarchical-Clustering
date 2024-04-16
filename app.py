
import streamlit as st
import pandas as pd
import numpy as np

# Assuming KmeansClustering and hierarchicalClustering have the necessary methods
from kmeans_Clustering import KmeansClustering
from hierarchical_Clustering import hierarchicalClustering
from elbow_Method import ElbowMethod
from style import styles 

styles()

# Function Definitions
def run_kmeans_clustering(data, n_clusters, distance_metric):
    if data.shape[1] < 2:
        st.error("The dataset must contain at least two numeric columns for clustering.")
        return
    kmeans = KmeansClustering(k=n_clusters, distance=distance_metric)
    labels = kmeans.fit(data.values)
    fig = kmeans.plot_clusters(data.values, labels)
    st.pyplot(fig)

def run_hierarchical_clustering(data, n_clusters_hier,linkage):
    if data.shape[1] < 2:
        st.error("The dataset must contain at least two numeric columns for clustering.")
        return
    hierarchical_clustering = hierarchicalClustering(n_clusters=n_clusters_hier, linkage=linkage)
    labels = hierarchical_clustering.fit_predict(data.values)
    fig_dendrogram = hierarchical_clustering.plot_dendrogram()
    st.pyplot(fig_dendrogram)
    fig_clusters = hierarchical_clustering.plot_clusters(data.values, labels)
    st.pyplot(fig_clusters)

def run_Elbow_Method(data):
    elbow = ElbowMethod(data, k_range=(1, 10))
    elbow.fit()
    optimal_k = elbow.find_optimal_k()
    fig = elbow.plot_elbow_curve()
    st.write("The Elbow Method is a heuristic used to determine the optimal number of clusters in a dataset. The method consists of plotting the explained variation as a function of the number of clusters, and picking the elbow of the curve as the number of clusters to use.")
    st.pyplot(fig)
    st.write(f"Optimal cluster number: {optimal_k}")
    return optimal_k

# Streamlit App Layout
st.title('Clustering Visualization')
st.write("by [Hassan EL QADI](https://www.linkedin.com/in/el-qadi/)")

# Sidebar for clustering options
method = st.sidebar.radio("Choose a clustering method:", ('K-Means Clustering', 'Hierarchical Clustering'), key='clustering_method')

n_clusters = None
distance_metric = None

if method == 'K-Means Clustering':
    distance_metric = st.sidebar.selectbox('Distance metric', ('Euclidean', 'Manhattan', 'Cosine'), key='kmeans_distance_metric')
    use_optimal_k = st.sidebar.checkbox("Use optimal K from Elbow Method", True, key='use_optimal_k')
    if use_optimal_k:
        if 'optimal_k' not in st.session_state:
            st.session_state.optimal_k = 3
        n_clusters = st.session_state.optimal_k
    else:
        n_clusters = st.sidebar.slider('Number of clusters', value=3, min_value=2, max_value=10, step=1, key='kmeans_n_clusters')


n_clusters_hier = None
if method == 'Hierarchical Clustering':
    linkage = st.sidebar.selectbox('Linkage method', ('ward', 'single', 'complete', 'average'), key='hier_linkage')
    n_clusters_hier = st.sidebar.slider('Number of clusters', value=3, min_value=2, max_value=10, step=1, key='hier_n_clusters')
    
# Data input section
st.write("### Data Input:")
data_input_method = st.radio("Choose how to input data:", ('Upload a file', 'Manual input', 'Random dataset'), key='data_input_method')

if data_input_method == 'Upload a file':
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'txt'])
    if uploaded_file is not None:
        st.session_state.data = pd.read_csv(uploaded_file)
elif data_input_method == 'Manual input':
    num_columns = st.number_input("Number of columns", min_value=1, value=2, step=1)
    column_names = [st.text_input(f"Column {i+1} name:", key=f'manual_col_name_{i}') for i in range(num_columns)]
    num_rows = st.number_input("Number of rows to generate", min_value=1, value=100, step=1)
    min_val, max_val = st.slider("Value range for random data", 0, 100, (0, 100))
    
    if st.button("Generate Data"):
        random_data = np.random.randint(min_val, max_val+1, size=(num_rows, len(column_names)))
        st.session_state.data = pd.DataFrame(random_data, columns=column_names)
elif data_input_method == 'Random dataset':
    st.session_state.data = pd.read_csv('datasets/data1.txt', header=None)

# Ensure data preview and clustering execution only if data is available
if 'data' in st.session_state and not st.session_state.data.empty:
    st.write("#### Data Preview:")
    st.write(st.session_state.data)

    # Preprocess data
    numeric_data = st.session_state.data.apply(pd.to_numeric, errors='coerce').dropna()
    
    # Clustering execution
    if method == 'K-Means Clustering' and n_clusters and distance_metric:
        st.write("### Choosing the best K parameter for K-Means Clustering:")
        st.session_state.optimal_k = run_Elbow_Method(numeric_data)
        st.write("### K-Means Clustering:")
        run_kmeans_clustering(numeric_data, n_clusters, distance_metric)
            
    elif method == 'Hierarchical Clustering' and n_clusters_hier:
        run_hierarchical_clustering(numeric_data, n_clusters_hier ,linkage)
