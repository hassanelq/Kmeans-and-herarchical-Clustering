# Clustering Visualization

This project provides a Streamlit application for visualizing clustering algorithms. The application supports both K-Means and Hierarchical Clustering methods and includes the Elbow Method to help determine the optimal number of clusters.

## Features

- **K-Means Clustering**: Choose distance metrics (Euclidean, Manhattan, Cosine) and determine the optimal number of clusters using the Elbow Method.
- **Hierarchical Clustering**: Select linkage methods (ward, single, complete, average) and visualize dendrograms.
- **Data Input Options**: Upload a file, manually input data, or generate a random dataset.
- **Data Preview**: View the uploaded or generated data before performing clustering.
- **Interactive Interface**: Easily adjust parameters and see results in real-time.

## Installation

To run this application locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/clustering-visualization.git
    cd clustering-visualization
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

## Usage

### Sidebar Options

- **Clustering Method**: Choose between K-Means Clustering and Hierarchical Clustering.
- **Distance Metric** (for K-Means): Select Euclidean, Manhattan, or Cosine.
- **Use Optimal K** (for K-Means): Use the optimal number of clusters determined by the Elbow Method or manually specify the number of clusters.
- **Linkage Method** (for Hierarchical): Select ward, single, complete, or average.
- **Number of Clusters**: Adjust the number of clusters for the selected method.

### Data Input Methods

1. **Upload a File**: Upload a CSV or TXT file.
2. **Manual Input**: Specify the number of columns and rows, and generate random data within a specified range.
3. **Random Dataset**: Load a pre-defined random dataset.

### Running Clustering

After uploading or generating data:
1. Preview the data to ensure correctness.
2. Run the selected clustering method with the specified parameters.
3. Visualize the results, including cluster plots and dendrograms for hierarchical clustering.

## Code Structure

- **app.py**: Main application script.
- **kmeans_Clustering.py**: Contains the KmeansClustering class and related methods.
- **hierarchical_Clustering.py**: Contains the hierarchicalClustering class and related methods.
- **elbow_Method.py**: Contains the ElbowMethod class for determining the optimal number of clusters.
- **style.py**: Contains styles for the Streamlit app.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

Developed by [Hassan EL QADI](https://www.linkedin.com/in/el-qadi/).

## Acknowledgments

- Streamlit for providing an easy-to-use platform for creating data apps.
- scikit-learn for providing clustering algorithms and tools.
- Matplotlib for data visualization.

Feel free to contribute to this project by submitting issues or pull requests.

