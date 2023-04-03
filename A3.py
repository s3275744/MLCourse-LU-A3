import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator

from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score, completeness_score, homogeneity_score
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

if __name__ == '__main__':

    """Task 1
    Implement code that does the following:
    - Apply KMeans and KMedoids using k=5
    - Make three plots side by side, showing the clusters identified by the models and the ground truth
    - Plots should have titles, axis labels, and a legend.
    - The plots should also show the centroids of the KMeans and KMedoids clusters.
      Use a different marker style to make them clearly identifiably.
    """

    url = 'https://raw.githubusercontent.com/MLCourse-LU/Datasets/main/vehicles.csv'
    df = pd.read_csv(url, header=0)

    X = df.iloc[:, :-1]  # all except the last column
    y = df.iloc[:, -1]  # the last column

    k = 5

    # KMeans
    kmeans_model = KMeans(n_clusters=k)
    kmeans_model.fit(X)

    # KMedoids
    kmedoids_model = KMedoids(n_clusters=k)
    kmedoids_model.fit(X)

    # Labels
    x = df['weight']
    y = df['speed']
    c = df['label']

    # Plotting
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 7), sharex=True, sharey=True)

    # KMeans plot
    kmeans_data = ax0.scatter(x, y, c=kmeans_model.labels_, s=5)
    kmeans_centroids = ax0.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:, 1], marker='x',
                                   s=50, linewidths=3, color='orange')
    ax0.set_title('KMeans Clustering')
    ax0.set_xlabel('X0')
    ax0.set_ylabel('X1')
    ax0.legend((kmeans_data, kmeans_centroids), ('Data Points', 'Centroids'))

    # KMedoids plot
    kmedoids_data = ax1.scatter(x, y, c=kmedoids_model.labels_, s=5)
    kmedoids_centroids = ax1.scatter(kmedoids_model.cluster_centers_[:, 0], kmedoids_model.cluster_centers_[:, 1],
                                     marker='x', s=50, linewidths=3, color='orange')
    ax1.set_title('KMedoids Clustering')
    ax1.set_xlabel('weight')
    ax1.set_ylabel('speed')
    ax1.legend((kmedoids_data, kmedoids_centroids), ('Data Points', 'Centroids'))

    # Ground truth plot
    ground_truth_scatter = ax2.scatter(x, y, c=c, s=5)
    ax2.set_title('Ground Truth')
    ax2.set_xlabel('weight')
    ax2.set_ylabel('speed')
    ax2.legend((ground_truth_scatter,), ('Data Points',))

    plt.show()



    """ Task 2
    Apply KMeans and KMedoids to the following dataset. The choice of K is up to you.
    - Make plots of the best results you got with KMeans and KMedoids.
    - In the title of the plots, indicate the K used, and the homogeneity and completeness score achieved.
    """

    url = 'https://raw.githubusercontent.com/MLCourse-LU/Datasets/main/dataset-task-2.csv'
    df = pd.read_csv(filepath_or_buffer=url, header=0)
    X = df.iloc[:, :-1].values  # all except the last column
    y = df.iloc[:, -1].values  # the last column
    feature_names = df.columns[:-1]
    k = 4

    # Labels
    x = df['X0']
    y = df['X1']
    c = df['Y']

    # Plotting
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 7), sharex=True, sharey=True)

    # base values KMeans
    completeness_score_KMeans = 0
    homogeneity_score_KMeans = 0
    combined_KMeans = 0
    k_values_KMeans = 0

    # base values KMedoids
    completeness_score_KMedoids = 0
    homogeneity_score_KMedoids = 0
    combined_KMedoids = 0
    k_values_KMedoids = 0


    k_values = range(2, 11)  # loop over k=2 to k=10 to find best k_value
    for k in k_values:

        # KMeans
        kmeans_model = KMeans(n_clusters=k)
        kmeans_model.fit(X)

        if (homogeneity_score(c, kmeans_model.labels_) + completeness_score(c, kmeans_model.labels_)) > combined_KMeans:
            combined_KMeans = (homogeneity_score(c, kmeans_model.labels_) + completeness_score(c, kmeans_model.labels_))
            completeness_score_KMeans = completeness_score(c, kmeans_model.labels_)
            homogeneity_score_KMeans = homogeneity_score(c, kmeans_model.labels_)
            k_values_KMeans = k

        # KMedoids
        kmedoids_model = KMedoids(n_clusters=k)
        kmedoids_model.fit(X)

        if (homogeneity_score(c, kmedoids_model.labels_) + completeness_score(c, kmedoids_model.labels_)) > combined_KMedoids:
            combined_KMedoids = (homogeneity_score(c, kmedoids_model.labels_) + completeness_score(c, kmedoids_model.labels_))
            completeness_score_KMedoids = completeness_score(c, kmedoids_model.labels_)
            homogeneity_score_KMedoids = homogeneity_score(c, kmedoids_model.labels_)
            k_values_KMedoids = k


    # KMeans plot
    kmeans_model = KMeans(n_clusters=k_values_KMeans)
    kmeans_model.fit(X)

    kmeans_data = ax0.scatter(x, y, c=kmeans_model.labels_, s=5)
    ax0.set_title('KMeans Clustering')
    ax0.set_xlabel('X0')
    ax0.set_ylabel('X1')
    ax0.legend((kmeans_data,), ('Data Points',))

    # KMedoids plot
    kmedoids_model = KMedoids(n_clusters=k_values_KMedoids)
    kmedoids_model.fit(X)

    kmedoids_data = ax1.scatter(x, y, c=kmedoids_model.labels_, s=5)
    ax1.set_title('KMedoids Clustering')
    ax1.set_xlabel('X0')
    ax1.set_ylabel('X1')
    ax1.legend((kmedoids_data,), ('Data Points',))

    # Ground truth plot
    ground_truth_scatter = ax2.scatter(x, y, c=c, s=5)
    ax2.set_title('Ground Truth')
    ax2.set_xlabel('X0')
    ax2.set_ylabel('X1')
    ax2.legend((ground_truth_scatter,), ('Data Points',))

    plt.savefig('Figure2.pdf')
    plt.show()


    """ Task 3

    Adapt the code used in the example to instead make a comparison between KMeans and KMedoids.
    - Set K at 4
    - Make a plot for both models
    """

    url = 'https://raw.githubusercontent.com/MLCourse-LU/Datasets/main/dataset-task-3.csv'
    df = pd.read_csv(filepath_or_buffer=url, header=0)

    X = df.values

    range_n_clusters = [4]

    for n_clusters in range_n_clusters:

        ### FIRST, DO THE ACTUAL CLUSTERING ###

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMedoids(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(f"For n_clusters = {n_clusters}, the average silhouette_score is :{silhouette_avg}")

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        ### SET UP THE TWO-PLOT FIGURE ###

        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        ### LEFT PLOT ###

        # The 1st subplot is the silhouette plot
        ax1.set_title("The silhouette plot for the various clusters.")

        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_xlim([-0.1, 1])
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylabel("Cluster label")
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
        ax1.set_yticks([])  # Clear the yaxis labels / ticks

        y_lower = 10  # starting position on the y-axis of the next cluster to be rendered

        for i in range(n_clusters):  # Here we make the colored shape for each cluster

            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()

            # Figure out how much room on the y-axis to reserve for this cluster
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            y_range = np.arange(y_lower, y_upper)

            # Use matplotlib color maps to make each cluster a different color, based on the total number of clusters.
            # We use this to make sure the colors in the right plot will match those on the left.
            color = cm.nipy_spectral(float(i) / n_clusters)

            # Draw the cluster's overall silhouette by drawing one horizontal stripe for each datapoint in it
            ax1.fill_betweenx(y=y_range,  # y-coordinates of the stripes
                              x1=0,  # all stripes start touching the y-axis
                              x2=ith_cluster_silhouette_values,  # ... and they run as far as the silhouette values
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ### RIGHT PLOT ###

        # 2nd Plot showing the actual clusters formed
        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(f"Silhouette analysis for KMedoids clustering on sample data with n_clusters = {n_clusters}",
                     fontsize=14, fontweight='bold')

        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)  # make the colors match with the other plot
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=200, edgecolor='k')
        # Put numbers in those circles
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')

    ### FINISH ###
    plt.show()  # show all the plots, in the order they were generated


    for n_clusters in range_n_clusters:

        ### FIRST, DO THE ACTUAL CLUSTERING ###

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(f"For n_clusters = {n_clusters}, the average silhouette_score is :{silhouette_avg}")

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        ### SET UP THE TWO-PLOT FIGURE ###

        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        ### LEFT PLOT ###

        # The 1st subplot is the silhouette plot
        ax1.set_title("The silhouette plot for the various clusters.")

        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_xlim([-0.1, 1])
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylabel("Cluster label")
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
        ax1.set_yticks([])  # Clear the yaxis labels / ticks

        y_lower = 10  # starting position on the y-axis of the next cluster to be rendered

        for i in range(n_clusters):  # Here we make the colored shape for each cluster

            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()

            # Figure out how much room on the y-axis to reserve for this cluster
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            y_range = np.arange(y_lower, y_upper)

            # Use matplotlib color maps to make each cluster a different color, based on the total number of clusters.
            # We use this to make sure the colors in the right plot will match those on the left.
            color = cm.nipy_spectral(float(i) / n_clusters)

            # Draw the cluster's overall silhouette by drawing one horizontal stripe for each datapoint in it
            ax1.fill_betweenx(y=y_range,  # y-coordinates of the stripes
                              x1=0,  # all stripes start touching the y-axis
                              x2=ith_cluster_silhouette_values,  # ... and they run as far as the silhouette values
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ### RIGHT PLOT ###

        # 2nd Plot showing the actual clusters formed
        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(f"Silhouette analysis for KMeans clustering on sample data with n_clusters = {n_clusters}",
                     fontsize=14, fontweight='bold')

        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)  # make the colors match with the other plot
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=200, edgecolor='k')
        # Put numbers in those circles
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')

    ### FINISH ###
    plt.show()  # show all the plots, in the order they were generated




    """Task 4

    Write code to generate elbow plots for the datasets given here.
    Use them to figure out the likely K for each of them.
    Put the plots you used to make your decision in your report.
    """
for dataset in ['4a', '4b']:
    url = f'https://raw.githubusercontent.com/MLCourse-LU/Datasets/main/dataset-task-{dataset}.csv'
    df = pd.read_csv(filepath_or_buffer=url, header=0)
    X = df.values

    # Set up a list of k values to try
    k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    # Create an empty list to store inertia values
    inertia = []

    # Loop over each k value and fit the KMeans model
    for k in k_list:
        model = KMeans(n_clusters=k)
        model.fit(X)
        inertia.append(model.inertia_)

    # Plot the elbow curve for the dataset
    fig, ax = plt.subplots()
    ax.set_xticks(k_list)
    ax.set_xticklabels(k_list)

    # Plot the data as a line plot
    ax.plot(k_list, inertia)

    # Set the x and y axis labels
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia')

    # Set the title of the plot
    ax.set_title(f'Elbow Curve for Dataset {dataset}')

    # Display the plot
    plt.show()

    # Save the figure as a pdf file
    plt.tight_layout()
    plt.savefig(f'Figure 4-{dataset}.pdf')



    """ Task 5

    Write code that generates a dataset with k >= 3 and 2 feature dimensions.
    - It should be easy for a human to cluster with the naked eye.
    - It should NOT be easy for KMedoids to cluster, even when using the correct value of K.
    - Plot the ground truth of your dataset, so that we can see that a human indeed clusters it easily.
    - Plot the clustering found by KMedoids to show that it doesn't do it well.
    """

    # TODO: student code here

    plt.tight_layout()
    plt.savefig('Figure5.pdf')
    plt.show()

