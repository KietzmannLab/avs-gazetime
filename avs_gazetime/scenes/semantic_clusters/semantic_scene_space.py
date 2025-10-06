"""This script is the backend for visualizing the semantic embedding space for aset of scenes. It runs tsne on the semantic embeddings per scene. 
The size of the dots in the visualization is determined by the mean pairiwse similarity of the embeddings for each scene from all 5 mscoco subjects."""

# run the imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import manifold
from sklearn import metrics
from sklearn import preprocessing
from sklearn import cluster
import scipy

import cycler
import matplotlib

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from skimage.transform import resize
from joblib import Parallel, delayed
from avs_gazetime.config import PLOTS_DIR_NO_SUB, PLOTS_DIR, PLOTS_DIR_BEHAV
import avs_machine_room.dataloader.tools.avs_directory_tools as avs_directory


# set the random seed
seed = 42
rng = np.random.RandomState(seed)

def compute_embedding_similarity_parallel(embeddings_scene, cocoID):
    # compute the pairwise similarity for the embeddings for one scene
    #embeddings_scene should be n_coco_subjects x n_features
    # we will use the cosine similarity for this (pdist "cosine")
    #print(f'Computing similarity for scene {cocoID}')
    ##rint(embeddings_scene.shape)
    pairwise_similarities = scipy.spatial.distance.pdist(embeddings_scene, metric='cosine')
    # we will use the mean pairwise similarity for the scene
    mean_similarity = np.mean(pairwise_similarities)
    return mean_similarity

def read_scene_parallel(cocoID, coco_dir, dataset_names = ['val2017', 'train2017', 'test2017']):
    # read the scene image
    found = False
    for dataset_name in dataset_names:
        scene_fname = os.path.join(coco_dir, dataset_name, str(int(cocoID)).zfill(12) + '.jpg')
        if os.path.exists(scene_fname):
            scene_im = plt.imread(scene_fname)
            found = True
            break
    if not found:
        scene_im = np.NaN
        print(scene_fname)
        print(f"Scene {cocoID} not found")
    return scene_im

if __name__ == '__main__':
    debug = False
 
    basis = "mpnet"
    #basis = "sigOnly"
    n_jobs = -2
    tsne_perplexity = 35
    n_clusters = "precomputed"
    plot_with_ims = False

    
    avs_scene_selection_path = "/share/klab/datasets/avs/input/scene_sampling_MEG/experiment_cocoIDs.csv"
    avs_scenes = pd.read_csv(avs_scene_selection_path)
    coids_dir = "/share/klab/datasets/_for_philip_from_adrien/nsd_tsne"
    input_dir = avs_directory.get_input_dirs(server="uos")
    output_dir = os.path.join(PLOTS_DIR_BEHAV, "semantic_clusters")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    coco_dir = os.path.join(input_dir, "mscoco_scenes")
    dataset_names = ['val2017', 'train2017', 'test2017']
    
    #input/NSD_info/NSD_ids_with_shared1000_and_special100.csv
    nsd_info = pd.read_csv(os.path.join(input_dir, "NSD_info", "NSD_ids_with_shared1000_and_special100.csv"))
    # rename cocoId to cocoID
    nsd_info.rename(columns={'cocoId': 'cocoID'}, inplace=True)
    print(nsd_info)
   
    df_mean_embeddings = pd.read_csv(os.path.join(input_dir, "scene_sampling_MEG", "df_mean_embeddings_clustered_60.csv"))
    # filter the df_mean_embeddings for the scenes that are in the NSD dataset
    df_mean_embeddings = df_mean_embeddings[df_mean_embeddings.cocoID.isin(nsd_info.cocoID.values)]
    
    print(df_mean_embeddings)

   
    



    # get the lists from the average embeddings columns
    embeddings_mean = np.array([np.fromstring(embedding[1:-1], sep=' ') for embedding in df_mean_embeddings['average_embedding']])
    print(embeddings_mean.shape)
    
    if debug:
        print(f"Embeddings shape: {embeddings_mean.shape}")
        n_iter = 251
        # subselect the embeddings for debugging
        n_subselect = 1000
        embeddings_mean = embeddings_mean[:n_subselect]
        df_mean_embeddings = df_mean_embeddings.iloc[:n_subselect]
        
    else:
        n_iter = 1000
  
        
    print(df_mean_embeddings)
    # compute the tSNE for the mean embeddings
    
    tsne_output_path = os.path.join(output_dir, f"tsne_mean_embeddings_{tsne_perplexity}.npy")
    
    
    if os.path.exists(tsne_output_path):
        Y = np.load(tsne_output_path)
        print("Loaded existing t-SNE coordinates.")
    else:
        tSNE = sklearn.manifold.TSNE(n_components=2, perplexity=tsne_perplexity, 
                            early_exaggeration=12.0, learning_rate=200.0, n_iter=n_iter, 
                            n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean',
                            angle=0.5, n_jobs=n_jobs, random_state=seed)
        Y = tSNE.fit_transform(embeddings_mean)
        print("Computed and saved new t-SNE coordinates.")
        np.save(tsne_output_path, Y)
    

    # add the tSNE coordinates to the df
    df_mean_embeddings['x'] = Y[:,0]
    df_mean_embeddings['y'] = Y[:,1]
    #print(np.mean(df_mean_embeddings['average_dissimilarity_scaled']))
    print(np.std(df_mean_embeddings['x']))
    print(np.std(df_mean_embeddings['y']))

    # compute a KMeans clustering for semantic embedding space
    if n_clusters == "precomputed":
        labels = df_mean_embeddings['cluster'].values
        df_mean_embeddings['clid'] = labels
        n_colors = len(np.unique(df_mean_embeddings['cluster']))
    else:
        kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=n_clusters, random_state=seed)  
        labels = kmeans.fit_predict(embeddings_mean)
        df_mean_embeddings['clid'] = labels
        n_colors = n_clusters

    part_of_avs = np.isin(df_mean_embeddings.cocoID.values, avs_scenes.cocoID.values)
    # how many scenes are part of the AVS dataset
    print(f"Number of scenes part of AVS dataset: {np.sum(part_of_avs)}")
    # how many scenes are not part of the AVS dataset
    print(f"Number of scenes not part of AVS dataset: {np.sum(~part_of_avs)}")
    #how many False in average dissimilarity
    
    # add the part_of_avs column to the df_mean_embeddings
    df_mean_embeddings['part_of_avs'] = part_of_avs
 
    ########### Plot fraction of scenes in AVS and NSD per cluster ##########################
    sns.set_context("poster")

    # Create subplots with shared x-axis
    fig, (ax2, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(9, 10), 
                                sharex=True, gridspec_kw={'hspace': 0.1})

    # Relabel the part_of_avs column to be more readable (make categorical)
    df_mean_embeddings['part_of_avs'] = df_mean_embeddings['part_of_avs'].replace({True: 'in AVS', False: 'in NSD'})


    n_scenes_in_avs = np.sum(part_of_avs)
    n_scenes_in_nsd = len(df_mean_embeddings)
    # what is the share of each cluster in avs and nsd. Please note that all scenes are in NSD and only a subset is in AVS
    # first create a df with the number of scenes per cluster

    # add the total number of scnes per cluster (all scenes taken together)
    n_per_cluster_total = np.unique(df_mean_embeddings['clid'], return_counts=True)[1]
    n_per_cluster_avs = np.unique(df_mean_embeddings[part_of_avs]['clid'], return_counts=True)[1]
    
    frac_avs = n_per_cluster_avs / n_scenes_in_avs # 
    frac_nsd = n_per_cluster_total / n_scenes_in_nsd
    # make this percentage
    frac_avs = frac_avs * 100
    frac_nsd = frac_nsd * 100
    
    # make a df and plot the share of each cluster in avs and nsd
    df_cluster_counts = pd.DataFrame({'clid': np.arange(n_colors), 'share_avs': frac_avs, 'share_nsd': frac_nsd})
    # plot 
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 10))
    
    sns.set_context("poster")
    # make bar plot with the share of each cluster in avs and nsd
    
    df_cluster_counts = df_cluster_counts.melt(id_vars='clid', value_vars=['share_avs', 'share_nsd'],
                                                var_name='dataset', value_name='share')
    print(df_cluster_counts)
    # plot bar 
    sns.barplot(data=df_cluster_counts, y='clid', x='share', hue='dataset', dodge=True,
            ax=ax, alpha=1, hue_order=['share_avs', 'share_nsd'], orient='horizontal', palette=['darkgreen', 'darkgrey'])
    # only diaolay every 10 x-ticklabels
    ax.set_yticks(np.arange(0, n_colors, 10))
    ax.set_yticklabels(np.arange(0, n_colors, 10))
    # inverse the y-axis
    ax.invert_yaxis()
    # save the figure
    ax.set_xlabel('proportion of cluster scenes in dataset [%]')
    # despine the axes
    sns.despine()
    ax.set_ylabel('caption LLM-embedding cluster')
    #ax.set_ylabel('number of scenes')
    #ax.set_title('Share of each cluster in AVS and NSD')
    
    # save the figure
    plt.tight_layout()
    fname = f"clusters_share_avs_nsd.pdf"
    fig.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)
   
    ################# tSNE plot with colors for each cluster and part of AVS ##########################
    label_color_hex = dict()
    colors_space = sns.color_palette("husl", n_colors=n_colors)
    for lindex, label in enumerate(np.unique(labels)):
        color_hex = matplotlib.colors.to_hex(colors_space[lindex])
        label_color_hex[label] = color_hex
    for idx in df_mean_embeddings.index:
        df_mean_embeddings.loc[idx, 'color'] = label_color_hex[df_mean_embeddings.loc[idx, 'clid']]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(17,17))
    colors = df_mean_embeddings['color'].values
    #sizes = np.full_like(colors, 200)
    edgecolors = np.where(part_of_avs, 'black', 'black')
    facecolors = np.where(part_of_avs, colors, 'gray')
    
    print(df_mean_embeddings)

    ax.scatter(df_mean_embeddings[~part_of_avs]['x'], df_mean_embeddings[~part_of_avs]['y'], 
               edgecolor=edgecolors[~part_of_avs], facecolor=facecolors[~part_of_avs], s=200)
    ax.scatter(df_mean_embeddings[part_of_avs]['x'], df_mean_embeddings[part_of_avs]['y'], 
               edgecolor=edgecolors[part_of_avs], facecolor=facecolors[part_of_avs], s=200)
    ax.axis('off')
    fig.tight_layout()
    fname = f"t-SNE_all_subjects_perplexity_{tsne_perplexity}.png"
    print("saving figure to ", os.path.join(PLOTS_DIR_NO_SUB, "semantic_clusters", fname))
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
    # dpi=300
    
    #save smaller df with just the relevant columns
    df_mean_embeddings = df_mean_embeddings[['cocoID', 'x', 'y', 'cluster', 'clid', 'color', 'part_of_avs']]
    
    
    ########## Plot piechart of the selected scenes that are part of the AVS dataset ##########

    # Save the combined dataframe
    df_mean_embeddings.to_csv(os.path.join(output_dir, "df_mean_embeddings_clustered_reduced.csv"), index=False)
    print("Saved combined dataframe to CSV.")
    # make a piechart of the selected scenes that are part of the AVS dataset
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3,3))
    # context poster
    sns.set_context("poster")
    sizes = [np.sum(part_of_avs), np.sum(~part_of_avs)]
    labels = ['included in AVS', 'only in NSD']
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    fname = f"piechart_selected_scenes.pdf"

    fig.savefig(output_dir + fname, transparent=True, dpi = 300)
    # for a set of example clusters plot the full embedding cluster and colour the selected scenes (the rest grey).
    # in addition plot a selection of 4 scenes from that cluster
    
    
    ########## Plot individual clusters with highlighted AVS scenes ##########

    n_clusters_to_plot = 20  # Plot at most 12 clusters
    # Plot individual clusters with highlighted AVS scenes
    output_cluster_dir = os.path.join(output_dir, "individual_clusters")
    os.makedirs(output_cluster_dir, exist_ok=True)

    for cluster_id in range(n_clusters_to_plot):
        cluster_fname = os.path.join(output_cluster_dir, f"cluster_{cluster_id}.png")
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(17,17))
        colors = df_mean_embeddings['color'].values
        sizes = np.full_like(colors, 200)
        part_of_cluster = df_mean_embeddings['clid'] == cluster_id
        part_of_cluster_avs = part_of_cluster & part_of_avs
        
        edgecolors = np.where(part_of_cluster, 'black', 'black')
        facecolors = np.where(part_of_cluster, colors, 'gray')
        ax.scatter(df_mean_embeddings[~part_of_cluster]['x'], df_mean_embeddings[~part_of_cluster]['y'], 
                   edgecolor=edgecolors[~part_of_cluster], facecolor=facecolors[~part_of_cluster], s=200)
        ax.scatter(df_mean_embeddings[part_of_cluster]['x'], df_mean_embeddings[part_of_cluster]['y'],
                     edgecolor=edgecolors[part_of_cluster], facecolor=facecolors[part_of_cluster], s=200)
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(cluster_fname, dpi=300)
        plt.close(fig)
        # plot a selection of 4 scenes from that cluster into a new figure
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
        ax = ax.flatten()
        cluster_scenes = df_mean_embeddings[part_of_cluster_avs].sample(4, random_state = seed)
        # how many scenes are part of the AVS dataset
        
        for idx, scene in cluster_scenes.iterrows():
            scene_im = read_scene_parallel(scene.cocoID, coco_dir, dataset_names)
            ax[idx % 4].imshow(scene_im)
            ax[idx % 4].axis('off')
        # set the title of the figure
        fig.suptitle(f"Cluster {cluster_id}")
        fig.tight_layout()
        cluster_fname = os.path.join(output_cluster_dir, f"cluster_{cluster_id}_scenes.pdf")
        fig.savefig(cluster_fname, dpi=300)
        plt.close(fig)
            
        print(f"Saved cluster {cluster_id} to {cluster_fname}")
    print("Done.")
    
    
    
    