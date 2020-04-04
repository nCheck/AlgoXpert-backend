import numpy as np 
import pandas as pd  
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN 
from sklearn.decomposition import PCA 
from sklearn.cluster import AgglomerativeClustering , KMeans
from sklearn.preprocessing import StandardScaler, normalize 
from sklearn.metrics import silhouette_score 
import scipy.cluster.hierarchy as shc 
from sklearn.utils import resample
from subprocess import call
from random import random



def dbscan(X_principal , failed = False):

    # DBSCAN

    X = np.array(X_principal)

    eps = 0.001

    done = False
    while eps < 0.03:
        
        eps += 0.0025
        
        for min_samples in range(10 , 50 , 3):
            
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(X) 
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool) 
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_ 

            # Number of clusters in labels, ignoring noise if present. 
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            
            print("current clusters ", n_clusters_)
            if (n_clusters_ < 10 or n_clusters_ > 15) and not failed:
                continue

            print(labels) 


            import matplotlib.pyplot as pl
 

            # Black removed and is used for noise instead. 
            unique_labels = set(labels) 
            colors = list(mcolors.TABLEAU_COLORS)
            print(colors)
            fig = pl.figure() 
            for k, col in zip(unique_labels, colors): 
                if k == -1: 
                    # Black used for noise. 
                    col = 'k'

                class_member_mask = (labels == k) 

                xy = X[class_member_mask & core_samples_mask] 
                pl.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, 
                                                markeredgecolor='k',  
                                                markersize=6) 

                xy = X[class_member_mask & ~core_samples_mask] 
                pl.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, 
                                                markeredgecolor='k', 
                                                markersize=6) 

            pl.title('number of clusters: %d' %n_clusters_)
            name = 'dbscan' + str(int(random()*15)) + '.png'
            fig.savefig('static/cluster/'+name)
            
            done = True
            return name
        
        if done:
            break

    if not failed:
        dbscan(X_principal , True)
    
    return "not_found.jpeg"





def kmeans_cluster( X_principal , n_cluster):

    X = X_principal
    kmeans = KMeans(n_clusters=n_cluster, init='k-means++', max_iter=25, n_init=10, random_state=0)
    pred_y = kmeans.fit_predict(X)
    
    plt.figure(figsize =(6, 6)) 
    plt.scatter(X_principal['P1'], X_principal['P2'])
    plt.title('Number of Clusters = ' + str(n_cluster))
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
    name = 'kmean_clust' + str(int(random()*15)) + '.png'
    plt.savefig('static/cluster/'+name)

    return name


def kmeans(X_principal):
    wcss = []
    for i in range(1, 25):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=15, n_init=10, random_state=0)
        kmeans.fit(X_principal)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize =(6, 6))  
    plt.plot(range(1, 25), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    name = 'kmeans' + str(int(random()*15)) + '.png'
    plt.savefig('static/cluster/'+name)

    return name

def aglo(X_principal):


    #Dendogram

    plt.figure(figsize =(8, 8)) 
    plt.title('Visualising the data') 
    Dendrogram = shc.dendrogram((shc.linkage( resample(X_principal, n_samples=350, random_state=0) , method ='ward')))

    name1 = 'dendo' + str(int(random()*15)) + '.png'
    plt.savefig('static/cluster/'+name1) 



    ac2 = AgglomerativeClustering(n_clusters = 3) 
    
    # Visualizing the clustering 
    plt.figure(figsize =(6, 6))
    plt.title('Number of Clusters = 3')  
    plt.scatter(X_principal['P1'], X_principal['P2'],  
            c = ac2.fit_predict(X_principal), cmap ='rainbow') 

    name3 = 'aglo_3' + str(int(random()*15)) + '.png'
    plt.savefig('static/cluster/'+name3)

    ac2 = AgglomerativeClustering(n_clusters = 4) 
    
    # Visualizing the clustering 
    plt.figure(figsize =(6, 6))
    plt.title('Number of Clusters = 4') 
    plt.scatter(X_principal['P1'], X_principal['P2'],  
            c = ac2.fit_predict(X_principal), cmap ='rainbow') 
    # plt.show() 

    name4 = 'aglo_4' + str(int(random()*15)) + '.png'
    plt.savefig('static/cluster/'+name4)

    return name1 , name3 , name4








def clustering(X_principal):

    call('rm -r static/cluster/*.png',shell=True)
    dbsc = dbscan(X_principal)
    dendo , algo_3 , algo_4 = aglo(X_principal)
    kmean = kmeans(X_principal)
    
    return { "dendo" : "cluster/"+dendo , "algo_3" : "cluster/"+algo_3 ,
             "algo_4" : "cluster/"+algo_4 , "dbscan" : "cluster/"+dbsc , "kmean" : "cluster/"+kmean }