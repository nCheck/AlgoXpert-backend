import numpy as np 
import pandas as pd  
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN 
from sklearn.decomposition import PCA 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.preprocessing import StandardScaler, normalize 
from sklearn.metrics import silhouette_score 
import scipy.cluster.hierarchy as shc 
from sklearn.utils import resample





def dbscan(X_principal):

    # DBSCAN

    X = np.array(X_principal)

    eps = 0.001

    done = False

    while eps < 0.02:
        
        eps += 0.0025
        
        for min_samples in range(10 , 50 , 3):
            
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(X) 
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool) 
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_ 

            # Number of clusters in labels, ignoring noise if present. 
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            
            if n_clusters_ < 10 or n_clusters_ > 15:
                continue

            print(labels) 


            import matplotlib.pyplot as pl
 

            # Black removed and is used for noise instead. 
            unique_labels = set(labels) 
            colors = list(mcolors.TABLEAU_COLORS)
            print(colors) 
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
            pl.savefig('static/cluster/dbscan.png')
            done = True
            break
        
        if done:
            break

    
    return "done"



def aglo(X_principal):


    #Dendogram

    plt.figure(figsize =(8, 8)) 
    plt.title('Visualising the data') 
    Dendrogram = shc.dendrogram((shc.linkage( resample(X_principal, n_samples=350, random_state=0) , method ='ward')))
    plt.savefig('static/cluster/dendo.png') 



    ac2 = AgglomerativeClustering(n_clusters = 3) 
    
    # Visualizing the clustering 
    plt.figure(figsize =(6, 6))
    plt.title('Number of Clusters = 3')  
    plt.scatter(X_principal['P1'], X_principal['P2'],  
            c = ac2.fit_predict(X_principal), cmap ='rainbow') 

    plt.savefig('static/cluster/aglo_3.png')

    ac2 = AgglomerativeClustering(n_clusters = 4) 
    
    # Visualizing the clustering 
    plt.figure(figsize =(6, 6))
    plt.title('Number of Clusters = 4') 
    plt.scatter(X_principal['P1'], X_principal['P2'],  
            c = ac2.fit_predict(X_principal), cmap ='rainbow') 
    # plt.show() 
    plt.savefig('static/cluster/aglo_4.png')

    return "done"








def clustering(X_principal):

    dbscan(X_principal)
    aglo(X_principal)