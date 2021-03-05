"""EXTERNAL CLUSTERING INDICES

1.  Entropy #
2.  Purity  #

3.  Precision
4.  Recall
5.  F-measure
6.  F-measure(beta)

7.  Folkes-Mallows index (max,[0,1])    #
8.  Rand Index (max,[0,1])  #
9.  Adjusted Rand Index     #

10. Adjusted Mutual Information
11. Normalized Mutual Information   #

12. Homogenity
13. Completeness
14. V-measure

15. Jaccard Index (max,[0,1])   #
16. Hubert T statistics     #
17. Kulczynski Index (max,[0,1])    #

18. McNemar Index
19. Phi Index
20. Rogers-Tanimoto index (max,[0,1])   #
21. Russel-Rao index

22. Sokal-Sneath index (type 1)
23. Sokal-Sneath index (type 2)

# - teseted
"""

from itertools import combinations
from math import sqrt
# import warnings

import numpy as np
from sklearn import metrics


# warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.filterwarnings("ignore", category=DeprecationWarning) 


class ExternalIndices:

    def __init__(self,class_labels,cluster_labels):

        """Creates class labels and cluster label members and computes yy,yn,ny,nn"""   
        if len(class_labels)!=len(cluster_labels):
            raise Exception("length of class and cluster labels don't match")

        self.class_labels=np.array(class_labels)
        self.cluster_labels=np.array(cluster_labels)
        self.n_samples=self.class_labels.shape[0]

        #compute TP (True-positive:yy,a), FN (False-negative:yn,c), FP (False-Positive:ny,b),TN (True-negative:nn,d)
        TP, FN, FP, TN = 0,0,0,0

        for i,j in combinations(range(self.n_samples),2):
            same_class = (self.class_labels[i]==self.class_labels[j])
            same_cluster = (self.cluster_labels[i]==self.cluster_labels[j])

            #print(self.class_labels[i],self.class_labels[j])
            #print(self.cluster_labels[i],self.cluster_labels[j])

            if same_class and same_cluster:
                TP += 1
                #print("TP")
            elif same_class and not same_cluster:
                FN += 1
                #print("FN")
            elif not same_class and same_cluster:
                FP += 1
                #print("FP")                
            else:
                TN += 1
                #print("TN")
            #print()
        self.TP,self.FN,self.FP,self.TN = TP,FN,FP,TN


    """ Classification Oriented Measures -- start """

    def entropy(self,average=True):
        """Entropy : degree to which each cluster contains objects of a single class.

        References : Chapter 8 Cluster Analysis: Basic Concepts and Algorithms
        range : 0 (cluster labels and class labels match) to 1 (randomness in cluster and class labels)
        """
        '''
        #removing noise points from data : copy code to __init__
        selected_pts=(self.cluster_labels>=0)

        cluster_labels=self.cluster_labels[selected_pts]
        class_labels=self.class_labels[selected_pts]
        '''

        #TODO: if cluster labels allow noise (label = -1), cast to np.array internally
        n_clusters=np.unique(self.cluster_labels).max()+1

        cluster_entropies=np.zeros(n_clusters)
        cluster_sizes=np.zeros(n_clusters)

        A=np.c_[(self.cluster_labels,self.class_labels)]

        for cluster_i in np.unique(A[:,0]):
            corres_class_labels=A[(A[:,0]==cluster_i),1]

            cluster_i_size=corres_class_labels.shape[0]
            class_dist=np.bincount(corres_class_labels)

            valid_class_indices=(class_dist>0)
            class_dist_fraction=class_dist[valid_class_indices]/cluster_i_size

            entropy_i = -1*np.sum(class_dist_fraction*np.log2(class_dist_fraction))

            cluster_entropies[cluster_i] = entropy_i
            cluster_sizes[cluster_i] = cluster_i_size

        if average is False:
            return cluster_entropies

        else:
            return np.sum(cluster_sizes*cluster_entropies)/self.n_samples


    def precision_coefficient(self):
        """ *Precision coefficient : fraction of pairs of points correctly grouped together to total pair of point grouped together
        
        The precision is intuitively the ability of the classifier not to label as positive a sample that is negative (sklearn Documentation),i.e., P(g1/g2)

        range : 0 (worst) to 1 (best)
        """
        #return metrics.precision_score(self.class_labels,self.cluster_labels,average='samples')
        return self.TP/(self.TP+self.FP)


    def recall_coefficient(self):
        """ *Recall Coefficient : fraction of pairs of points that were correctly grouped togther to that supposed to grouped together according to class labels.

        The recall is intuitively the ability of the classifier to find all the positive samples (Sklearn Documentation),i.e., P(g2/g1)

        range : 0 (worst) to 1 (best)
        """
        return self.TP/(self.TP+self.FN)


    def f_measure(self):
        """F-measure : harmonic mean of precision-coefficient and recall-coefficient
        Alias : Czekanowski-Dice index, the Ochiai index

        range : 0 (worst) to 1 (best)
        """
        return 2*self.TP/(2*self.TP+self.FN+self.FP)


    def weighted_f_measure(self,beta=1):
        """F-measure (alpha) : F-measure, which gives beta more weightage to recall over precision

        Reference : https://en.wikipedia.org/wiki/F1_score
                    Clustering Indices, Bernard Desgraupes (April 2013)

        range : 0 (worst) to 1 (best)
        """
        return ((1+beta*beta)*self.TP)/((1+beta*beta)*self.TP+beta*beta*self.FN+self.FP)

    def purity(self):
        """ * Purity
        Reference: http://www.caner.io/purity-in-python.html
        """ 
        A = np.c_[(self.cluster_labels,self.class_labels)]
        n_accurate = 0.
        for j in np.unique(A[:,0]):
            z = A[A[:,0] == j, 1]
            x = np.argmax(np.bincount(z))
            n_accurate += len(z[z == x])

        return n_accurate / A.shape[0]


    """ Classification Oriented Measures -- end """

    #TODO : May remove sklearn implementation after check
    #TODO : Check range, giving value > 1
    def folkes_mallows_index(self):
        """Folkes-Mallows (FM) index is the geometric mean of precision and recall
        
        Reference : http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fowlkes_mallows_score.html#sklearn.metrics.fowlkes_mallows_score
        
        range : 0 (low similarity) to 1 (high similarity)
        """
        return np.true_divide(self.TP,sqrt((self.TP+self.FP)*(self.TP+self.FN)))
        #metrics.fowlkes_mallows_score(self.class_labels, self.cluster_labels)


    '''More cluster indices -- start'''

    def rand_index(self):
        """Rand index : ratio of pairs that are assigned in the same or different clusters in the predicted and true clusterings to total pairs of points
        
        Reference : Clustering -- RUI XU,DONALD C. WUNSCH, II (IEEE Press)
        
        range : 0 (low similarity) to 1 (high similarity)
        """
        return (self.TP+self.TN)/(self.TP+self.TN+self.FP+self.FN)


    def adjusted_rand_index(self):
        """ * Adjusted Rand Index (SKLEARN) : RAND INDEX adjusted for chance
        
        Reference:
        http://scikit-learn.org/stable/modules/clustering.html#adjusted-rand-index 
        
        range : 0 (random labeling) to 1 (identical upto a permutation) (sklearn source : [-1,1])
        """
        return metrics.adjusted_rand_score(self.class_labels,self.cluster_labels) 


    def adjusted_mutual_info(self):
        """Adjusted Mutual Information score (SKLEARN)

        Reference: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html#sklearn.metrics.adjusted_mutual_info_score
        """
        return metrics.adjusted_mutual_info_score(self.class_labels,self.cluster_labels)  


    def normalized_mutual_info(self):
        """Normalized Mutual Information score (SKLEARN)

        Reference: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html#sklearn.metrics.normalized_mutual_info_score
        """
        return metrics.normalized_mutual_info_score(self.class_labels,self.cluster_labels) 


    def homogeneity_score(self):
        """homogeneity_score(SKLEARN) : each cluster contains only members of a single class. (Homogenity)

        Reference: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html#sklearn.metrics.homogeneity_score
        
        range : 0 to 1 (perfectly complete labelling)
        """
        return metrics.homogeneity_score(self.class_labels, self.cluster_labels) 


    def completeness_score(self):
        """completeness_score(SKLEARN) : all members of a given class are assigned to the same cluster (Completeness)
        
        Reference: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.completeness_score.html#sklearn.metrics.completeness_score

        range : 0 to 1 (perfectly complete labelling)
        """
        return metrics.completeness_score(self.class_labels,self.cluster_labels) 


    def v_measure_score(self):
        """ * v_measure_score(SKLEARN) : harmonic mean of completeness and homogenity

        Reference: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html#sklearn.metrics.v_measure_score

        range : 0 to 1 (perfectly complete labelling)
        """
        return metrics.v_measure_score(self.class_labels,self.cluster_labels)  


    def jaccard_coeff(self):
        """Jaccard's coefficient
        References: Chapter 10 - Bible of clustering
                    Clustering Indices, Bernard Desgraupes (April 2013)
        """
        #sklearn testing failed
        return self.TP / (self.TP+self.FN+self.FP)
        #return metrics.jaccard_similarity_score(self.class_labels,self.cluster_labels)

    def hubert_T_index(self):
        """Hubert T statistics : correlation coefficient of the indicator variables
        
        References: Chapter 10 -bible of clustering
                    Clustering Indices, Bernard Desgraupes (April 2013)

        Gamma = ((yy+ny+yn+nn)yy-(yy+yn)(yy+ny))/sqrt((yy+yn) (yy+ny) (nn+yn) (nn+ny))
        range : -1 to 1
        """
        M = self.TP +self.FN + self.FP +self.TN
        m1 = self.TP + self.FP
        m2 = self.TP + self.FN

        numerator = (M*self.TP) - (m1*m2)
        denominator = sqrt(m1 * m2 * (M - m1 ) * (M - m2))
        return numerator/denominator


    def kulczynski_index(self):
        """Kulczynski_index : arithmetic mean of the precision and recall coefficients:

        Reference :  Clustering Indices, Bernard Desgraupes (April 2013)

        KI=1/2((yy(yy+ny)) + (yy/(yy+yn)))
        KI= 1\2(Precision  + RecaLL)
        """
        term1 = self.TP/(self.TP + self.FP)
        term2 = self.TP/(self.TP + self.FN)
        kulczynski_index = 0.5 * (term1 + term2)

        return kulczynski_index

    ''' checked above -- checking only formulas below'''

    def mcnemar_index(self):
        """McNemar Index
        
        References :    Clustering Indices, Bernard Desgraupes (April 2013)
        McN=(nn - ny)/sqrt(nn + ny)
        """  
        numerator = self.TN - self.FP
        denominator = sqrt(self.TN + self.FP)

        return numerator/denominator

    #division by zero error
    def phi_index(self):
        """Phi index
        
        References :    Clustering Indices, Bernard Desgraupes (April 2013)

        The Phi index is a classical measure of the correlation between two dichotomic variables.
        phi =(yy*nn-yn*ny)/((yy+yn)(yy+ny)(yn+nn)(ny+nn))
        """  
        numerator = (self.TP * self.TN) - (self.FN * self.FP)
        denominator = (self.TP + self.FN)*(self.TP + self.FP)*(self.FN + self.TN)*(self.FP + self.TN)
        return numerator/denominator


    def rogers_tanimoto_index(self):
        """Rogers Tanimoto index
        
        The Rogers-Tanimoto index is defined like this:
        RT = (yy + nn)/(yy + nn + 2(yn+ny))
        """
        numerator = self.TP + self.TN
        denominator= numerator + 2*(self.FN+self.FP)
        return numerator/denominator

    def russel_rao_index(self):
        """Russel-Rao index
        
        The Russel-Rao index measures the proportion of concordances between the two partitions. 
        
        The Russel-Rao index is defined like this:
            RR=yy/(yy+yn+ny+nn)
        """
        
        denominator = self.TP + self.FN + self.FP + self.TN
        return self.TP/denominator

    def sokal_sneath_index1(self):
        """Sokal-Sneath indices
        ss1= yy/(yy+2(yn + ny))
        """
        return self.TP /(self.TP + 2 * ( self.FN + self.FP))

    def sokal_sneath_index2(self):
        """Sokal-Sneath indice      
        ss2= (yy+nn)/(yy+nn+(1/2)(yn+ny))
        """
        numerator = self.TP + self.TN
        denominator = self.TP + self.TN + 0.5 * (self.FN + self.FP)
        return numerator/denominator


EXTERNAL_INDICES_METHOD_NAMES_DICT = {
        'Entropy' : 'entropy',
        'Purity' : 'purity',
        'Precision' : 'precision_coefficient',
        'Recall' : 'recall_coefficient',
        'F' : 'f_measure',
        'Weighted-F' : 'weighted_f_measure',
        'Folkes-Mallows' : 'folkes_mallows_index',
        'Rand' : 'rand_index',
        'Adjusted-Rand' : 'adjusted_rand_index',
        'Adjusted-Mutual-Info' : 'adjusted_mutual_info',
        'Normalized-Mutual-Info' : 'normalized_mutual_info',
        'Homogeneity': 'homogeneity_score',
        'Completeness' : 'completeness_score',
        'V-Measure' : 'v_measure_score',
        'Jaccard' : 'jaccard_coeff',
        'Hubert Γ̂':'hubert_T_index',
        'Kulczynski' : 'kulczynski_index',
        'McNemar' : 'mcnemar_index',
        'Phi' : 'phi_index',
        'Russel-Rao' : 'russel_rao_index',
        'Rogers-Tanimoto' : 'rogers_tanimoto_index',
        'Sokal-Sneath1' : 'sokal_sneath_index1',
        'Sokal-Sneath2' : 'sokal_sneath_index2',
}
