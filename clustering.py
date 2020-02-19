from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score


def clustering_kmeans(trans_induc, vertices_embed=None, label=None, X_train=None, X_test=None, Y_train=None, Y_test=None):

    if trans_induc == 'transductive':
        prediction_label = KMeans(int(max(label))).fit(vertices_embed).predict(vertices_embed)
        print('NMI: %.4f' % (normalized_mutual_info_score(label, prediction_label)))
    elif trans_induc == 'inductive':
        prediction_label = KMeans(int(max(Y_train))).fit(X_train).predict(X_test)
        print('NMI: %.4f' % (normalized_mutual_info_score(Y_test, prediction_label)))

