from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np


def classification_knn(trans_induc, vertices_embed=None, label=None, X_train=None, X_test=None, Y_train=None, Y_test=None):

    if trans_induc == 'transductive':
        for k in [10, 20, 30, 40, 50]:
            prediction_label = []
            for idx in range(len(vertices_embed)):
                vertices_embed_tmp = np.delete(vertices_embed, idx, axis=0)
                label_tmp = np.delete(label, idx)
                classifier = KNeighborsClassifier(n_neighbors=k)
                classifier.fit(vertices_embed_tmp, label_tmp)
                prediction_label.append(np.squeeze(classifier.predict([vertices_embed[idx]])))
            print('Test accuracy %d: %.4f' % (k, accuracy_score(label, prediction_label)))
    elif trans_induc == 'inductive':
        for k in [10, 20, 30, 40, 50]:
            classifier = KNeighborsClassifier(n_neighbors=k)
            classifier.fit(X_train, Y_train)
            prediction_label = classifier.predict(X_test)
            print('Test accuracy %d: %.4f' % (k, accuracy_score(Y_test, prediction_label)))