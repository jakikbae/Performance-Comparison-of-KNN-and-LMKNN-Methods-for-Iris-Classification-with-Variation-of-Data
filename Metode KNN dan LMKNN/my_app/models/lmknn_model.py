import os
import numpy as np
import pandas as pd
import time
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold

class LMKNN:
    def __init__(self, k):
        self.k = k

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = []
        for x in X:
            labels = self._get_neighbors_labels(x)
            if len(labels) > 0:
                class_means = self._calculate_class_means(labels)
                class_distances = self._calculate_class_distances(x, class_means)
                min_distance_label = min(class_distances, key=class_distances.get)
                y_pred.append(min_distance_label)
            else:
                y_pred.append(None)
        return np.array(y_pred)

    def _get_neighbors_labels(self, x):
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        return k_nearest_labels

    def _calculate_class_means(self, labels):
        class_means = {}
        for label in np.unique(labels):
            class_indices = [i for i, y in enumerate(labels) if y == label]
            class_data = [self.X_train[i] for i in class_indices]
            class_means[label] = np.mean(class_data, axis=0)
        return class_means

    def _calculate_class_distances(self, x, class_means):
        class_distances = {label: self.euclidean_distance(x, mean) for label, mean in class_means.items()}
        return class_distances

class Proses_LMKNN:
    def proses(self, k_lmknn, dataset_name):

        data_dir = "C:/Users/jakikbae/OneDrive/Documents/BISMILLAH KERJA/Portofolio/Metode KNN dan LMKNN/my_app/data/"
        paths = {
            'Iris': os.path.join(data_dir, "Iris.xlsx"),
            'Irisd2': os.path.join(data_dir, "Irisd2.xlsx")
        }

        if dataset_name not in paths:
            print("Invalid dataset name.")
            return None, None

        path = paths[dataset_name]

        if not os.path.exists(path):
            print(f"File {path} not found.")
            return None, None

        df = pd.read_excel(path)
        feature_columns = ['A1', 'A2', 'A3', 'A4'] if dataset_name == "Iris" else ['A1', 'A2']
        X = df[feature_columns].values
        Y = df["Label"].values

        lmknn = LMKNN(k=k_lmknn)

        scores = [['Uji ke', 'Akurasi', 'Precision', 'Recall', 'F-Measure', 'Waktu Komputasi']]
        cv = KFold(n_splits=10, shuffle=True, random_state=42)
        combined_cm = None

        for index_hasil, (train_index, test_index) in enumerate(cv.split(X), 1):
            X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
            start_time = time.time()

            lmknn.fit(X_train, Y_train)
            y_pred = lmknn.predict(X_test)
            Cm = confusion_matrix(Y_test, y_pred)
            classes = sorted(list(set(Y)))

            # print(f"Confusion matrix ke-{index_hasil}")
            # print(pd.DataFrame(Cm, index=classes, columns=classes))

            if combined_cm is None:
                combined_cm = Cm
            else:
                combined_cm += Cm

            acc = round(accuracy_score(Y_test, y_pred), 2)
            prec = round(precision_score(Y_test, y_pred, zero_division=1, average='macro'), 2)
            rec = round(recall_score(Y_test, y_pred, zero_division=1, average='macro'), 2)
            f1 = round(f1_score(Y_test, y_pred, average='macro'), 2)
            execution_time = round((time.time() - start_time), 2)
            scores.append([index_hasil, acc, prec, rec, f1, execution_time])

        avg_scores = ['Rata-rata', 0, 0, 0, 0, 0]
        for i in range(1, 11):
            for j in range(1, 6):
                avg_scores[j] += scores[i][j]
        for i in range(1, 6):
            avg_scores[i] = round((avg_scores[i] / 10), 2)

        scores.append(avg_scores)
        # print("Combined confusion matrix:")
        # print(pd.DataFrame(combined_cm, index=classes, columns=classes))
        # print(scores)

        return scores, pd.DataFrame(combined_cm, index=classes, columns=classes)

# Contoh penggunaan fungsi:
# process_lmknn = Proses_LMKNN()
# scores, combined_cm = process_lmknn.proses(k_lmknn=5, dataset_name='Iris')
