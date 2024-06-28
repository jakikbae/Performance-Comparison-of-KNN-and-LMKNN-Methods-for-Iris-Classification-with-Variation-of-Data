import os
import numpy as np
import pandas as pd
import time
from collections import Counter
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
        print("Proses LMKNN")
        print(dataset_name)
        # Define paths for the datasets
        data_dir = "C:/Users/jakikbae/OneDrive/Documents/BISMILLAH KERJA/Portofolio/Metode KNN dan LMKNN/my_app/data/"
        iris_path = os.path.join(data_dir, "Iris.xlsx")
        irisd2_path = os.path.join(data_dir, "Irisd2.xlsx")

        # Debugging statements
        print("Current directory:", os.getcwd())
        print("Files in data directory:", os.listdir(data_dir))
        
        if dataset_name == 'Iris':
            path = iris_path
            
        elif dataset_name == 'Irisd2':
            path = irisd2_path
        else:
            print("Invalid dataset name.")
            return None, None
        
        print(f"Attempting to load file: {path}")
        if os.path.exists(path):
            df = pd.read_excel(path)
        else:
            print(f"File {path} not found.")
            return None, None
        
        if  dataset_name == "Iris":

            df = pd.read_excel(path)
            X = df[['A1', 'A2', 'A3', 'A4']].values
            Y = df["Label"].values
            # Membuat objek LMKNN
            lmknn = LMKNN(k=k_lmknn)

            scores = []
            scores.append(['Uji ke', 'Akurasi', 'Precision', 'Recall', 'F-Measure', 'Waktu Komputasi'])

            cv = KFold(n_splits=10, shuffle=True, random_state=42)
            index_hasil = 1

            combined_cm = None

            for index_hasil, (train_index, test_index) in enumerate(cv.split(X), 1):
                X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
                start_time = time.time()

                # Melatih model
                lmknn.fit(X_train, Y_train)

                # Melakukan prediksi
                y_pred = lmknn.predict(X_test)
                Cm = confusion_matrix(Y_test, y_pred)
                classes = sorted(list(set(Y)))
                print("confusion matriks ke-", index_hasil)
                print(pd.DataFrame(Cm, index=classes, columns=classes))
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
                index_hasil += 1

            temp = ['Rata-rata', 0, 0, 0, 0, 0]
            for i in range(1, 11):
                for j in range(1, 6):
                    temp[j] += scores[i][j]

            for i in range(1, 6):
                temp[i] = round((temp[i] / 10), 2)

            scores.append(temp)
            print("combined cm")
            print(pd.DataFrame(combined_cm, index=classes, columns=classes))
            print(scores)
            return scores, pd.DataFrame(combined_cm, index=classes, columns=classes)

        else:
            print(path)

            df = pd.read_excel(path)
            X = df[['A1', 'A2']].values
            Y = df["Label"].values

            lmknn = LMKNN(k=k_lmknn)

            scores = []
            scores.append(['Uji ke', 'Akurasi', 'Precision', 'Recall', 'F-Measure', 'Waktu Komputasi'])

            cv = KFold(n_splits=10, shuffle=True, random_state=42)
            index_hasil = 1

            combined_cm = None

            for index_hasil, (train_index, test_index) in enumerate(cv.split(X), 1):
                X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
                start_time = time.time()

                # Melatih model
                lmknn.fit(X_train, Y_train)

                # Melakukan prediksi
                y_pred = lmknn.predict(X_test)
                Cm = confusion_matrix(Y_test, y_pred)
                classes = sorted(list(set(Y)))
                print("confusion matriks ke-", index_hasil)
                print(pd.DataFrame(Cm, index=classes, columns=classes))
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
                index_hasil += 1

            temp = ['Rata-rata', 0, 0, 0, 0, 0]
            for i in range(1, 11):
                for j in range(1, 6):
                    temp[j] += scores[i][j]
            for i in range(1, 6):
                temp[i] = round((temp[i] / 10), 2)

            scores.append(temp)
            print("combined cm")
            print(pd.DataFrame(combined_cm, index=classes, columns=classes))
            print(scores)

            return scores, pd.DataFrame(combined_cm, index=classes, columns=classes)