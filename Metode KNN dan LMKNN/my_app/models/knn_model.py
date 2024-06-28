import os
import pandas as pd
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class KNN:
    def proses(self, k, dataset_name):
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
        
        if dataset_name == 'Iris':
            X = df[['A1', 'A2', 'A3', 'A4']].values
            Y = df["Label"].values
        else:
            X = df[['A1', 'A2']].values
            Y = df["Label"].values
        
        # Membuat objek KNN dengan k
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')

        scores = []
        scores.append(['Uji ke', 'Akurasi', 'Precision', 'Recall', 'F-Measure', 'Waktu Komputasi'])

        cv = KFold(n_splits=10, shuffle=True, random_state=42)
        index_hasil = 1

        combined_cm = None

        for index_hasil, (train_index, test_index) in enumerate(cv.split(X), 1):
            X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
            start_time = time.time()
            
            knn.fit(X_train, Y_train)
            y_pred = knn.predict(X_test)
            Cm = confusion_matrix(Y_test, y_pred)
            classes = sorted(list(set(Y)))
            # print("Confusion Matrix ke-", index_hasil)
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
            index_hasil += 1
            
        temp = ['Rata-rata', 0, 0, 0, 0, 0]
        for i in range(1, 11):
            for j in range(1, 6):
                temp[j] += scores[i][j]
        
        for i in range(1, 6):
            temp[i] = round((temp[i] / 10), 2)
        
        scores.append(temp)
        print("Combined Confusion Matrix")
        print(pd.DataFrame(combined_cm, index=classes, columns=classes))
        print(scores)
        return scores, pd.DataFrame(combined_cm, index=classes, columns=classes)
