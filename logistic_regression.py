import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression,LinearRegression
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class MyLogisticRegression:
    def __init__(self, dataset_num, perform_test):
        self.training_set = None
        self.test_set = None
        self.model_logistic = LogisticRegression()
        self.model_linear = LinearRegression()
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        self.perform_test = perform_test
        self.dataset_num = dataset_num
        self.read_csv(self.dataset_num)

    def read_csv(self, dataset_num):
        if self.dataset_num == '1':
            train_dataset_file = 'train_q1_1.csv'
            test_dataset_file = 'test_q1_1.csv'
        elif self.dataset_num == '2':
            train_dataset_file = 'train_q1_2.csv'
            test_dataset_file = 'test_q1_2.csv'
        else:
            print("unsupported dataset number")

        self.training_set = pd.read_csv(train_dataset_file, sep=',', header=0)
        if self.perform_test:
            self.test_set = pd.read_csv(test_dataset_file, sep=',', header=0)
        
        
    def model_fit_linear(self):
        y_train = self.training_set['label']
        x_train = self.training_set[['exam_score_1', 'exam_score_2']]

        if self.perform_test:
            self.y_test = self.test_set['label']
            self.X_test = self.test_set[['exam_score_1', 'exam_score_2']]

        self.model_linear.fit(x_train, y_train)
        pass
    
    def model_fit_logistic(self):
        '''
        Initialize self.model_logistic here and call the fit function.
        '''
        y_train = self.training_set['label']
        x_train = self.training_set[['exam_score_1', 'exam_score_2']]

        if self.perform_test:
            self.y_test = self.test_set['label']
            self.X_test = self.test_set[['exam_score_1', 'exam_score_2']]

        self.model_logistic.fit(x_train, y_train)
    
    def model_predict_linear(self):
        '''
        Calculate and return the accuracy, precision, recall, f1, support of the model.
        '''
        self.read_csv(1)
        self.model_fit_linear()
        assert self.model_linear is not None, "Initialize the model, i.e. instantiate the variable self.model_linear in model_fit_linear method"
        assert self.training_set is not None, "self.read_csv function isn't called or the self.training_set hasn't been initialized"

        if self.X_test is not None:
            predictions = self.model_linear.predict(self.X_test)
            
            predictions_binary = (predictions >= 0.5).astype(int)

            accuracy = accuracy_score(self.y_test, predictions_binary)
            precision, recall, f1, support = precision_recall_fscore_support(self.y_test, predictions_binary, average=None)

            print("Linear Regression Accuracy: {:.2f}".format(accuracy))
            print("Precision: ", precision)
            print("Recall: ", recall)
            print("F1 Score: ", f1)
            print("Support: ", support)
        else:
            accuracy = 0.0
            precision, recall, f1, support = np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0])
        
        return [accuracy, precision, recall, f1, support]



    def model_predict_logistic(self):
        '''
        Calculate and return the accuracy, precision, recall, f1, support of the model.
        '''
        self.model_fit_logistic()
        assert self.model_logistic is not None, "Initialize the model, i.e. instantiate the variable self.model_logistic in model_fit_logistic method"
        assert self.training_set is not None, "self.read_csv function isn't called or the self.training_set hasn't been initialized"

        if self.X_test is not None:
            predictions = self.model_logistic.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, predictions)
            
            precision, recall, f1, support = precision_recall_fscore_support(self.y_test, predictions, average='binary')

            print("Logistic Regression Accuracy: {:.2f}".format(accuracy))
            print("Precision: {:.2f}".format(precision))
            print("Recall: {:.2f}".format(recall))
            print("F1 Score: {:.2f}".format(f1))
            print("Support: {}".format(support))
            
        else:
            accuracy = 0.0
            precision, recall, f1, support = np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0])
        
        return [accuracy, precision, recall, f1, support]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Regression')
    parser.add_argument('-d','--dataset_num', type=str, default = "1", choices=["1","2"], help='string indicating datset number. For example, 1 or 2')
    parser.add_argument('-t','--perform_test', action='store_true', help='boolean to indicate inference')
    args = parser.parse_args()
    classifier = MyLogisticRegression(args.dataset_num, args.perform_test)
    acc = classifier.model_predict_linear()
    acc = classifier.model_predict_logistic()
    
