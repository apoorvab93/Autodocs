
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import pickle
import os

from os.path import dirname, join as pjoin

from nltk import word_tokenize
from nltk import corpus
from nltk import NaiveBayesClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import tree, svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import  MLPClassifier

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from random import shuffle

def mlp_TrainedModel(training_data, training_labels, testing_data, testing_labels, tfidf_vect):
    grid_search_parameters = {
        'hidden_layer_sizes': [(100,)],
        'activation': ['relu'],
        'solver': ['adam'],
        'alpha': [0.05],
        'learning_rate': ['adaptive']
    }

    mlp_classifier = MLPClassifier()
    classifier = GridSearchCV(
        estimator=mlp_classifier,
        param_grid = grid_search_parameters,
        cv=5, scoring='accuracy', verbose=0, n_jobs=-1)

    
    classifier.fit(training_data, training_labels)
    prediction = classifier.predict(testing_data)
    f1_score = metrics.f1_score(testing_labels, prediction, average='macro')
    print(f'F1 score for MLP ANN {f1_score}')
    # Prints Confusion matrix
    c_matrix = confusion_matrix(testing_labels, prediction)
    print(f'Confusion matrix {c_matrix}')

    # Prints Performance of the model (accuracy, f-score, precision, recall)
    print(classification_report(testing_labels, prediction)) 

    # classify a new sentence
    df= pd.DataFrame({'text': ['What is up', 'How do I find the results?'],'class': [1, 1]})
    print(classifier.predict(tfidf_vect.transform(df['text'].values).toarray())) 

    actuals, predicts = plot_roc_curve(testing_labels, prediction, 'MLP ANN classifier')

    return classifier, actuals, predicts      

def randomForest_TrainedModel(training_data, training_labels, testing_data, testing_labels, tfidf_vect):
    random_forest_classifier = RandomForestClassifier(n_estimators=10)

    random_forest_classifier.fit(training_data, training_labels)
    prediction = random_forest_classifier.predict(testing_data)

    f1_score = metrics.f1_score(testing_labels, prediction, average='macro')
    print(f'F1 score for Random Forest {f1_score}')
    # Prints Confusion matrix
    c_matrix = confusion_matrix(testing_labels, prediction)
    print(f'Confusion matrix {c_matrix}')

    # Prints Performance of the model (accuracy, f-score, precision, recall)
    print(classification_report(testing_labels, prediction)) 

    # classify a new sentence
    df= pd.DataFrame({'text': ['What is up', 'How do I find the results?'],'class': [1, 1]})
    print(random_forest_classifier.predict(tfidf_vect.transform(df['text'].values).toarray())) 

    actuals, predicts = plot_roc_curve(testing_labels, prediction, 'Random Forest')

    return random_forest_classifier, actuals, predicts  

def naiveBayes_TrainedModel(training_data, training_labels, testing_data, testing_labels, tfidf_vect):
    naive_bayes_classifier = GaussianNB()
    naive_bayes_classifier.fit(training_data, training_labels)
    prediction = naive_bayes_classifier.predict(testing_data)

    f1_score = metrics.f1_score(testing_labels, prediction, average='macro')
    print(f'F1 score for Naive Bayes {f1_score}')
    # Prints Confusion matrix
    c_matrix = confusion_matrix(testing_labels, prediction)
    print(f'Confusion matrix {c_matrix}')

    # Prints Performance of the model (accuracy, f-score, precision, recall)
    print(classification_report(testing_labels, prediction)) 

    # classify a new sentence
    df= pd.DataFrame({'text': ['What is up', 'How do I find the results?'],'class': [1, 1]})
    print(naive_bayes_classifier.predict(tfidf_vect.transform(df['text'].values).toarray()))

    actuals, predicts = plot_roc_curve(testing_labels, prediction, 'NaiveBayes')

    return naive_bayes_classifier, actuals, predicts

def decisionTree_TrainedModel(training_data, training_labels, testing_data, testing_labels, tfidf_vect):
    treemodel = tree.DecisionTreeClassifier(criterion='entropy')
        
    # The following code trains the model on train data and train labels
    treemodel.fit(training_data, training_labels)        
    prediction = treemodel.predict(testing_data)
    f1_score = metrics.f1_score(testing_labels, prediction, average='macro')
    print(f'F1 score for Decision Tree {f1_score}')
    # Prints Confusion matrix
    c_matrix = confusion_matrix(testing_labels, prediction)
    print(f'Confusion matrix {c_matrix}')

    # Prints Performance of the model (accuracy, f-score, precision, recall)
    print(classification_report(testing_labels, prediction)) 

    # classify a new sentence
    df= pd.DataFrame({'text': ['What is up', 'How do I find the results?'],'class': [1, 1]})
    print(treemodel.predict(tfidf_vect.transform(df['text'].values).toarray()))

    actuals, predicts = plot_roc_curve(testing_labels, prediction, 'Decision Tree')
    return treemodel, actuals, predicts

def svm_TrainedModel(training_data, training_labels, testing_data, testing_labels, tfidf_vect):
    svm_classifier =  svm.SVC(kernel='linear')
    # Training the SVM classifier
    svm_classifier.fit(training_data, training_labels)
    # Predicting/Classifying the test data
    prediction = svm_classifier.predict(testing_data)
    f1_score = metrics.f1_score(testing_labels, prediction, average='macro')
    print(f'F1 score for SVM {f1_score}')
    # Prints Confusion matrix
    c_matrix = confusion_matrix(testing_labels, prediction)
    print(f'Confusion matrix {c_matrix}')

    # Prints Performance of the model (accuracy, f-score, precision, recall)
    print(classification_report(testing_labels, prediction))
    df= pd.DataFrame({'text': ['What is up', 'How do I find the results?'],'class': [1, 1]})
    print(svm_classifier.predict(tfidf_vect.transform(df['text'].values).toarray()))
    actuals, predicts = plot_roc_curve(testing_labels, prediction, 'SVM')
    return svm_classifier, actuals, predicts

def logisticRegression_TrainedModel(training_data, training_labels, testing_data, testing_labels, tfidf_vect):
    # 5. Logistic Regression classifier
    logistic_regression_model = LogisticRegression()
    logistic_regression_model.fit(training_data, training_labels)
    prediction = logistic_regression_model.predict(testing_data)
    f1_score = metrics.f1_score(testing_labels, prediction, average='macro')
    print(f'F1 score for Logistic Regression {f1_score}')
    # Prints Confusion matrix
    c_matrix = confusion_matrix(testing_labels, prediction)
    print(f'Confusion matrix {c_matrix}')

    # Prints Performance of the model (accuracy, f-score, precision, recall)
    print(classification_report(testing_labels, prediction)) 

    # classify a new sentence
    df= pd.DataFrame({'text': ['What is up', 'How do I find the results?'],'class': [1, 1]})
    print(logistic_regression_model.predict(tfidf_vect.transform(df['text'].values).toarray()))
    actuals, predicts = plot_roc_curve(testing_labels, prediction, 'Logistic')
    return logistic_regression_model, actuals, predicts

def load_nps_chat_data(randomSeed):
    data = corpus.nps_chat.xml_posts()
    features = [(get_features(each.text), each.get('class')) for each in data]
    features = np.array(features)    
    labels =  set(features[:, 1])
    dictionary = {}
    i = 0
    for each in labels:
        if(each == 'ynQuestion' or each == 'whQuestion'):
            dictionary[each] = 1
        else:
            dictionary[each] = 0

    features[:, 1] = [dictionary[each] for each in features[:, 1]]
    
    tfidf_vect= TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=False)
    input_list = np.array([(each.text, dictionary[each.get('class')]) for each in data])
    class_list = input_list[:, 1]
    input_list = input_list[:, 0]
    df = pd.DataFrame({'text':input_list,'class': class_list})
    X = tfidf_vect.fit_transform(df['text'].values)
    y = df['class'].values

    training_data, testing_data, training_labels, testing_labels = train_test_split(X, y, test_size=0.10, random_state=randomSeed)
    training_data = training_data.toarray()
    testing_data = testing_data.toarray()

    return training_data, training_labels, testing_data, testing_labels, tfidf_vect

def plot_roc_curve( testing_labels, predictions, model):
    testing_labels = np.array([int(each) for each in testing_labels])
    predictions = np.array([int(each) for each in predictions])
    auc_score = roc_auc_score(testing_labels, predictions)
    print(f'AUC Score for {model} is {auc_score}')
    return testing_labels, predictions
    false_positive_rate, true_positive_rate, thresholds = roc_curve(testing_labels, predictions)
    plot.plot(false_positive_rate, true_positive_rate, marker='.', label=model)    

def train_model():
    file_path = dirname(os.path.realpath(__file__))

    # Load nps chat data set
    # Fold the data between training and testing data K times 
    # Train 5 different models with different grid search parameters
    # Compare results
    for i in range(10):
        #K-fold the dataset into different training and testing dataset
        training_data, training_labels, testing_data, testing_labels, tfidf_vect = load_nps_chat_data(randomSeed = 42*i)
        
        # 1.Random Forest classifier    
        classifier, actuals, predictions = randomForest_TrainedModel(training_data, training_labels, testing_data, testing_labels, tfidf_vect)
        file_name = f'{file_path}\\models\\randomForest.sav'
        pickle.dump(classifier, open(file_name, 'wb'))

        false_positive_rate, true_positive_rate, thresholds = roc_curve(actuals, predictions)
        plot.plot(false_positive_rate, true_positive_rate, marker='.', label='Random Forest')

        # 2. Naive Bayes classifier   
        classifier, actuals, predictions = naiveBayes_TrainedModel(training_data, training_labels, testing_data, testing_labels, tfidf_vect)
        file_name = f'{file_path}\\models\\naiveBayest.sav'
        pickle.dump(classifier, open(file_name, 'wb'))

        false_positive_rate, true_positive_rate, thresholds = roc_curve(actuals, predictions)
        plot.plot(false_positive_rate, true_positive_rate, marker='.', label='Naive Bayes')

        # 3. Decision Tree classifier
        classifier, actuals, predictions = decisionTree_TrainedModel(training_data, training_labels, testing_data, testing_labels, tfidf_vect)
        file_name = f'{file_path}\\models\\decisionTree.sav'
        pickle.dump(classifier, open(file_name, 'wb'))
        false_positive_rate, true_positive_rate, thresholds = roc_curve(actuals, predictions)
        plot.plot(false_positive_rate, true_positive_rate, marker='.', label='Decision Tree')

        # 4. SVM classifier
        svm_TrainedModel(training_data, training_labels, testing_data, testing_labels, tfidf_vect)
        file_name = f'{file_path}\\models\\svm.sav'
        pickle.dump(classifier, open(file_name, 'wb'))
        false_positive_rate, true_positive_rate, thresholds = roc_curve(actuals, predictions)
        plot.plot(false_positive_rate, true_positive_rate, marker='.', label='SVM')
        
        # 5. Logistic Regression classifier
        classifier, actuals, predictions = logisticRegression_TrainedModel(training_data, training_labels, testing_data, testing_labels, tfidf_vect)  
        file_name = f'{file_path}\\models\\logisticRegression.sav'
        pickle.dump(classifier, open(file_name, 'wb'))
        false_positive_rate, true_positive_rate, thresholds = roc_curve(actuals, predictions)
        plot.plot(false_positive_rate, true_positive_rate, marker='.', label='Logistic Regression')

        # 6. MLP ANN classifier
        classifier, actuals, predictions = mlp_TrainedModel(training_data, training_labels, testing_data, testing_labels, tfidf_vect)
        file_name = f'{file_path}\\models\\mlp.sav'
        pickle.dump(classifier, open(file_name, 'wb'))
        false_positive_rate, true_positive_rate, thresholds = roc_curve(actuals, predictions)
        plot.plot(false_positive_rate, true_positive_rate, marker='.', label='MLP ANN')
    
    # axis labels
    plot.xlabel('False Positive Rate')
    plot.ylabel('True Positive Rate')
    # show the legend
    plot.legend()
    # show the plot
    plot.show()
    stop = 10

def get_features(data):
    features = {}
    for each in word_tokenize(data):
        features[f'has ({each.lower()})'] = True
    return features

def get_Model():
    training_data, training_labels, testing_data, testing_labels, tfidf_vect = load_nps_chat_data(randomSeed=42)
    file_path = dirname(os.path.realpath(__file__))
    file_name = f'{file_path}\\models\\svm.sav'
    model = pickle.load(open(file_name, 'rb'))
    result = model.predict(testing_data)
    return model, tfidf_vect

def isQuestion(sentence, model, tfidf_vect):
    df= pd.DataFrame({'text': [sentence]})    
    result = model.predict(tfidf_vect.transform(df['text'].values).toarray())
    if(result[0] == '0'):
         return False
    return True

if __name__ == "__main__":
    train_model()