The project contains 4 main python modules
1) questions.py - This file contains code to train a model for classifying a sentence as a question. The code here attempts training over 6 different models and tries grid search and Kfolding of data to achieve the best AUC score for a model. It also prints out the confusion matrix, classification report containing accuracy, precision and recall. It also plots the ROC curve for each of the 6 models being compared. It finally saves the best trained models into the models/{model_name}.sav file for use later
2) sentence_embed.py - This file loads facebook's infersent model and creates a vector embedding for a given set of sentences. Usually the question and canidate answers are passed to this method and it returns the cosine similarity and euclidean similarity between these sentences
3) qa_pairs.py - This is the executable code in the system and contains the main class to be run. The method qa_pairs() loads up a subset of enron emails for evaluation purposes. It then splits up the emails into meaningful sentences using textblob library and then for each sentence, it classifies if the sentence is a question or not. If the sentence is a question, we take the next n sentences and evaluate their cosine distance from the question. The sentence with the smallest angle from the question is selected as the answer. To further boost the results, we run the question and answer pair against spacy's dependency parse tree to find the root of the two sentences. IF they match, it increases the confidence score associated with each prediction.
4) models.py - This contains the pre-trained model acquired from google's  https://github.com/facebookresearch/InferSent repository


data/ 
1) contains a subset of enron emails used in evaluation
2) nltk's chat corpus is pre-loaded with NLTK's installation (use pip install nltk to install if not on the system)

models/
1) contains the trained models created by running questions.py and saving the models for future use.


Steps to run the system -
1) Download Glove data set from here (https://nlp.stanford.edu/projects/glove/) and store it the Glove folder. This is over 1GB in size and hence not included in the package
2) Download fastText data set from here (https://fasttext.cc/docs/en/english-vectors.html) and store it in the fastText folder.This is over 1GB in size and hence not included in package
3) Download https://dl.fbaipublicfiles.com/infersent/infersent1.pkl and https://dl.fbaipublicfiles.com/infersent/infersent2.pkl and store them in the folder called 'encoder'
4) Install all python dependencies
    a) spacy
    b) nltk
    c) sklearn
    d) torch
    e) numpy
    f) matplotlib
    g) pandas
5) Execute qa_pairs.py in the command line or in an editor such as Visual studio Code
Example - cd c:\work && cmd /C "set "PYTHONIOENCODING=UTF-8" && set "PYTHONUNBUFFERED=1" && "C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python37_64\python.exe"  --default --client --host localhost --port 56433 c:\work\Autodocs\qa_pairs.py "
6) See question-answers being predicted


Steps to train the system. This is not a required step because I have already attached the trained model in the models folder
1) Install all python dependencies
    a) spacy
    b) nltk
    c) sklearn
    d) torch
    e) numpy
    f) matplotlib
    g) pandas
2) Execute questions.py in the command line or in an editor such as Visual studio Code
Example - cd c:\work && cmd /C "set "PYTHONIOENCODING=UTF-8" && set "PYTHONUNBUFFERED=1" && "C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python37_64\python.exe"  --default --client --host localhost --port 56433 c:\work\Autodocs\questions.py "