from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import linear_kernel
from os.path import dirname, join as pjoin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sentence_embed import build_encoding, load_infersent_model
from questions import isQuestion, get_Model
from textblob import TextBlob
from spacy import displacy

import matplotlib.pyplot as plot
import os
import pandas as pd
import spacy

def qa_pairs():
    vect = TfidfVectorizer(stop_words='english', max_df=0.50, min_df=2)
    email_dataFrame =  read_email_data()
    sentences = []
    nlp = spacy.load("en_core_web_sm")
    load_infersent_model()
    for i, text in email_dataFrame.iteritems():
        blob = TextBlob(text)
        for sentence in blob.sentences:
            sentences.append(sentence.raw)
    number = len(sentences)
    model, tfidf_vect = get_Model()

    # print(f'Question: what is the process to get a new hire onboarded?')
    # print(f'Chosen answer by cosine sim: The new hire goes through a dedicated weeklong onboarding process with HR - see details here -')
    # print(f'Chosen answer by eucledean sim: {email_dataFrame[index+ind2]}')
    detected_questions = []
    for index, each in enumerate(sentences):
        isQ = isQuestion(each, model, tfidf_vect)
        # print(f'{each} is a question - {isQ}')        
        if isQ and index+10 < number:
            detected_questions.append(each)
            candidateSentences= sentences[index:index+10]
            candidateAnswers = []
            candidateAnswers.append(each)
            for every in candidateSentences:
                if not isQuestion(every, model, tfidf_vect):
                    candidateAnswers.append(every)

            cosine_sim, euclidean = build_encoding(candidateAnswers)
            largest, ind = get_Index_Of_Closest(cosine_sim)
            largest, ind2 = get_Index_Of_Closest(euclidean)            
            print(f'Question: {each}')
            print(f'Chosen answer by eucledean sim: {candidateAnswers[ind]}')
            print(f'Chosen answer by cosine sim: {candidateAnswers[ind2]}')

            for answer in candidateAnswers:
                confidence, root = is_root_equal(nlp, each, answer)
                if confidence:                    
                    print(f'Based on equal root {root}, question-answer pair')
                    print(f'Question {each}')
                    print(f'Answer {answer}')


    data = vect.fit_transform(sentences)
    data_dense = data.todense()
    coordinates = PCA(n_components=2).fit_transform(data_dense)
    plot.scatter(coordinates[:, 0], coordinates[:, 1], c='m')
    plot.show()

def is_root_equal(nlp, question, candidateAnswer):
    q = nlp(question)
    a = nlp(candidateAnswer)
    root = ""
    root2 = ""
    for token in q:
        if(token.dep_ == 'ROOT' and token.text != 'is'):
            root = token.text        

    for token in a:
        if(token.dep_ == 'ROOT'and token.text != 'is'):
            root2 = token.text       
    areSame = root == root2 and root != ''
    # if areSame:             
    #     displacy.serve(q, style="dep")
    #     displacy.serve(a, style="dep")
    return areSame, root

def get_Index_Of_Closest(cosine_sim):
    largest = -1
    index = 0
    for i in range(1, len(cosine_sim[0])):
        if cosine_sim[0][i] > largest:
            largest = cosine_sim[0][i]
            index = i
    return largest, index

# helper code to read enron emails.
# credit and referenced from https://github.com/anthdm/ml-email-clustering
def read_email_data():
    file_path = dirname(os.path.realpath(__file__))
    emails_data = pd.read_csv(f'{file_path}\\data\\split_emails.csv')
    email_dataFrame = pd.DataFrame(parse_into_emails(emails_data.message))
    email_dataFrame.drop(email_dataFrame.query("body == '' | to == '' | from_ == ''").index, inplace=True)
    email_dataFrame.drop_duplicates(inplace=True)
    return email_dataFrame['body']

# helper code to read enron emails.
# credit and referenced from https://github.com/anthdm/ml-email-clustering
def parse_into_emails(messages):
    emails = []
    for each in messages:
        email_lines = each.split('\n')
        email = {}
        contentSoFar = ''
        for eachline in email_lines:
            if ':' in eachline:
                temp = eachline.split(':')
                key = temp[0].lower().strip()
                value = temp[1].strip()
                if key == 'from' or key == 'to':
                    email[key] = value
            else:
                contentSoFar += eachline.strip()
                email['body'] = contentSoFar
        emails.append(email)
    return {
        'body': getListFromMap(emails, 'body'), 
        'to': getListFromMap(emails, 'to'), 
        'from_': getListFromMap(emails, 'from')
    }

# helper code to read enron emails.
# credit and referenced from https://github.com/anthdm/ml-email-clustering
def getListFromMap(emails, key):
    results = []
    for email in emails:
        if key not in email:
            results.append('')
        else:
            results.append(email[key])
    return results

if __name__ == '__main__':
    qa_pairs()