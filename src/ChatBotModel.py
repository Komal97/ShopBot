import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense 
import random
import json
import pickle
import nltk 
from nltk.stem.lancaster import LancasterStemmer
import config

class ChatBotModel():

    def __init__(self):
        self.__stemmer = LancasterStemmer()
        self.__words = None
        self.__labels = None
        self.__training = None
        self.__output = None
        self.__data = self.__loadTrainingData()
        self.__model = None

    # load training data
    def __loadTrainingData(self):
        try:
            with open(config.INTENTS_FILE_PATH) as file:
                data = json.load(file)
            return data
        except Exception as e:
            print(f"Exception occured while loading training data: {e}")

    # convert training data to numeric bag of words
    def __createTrainingFormatData(self):

        docs_x = []
        docs_y = []
        words = []
        labels = []
        training = []
        output = []
        for intent in self.__data['intents']:
            for pattern in intent['patterns']:
                wrds = nltk.word_tokenize(pattern)
                words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(intent['tag'])

            if intent['tag'] not in labels:
                labels.append(intent['tag'])

        # stem all words and remove duplicates
        words = [self.__stemmer.stem(word.lower()) for word in words if word not in '?']
        words = sorted(list(set(words)))
        labels = sorted(labels)

        output_empty = [0 for _ in range(len(labels))]

        # create bag of words
        for x, doc in enumerate(docs_x):
            bag = []

            wrds = [self.__stemmer.stem(w) for w in doc]
            for w in words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)

            output_row = output_empty[:]
            output_row[labels.index(docs_y[x])] = 1
            training.append(bag)
            output.append(output_row)

        self.__words = words
        self.__labels = labels 
        self.__training = np.array(training)
        self.__output = np.array(output)

    # save vocabulary in numeric form to pickle file
    def __saveVocab(self):
        try:
            with open(config.VOCAB_PATH, 'wb') as file:
                pickle.dump((self.__words, self.__labels, self.__training, self.__output), file)
        except Exception as e:
            print(f"Exception occured while saving training format data: {e}")
    
    # load vocabulary in numeric form from pickle file
    def __loadVocab(self):
        try:
            with open(config.VOCAB_PATH, 'rb') as file:
                self.__words, self.__labels, self.__training, self.__output = pickle.load(file)
        except Exception as e:
            print(f"Exception occured while loading training format data: {e}")

    # train model and save to disk
    def __trainModel(self):
        self.__createTrainingFormatData()
        self.__saveVocab()

        try:
            self.__model  = Sequential()
            # input layer
            self.__model.add(Dense(8, input_shape = (len(self.__training[0]),), activation = 'relu')) 
            # hidden layer 1
            self.__model.add(Dense(8, activation = 'relu'))   
            # hidden layer 2                            
            self.__model.add(Dense(8, activation = 'relu'))     
            # output layer                          
            self.__model.add(Dense(len(self.__output[0]), activation = 'softmax'))              

            self.__model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
            self.__model.fit(self.__training, self.__output, epochs = 150, batch_size = 8)
            self.__model.save(config.MODEL_PATH)
            print('model saved to disk successfully...')

        except Exception as e:
            print(f"Exception occured while training model: {e}")

    # function to train a chatbot model
    def trainChatBotModel(self):
        self.__trainModel()
    
    # convert inputs data to numeric bag of words
    def __createPredictFormatData(self, sentence):
        
        bag = [0 for _ in range(len(self.__words))]

        s_words = nltk.word_tokenize(sentence)
        s_words = [self.__stemmer.stem(word.lower()) for word in s_words]

        for se in s_words:
            for i, w in enumerate(self.__words):
                if w == se:
                    bag[i] = 1
        #return np.array(bag)
        return bag

    # function to predict response
    def predictChatBotModel(self, inp):
    
        try:
            # load vocab 
            self.__loadVocab()

            # load trained model
            self.__model = load_model(config.MODEL_PATH)

            # find probability for each label
            results = self.__model.predict([self.__createPredictFormatData(inp)])
            
            # findex index of max probability
            result_index = np.argmax(results)   

            # if probability is less than 0.6 means tag is most likely identified wrong
            if results[0][result_index] < 0.6:                    
                return "I am not able to understand, please rephrase."
            else:
                # find the possible response
                tag = self.__labels[result_index]                           

                for intent in self.__data['intents']:
                    if intent['tag'] == tag:
                        responses = intent['responses']

                # return response
                return random.choice(responses)
           
        except Exception as e:
            print(f'Exception occured while predicting: {e}')
            return str("We will get back to you soon...")
