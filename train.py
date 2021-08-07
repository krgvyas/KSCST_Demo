def define_model():
    #Declare a sequential model.
    model = Sequential()
    #Add two LSTM layers a dropout layer and a dense layer with rectified linear unit as the activation function and a single output unit.
    model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, 300], return_sequences=True))
    model.add(LSTM(64, recurrent_dropout=0.3))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='relu'))

    #Compile the model
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy','mae'])
    model.summary()
    
    #Return the defined model.
    return model
df = pd.read_csv("./fyp/training_set_rel3.tsv", sep='\t', encoding='ISO-8859-1')

X=df
y = df['domain1_score']


#Define 5 splits for KFOLD training.
x = KFold(n_splits = 5, shuffle = True)
output = []
y_pred1 = []

fold = 1
#Perform training by creating a list from the dataset for each train and test datasets for 5 folds.
for train, test in x.split(X):
    print("\nFold {}\n".format(fold))
    #Declare test and train sets for each fold.
    x_train, x_test, y_train, y_test = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]
    
    #Define the test and train essays from the 'essay' column of the dataset.
    training_essays = x_train['essay']
    testing_essays = x_test['essay']
    
    a = []
    
    #Sentence tokenize each training essay.
    for essay in training_essays:
            a = a + essay_sentences(essay, rem_stopwords = True)
            
    no_feat = 300 
    word_count = 40
    no_workers = 4
    cont = 10
    sample = 1e-3

    #Predict the nearby words for each word in the sentence.
    model = Word2Vec(a, workers=no_workers, size=no_feat, min_count = word_count, window = cont, sample = sample)

    #Normalize vectors (Equal length)
    model.init_sims(replace=True)
    #Save the model.
    model.wv.save_word2vec_format('word2vecmodel.bin', binary=True)

    cleaning_train_essays = []
    
    #For each training essay generate a word list.
    for essay_1 in training_essays:
        cleaning_train_essays.append(essay_wordlist(essay_1, rem_stopwords=True))
    #Generate average feature vectors for the word lists.
    Vectors_train = AvgFeatureVectors(cleaning_train_essays, model, no_feat)
    
    #Similarly for the test essays generate word lists and average feature vectors.
    cleaning_test_essays = []
    for essay_1 in testing_essays:
        cleaning_test_essays.append(essay_wordlist( essay_1, rem_stopwords=True ))
    Vectors_test = AvgFeatureVectors( cleaning_test_essays, model, no_feat )
    
    #Reshape the average feature vectors of test and train datasets to the shape of first dimension of the respective data vectors.
    Vectors_train = np.array(Vectors_train)
    Vectors_test = np.array(Vectors_test)
    Vectors_train = np.reshape(Vectors_train, (Vectors_train.shape[0], 1, Vectors_train.shape[1]))
    Vectors_test = np.reshape(Vectors_test, (Vectors_test.shape[0], 1, Vectors_test.shape[1]))
    
    #Assign the defined model.
    lstm_model = define_model()
    #Fit the model.
    lstm_model.fit(Vectors_train, y_train, batch_size=64, epochs=20)
    #Load the model weights.
    lstm_model.load_weights('./fyp/model1.h5')
    y_predict = lstm_model.predict(Vectors_test)
    
    #Save the model when all the folds are completed.
    if fold == 5:
         lstm_model.save('./fyp/model1.h5')
    
    #Round off the predicted value.
    y_predict = np.around(y_predict)
    
    #Generate a kappa score for each fold.
    result = cohen_kappa_score(y_test.values,y_predict,weights='quadratic')
    print("Kappa Score for fold {fold} is {score}".format(fold = fold, score = result))
    #Add each kappa score to the overall score.
    output.append(result)


    #Increment the value of fold.
    fold = fold + 1

