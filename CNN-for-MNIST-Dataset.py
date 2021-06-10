from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization
from tensorflow.keras.optimizers import SGD

def loadDataset():
        """Loads the MNIST handwritten digit training and test sets.  Reshapes their feature vectors
        and uses one-hot encoding for target values.
        """

        (trainX, trainY), (testX, testY) = mnist.load_data()

        # Reshape feature vectors to have one channel
        trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
        testX = testX.reshape((testX.shape[0], 28, 28, 1))

	# One-hot encode target values
        trainY = to_categorical(trainY)
        testY = to_categorical(testY)

        return trainX, trainY, testX, testY

def scalePixels(trainingSet, testSet):
        """Normalises each pixel to be in the range 0 - 1 to prevent exploding
        gradient problems.

        Parameters
        ==========
        trainingSet : array
            The data with which to train the network.

        testSet : array
            The data with which to test the network.
        """
        
	# Convert from integers to floats
        normalisedTrainingSet = trainingSet.astype('float32')
        normalisedTestSet = testSet.astype('float32')
	
	# Normalise to range 0-1
        normalisedTrainingSet = normalisedTrainingSet / 255.0
        normalisedTestSet = normalisedTestSet / 255.0

        return normalisedTrainingSet, normalisedTestSet

def defineModel():
        """Creates a sequential neural network with 3 convolutional layers, 2 max-pooling
        layers and 2 dense layers.
        """
        
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten()) # Flatten the 2D output before passing it to the dense layers
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(Dense(10, activation='softmax'))

        # Compile model
        opt = SGD(learning_rate=0.02, momentum=0.8)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	
        return model

def evaluateModel(dataX, dataY, k=5):
        """Creates and evaluates the model using k-fold cross-validation.

        Parameters
        ==========
        dataX : array
            The feature vectors of the training data.

        dataY : array
            The target values of the training data.

        k : int, optional
            The number of folds to use in k-fold cross-validation
        """
        
        scores, histories = list(), list()

        # Prepare cross-validation
        kfold = KFold(k, shuffle=True, random_state=1)

        # Enumerate splits
        for train_ix, test_ix in kfold.split(dataX):
                model = defineModel()
                # Select rows for training and test sets
                trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
                # Fit model
                history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
                # Evaluate model
                _, acc = model.evaluate(testX, testY, verbose=0)
                print('> %.3f' % (acc * 100.0))
                # Stores scores
                scores.append(acc)
                histories.append(history)
        model.save("David_Vella_Zarb.h5")
        return scores, histories

def plotLearningCurves(histories):
        """Plots cross-entropy loss against classification accuracy.

        Parameters
        ==========
        histories : array
            The history that includes the loss and accuracy of each epoch of the model training.
        """
                
        for i in range(len(histories)):
                # Cross-entropy loss
                pyplot.subplot(2, 1, 1)
                pyplot.title('Cross Entropy Loss')
                pyplot.plot(histories[i].history['loss'], color='blue', label='train')
                pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')

                # Classification accuracy
                pyplot.subplot(2, 1, 2)
                pyplot.title('Classification Accuracy')
                pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
                pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')

        pyplot.show()

def summarisePerformance(scores):
        """Summarises the given performance of the model.  Plots mean and standard deviation of the scores
        for each epoch.

        Parameters
        ==========
        scores : array
            The score history for each epoch of training.
        """
	
        print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))

        pyplot.boxplot(scores)
        pyplot.show()

def runTestHarness():
        """Runs the end-to-end test harness for evaluating the model.
        """
        
        trainX, trainY, testX, testY = loadDataset()
        trainX, testX = scalePixels(trainX, testX)
        scores, histories = evaluateModel(trainX, trainY)
        plotLearningCurves(histories)
        summarisePerformance(scores)

runTestHarness()
