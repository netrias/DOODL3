from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten


def baseline_mnist_model(name, kernel_initializer, num_classes=2):
    model = Sequential(name=name)
    model.add(Conv2D(2, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1), kernel_initializer=kernel_initializer))
    model.add(MaxPooling2D(pool_size=(7, 7)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=kernel_initializer))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def baseline_iris_model():
    pass
