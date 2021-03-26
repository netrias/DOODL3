from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import initializers


def baseline_mnist_model(name, num_classes=2):
    initializer = initializers.RandomUniform(minval=-2, maxval=2, seed=None)
    model = Sequential(name=name)
    model.add(Conv2D(2, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1), kernel_initializer=initializer))
    model.add(MaxPooling2D(pool_size=(7, 7)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializer))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def mnist_2_layers(name, num_classes=2):
    initializer = initializers.RandomUniform(minval=-2, maxval=2, seed=None)
    model = Sequential(name=name)
    model.add(Conv2D(2, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1), kernel_initializer=initializer))
    model.add(Conv2D(2, kernel_size=(3, 3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(7, 7)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializer))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def mnist_3_layers(name, num_classes=2):
    initializer = initializers.RandomUniform(minval=-2, maxval=2, seed=None)
    model = Sequential(name=name)
    model.add(Conv2D(2, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1), kernel_initializer=initializer))
    model.add(Conv2D(2, kernel_size=(3, 3),
                     activation='relu'))
    model.add(Conv2D(2, kernel_size=(3, 3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(7, 7)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializer))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def mnist_4_layers(name, num_classes=2):
    initializer = initializers.RandomUniform(minval=-2, maxval=2, seed=None)
    model = Sequential(name=name)
    model.add(Conv2D(2, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1), kernel_initializer=initializer))
    model.add(Conv2D(2, kernel_size=(3, 3),
                     activation='relu'))
    model.add(Conv2D(2, kernel_size=(3, 3),
                     activation='relu'))
    model.add(Conv2D(2, kernel_size=(3, 3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(7, 7)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializer))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def baseline_iris_model(name):
    initializer = initializers.RandomUniform(minval=-2, maxval=2, seed=None)
    model = Sequential(name=name)
    model.add(Dense(3, input_dim=2, activation='softmax', kernel_initializer=initializer))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def iris_2_layers(name):
    initializer = initializers.RandomUniform(minval=-2, maxval=2, seed=None)
    model = Sequential(name=name)
    model.add(Dense(3, input_dim=2, activation='softmax', kernel_initializer=initializer))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def iris_3_layers(name):
    initializer = initializers.RandomUniform(minval=-2, maxval=2, seed=None)
    model = Sequential(name=name)
    model.add(Dense(3, input_dim=2, activation='softmax', kernel_initializer=initializer))
    model.add(Dense(3, activation='softmax'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def iris_4_layers(name):
    initializer = initializers.RandomUniform(minval=-2, maxval=2, seed=None)
    model = Sequential(name=name)
    model.add(Dense(3, input_dim=2, activation='softmax', kernel_initializer=initializer))
    model.add(Dense(3, activation='softmax'))
    model.add(Dense(3, activation='softmax'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def iris_3_nodes(name):
    initializer = initializers.RandomUniform(minval=-2, maxval=2, seed=None)
    model = Sequential(name=name)
    model.add(Dense(3, input_dim=2, activation='softmax', kernel_initializer=initializer))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def iris_4_nodes(name):
    initializer = initializers.RandomUniform(minval=-2, maxval=2, seed=None)
    model = Sequential(name=name)
    model.add(Dense(4, input_dim=2, activation='softmax', kernel_initializer=initializer))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def iris_5_nodes(name):
    initializer = initializers.RandomUniform(minval=-2, maxval=2, seed=None)
    model = Sequential(name=name)
    model.add(Dense(5, input_dim=2, activation='softmax', kernel_initializer=initializer))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def iris_6_nodes(name):
    initializer = initializers.RandomUniform(minval=-2, maxval=2, seed=None)
    model = Sequential(name=name)
    model.add(Dense(6, input_dim=2, activation='softmax', kernel_initializer=initializer))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
