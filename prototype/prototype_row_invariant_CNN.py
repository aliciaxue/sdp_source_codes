import random 
import numpy as np, numpy.random
from keras.layers import Dense
from keras.layers.convolutional import Conv1D
from keras.layers import Flatten
from keras.layers import Permute
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers import concatenate
from keras.models import Model
from keras.layers import Input
from keras.layers import Lambda
from keras.layers.core import RepeatVector
from keras.layers.pooling import AveragePooling1D

size = 1

#Generate Data
np.random.seed(3)
a = np.zeros((((size,30,21,4))))  # 30 lines, 1 tag + 20 elements per line
b = np.zeros((size,4)) # output as 4 percentages
p_of_one_hot = 0.99
for n in range(size):
    num_of_groups = random.randrange(1,20)
    raw_numbers = np.random.dirichlet(np.ones(num_of_groups),size=1)
    num_of_equal_elements = np.zeros(num_of_groups)
    for i in range(num_of_groups-1):
        num_of_equal_elements[i] = int(raw_numbers[0][i]*30)
    num_of_equal_elements[num_of_groups-1] = 30-sum(num_of_equal_elements)
    index = 0
    count = np.zeros(4)
    for j in range(num_of_groups):
        if int(num_of_equal_elements[j]) > 0:
            new_row = np.zeros((((1,1,20,4))))
            for m in range(20):  #data should be the same
                if random.random() <= p_of_one_hot:
                    x = random.random()*4
                    if x < 4:
                        new_row[0][0][m][int(x)] = 1
                    else:
                        new_row[0][0][m][3] = 1
        for k in range(int(num_of_equal_elements[j])):
            a[n][index][1:][:] = new_row
            #generate a new tag
            new_tag = np.zeros((((1,1,1,4))))
            x = -1
            if random.random() <= p_of_one_hot:
                x = random.random()*4
                if x < 4:
                    new_tag[0][0][0][int(x)] = 1
                else:
                    new_tag[0][0][0][3] = 1
                    x = 3
            a[n][index][0][:] = new_tag
            if x >= 0:
                count[int(x)] += num_of_equal_elements[j]
            index += 1
    sum_of_count = sum(count)
    for l in range(4):
        b[n][l] = count[l]/sum_of_count
    np.random.shuffle(a[n])

train_size = int(size*0.7)
train_input = a[:train_size, :, :, :]
train_output = b[:train_size, :]
test_input = a[train_size:, :, :, :]
test_output = b[train_size:, :]


def output_label_lambda(input_shape):
    return (input_shape[0], input_shape[1], 1, input_shape[3])

def output_data_lambda(input_shape):
    return (input_shape[0], input_shape[1], 20, input_shape[3])

def output_data_slide_lambda(input_shape):
    return (input_shape[0], input_shape[1], 18, input_shape[3])

inputs = Input(shape = (30, 21, 4,))
x_label = Lambda(lambda x: x[:,:,:1,:], output_shape = output_label_lambda)(inputs)
x_data = Lambda(lambda x: x[:,:,1:,:], output_shape = output_data_lambda)(inputs)
x_data_0 = Lambda(lambda x: x[:,:,2:,:], output_shape = output_data_slide_lambda)(x_data)
x_data_1 = Lambda(lambda x: x[:,:,1:19,:], output_shape = output_data_slide_lambda)(x_data)
x_data_2 = Lambda(lambda x: x[:,:,0:18,:], output_shape = output_data_slide_lambda)(x_data)
x_data_processed = concatenate([x_data_0, x_data_1])
x_data_processed = concatenate([x_data_processed, x_data_2])
x_data_processed = Reshape((30,18,4,3))(x_data_processed)
x_data_processed = Permute((1,4,3,2))(x_data_processed)
x_data_processed = Flatten()(x_data_processed) 
x_data_processed = RepeatVector(30)(x_data_processed)
x_data_processed = Reshape((30,30,18,4,3))(x_data_processed)
x_data_transpose = Permute((2,1,3,4,5))(x_data_processed)
x_data_combined = concatenate([x_data_processed,x_data_transpose])
x_data_combined = Reshape((16200,24))(x_data_combined)
x_data_combined = Conv1D(30, 1, padding='valid', activation='selu')(x_data_combined)
x_data_combined = Conv1D(8, 1, padding='valid', activation='selu')(x_data_combined)
x_data_combined = AveragePooling1D(pool_size=540, strides=None, padding='valid')(x_data_combined)
x_label = Reshape((30,4))(x_label)
x_combined = concatenate([x_label, x_data_combined])
x_combined = Conv1D(12, 1, padding = 'valid', activation = 'selu')(x_combined)
x_combined = Conv1D(8, 1, padding = 'valid', activation = 'selu')(x_combined)
x_combined = Flatten()(x_combined)
x_combined = Dense(120, activation = 'selu')(x_combined)
x_combined = Dropout(0.2)(x_combined)
predictions = Dense(4, activation = 'softmax')(x_combined)


model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='adam',
              loss='mean_absolute_error',
              metrics=['accuracy'])
model.summary()
model.fit(train_input, train_output, validation_data = (test_input, test_output), epochs = 200, batch_size = 32, verbose = 1)
scores = model.evaluate(test_input, test_output, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))