import random 
import numpy as np
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
import time


#Decoding and pre-process the data
def quality_score_scaling(mapq, baseq):
	k, b = 0.018, -0.08 #for linear scaling
	baseq = baseq*2
	return k*min(mapq, baseq) + b

data = np.load("/nas7/yfxue/data1.npy") #Load the processed data file; for convenience, only one data file is used in this code
result = np.genfromtxt("result1.csv", delimiter = ',', dtype = np.float32) #Load the output data file

max_read_length = 148
max_read_count = 60
num_of_sites = data.shape[0]

processed_data = np.zeros(((num_of_sites, max_read_count, 4*(2*max_read_length - 1) + 1)), dtype = np.float32)

for i in range(num_of_sites):
	start_time = time.time()
	for j in range(max_read_count):
		#discard all reads with mapq < 10
		mapq = data[i][j][-1]%128
		if mapq <= 10:
			continue

		strand = int(data[i][j][-1]/128)
		processed_data[i][j][-1] = strand

		for k in range(2*max_read_length - 1):
			baseq = data[i][j][k]%64
			if baseq <= 10:
				continue

			if int(data[i][j][k]/64) == 3: #base is "T"
				processed_data[i][j][4*k] = quality_score_scaling(mapq, baseq)
				
			if int(data[i][j][k]/64) == 2: #base is "G"
				processed_data[i][j][4*k + 1] = quality_score_scaling(mapq, baseq)
				
			if int(data[i][j][k]/64) == 1: #base is "C"
				processed_data[i][j][4*k + 2] = quality_score_scaling(mapq, baseq)
				
			if int(data[i][j][k]/64) == 0: #base is "A"
				processed_data[i][j][4*k + 3] = quality_score_scaling(mapq, baseq)

print("Data processing finished!")

#Divide the data into training and test sets
size_of_training_set = int(num_of_sites*0.9) #use 90% of the data for training, 10$ for validation
train_input = processed_data[:size_of_training_set, :, :]
train_output = result[:size_of_training_set, :]
test_input = processed_data[size_of_training_set:, :, :]
test_output = result[size_of_training_set:, :]

#Network
inputs = Input(shape = (60, 1181), dtype = "float32")
#Separate the inputs into data and tag & strand
tag = Lambda(lambda x: x[:, :, 588:592])(inputs) 
strand = Lambda(lambda x: x[:, :, 1180:])(inputs)
tag_and_strand = concatenate([tag, strand]) 
data_1 = Lambda(lambda x: x[:, :, :588])(inputs) 
data_2 = Lambda(lambda x: x[:, :, 592:1180])(inputs) 
data = concatenate([data_1, data_2])
#row-to-row convolution on data
data = Reshape((60, 294, 4))(data)
data_copy_1 = Lambda(lambda x: x[:, :, 2:, :])(data)
data_copy_2 = Lambda(lambda x: x[:, :, 1:293, :])(data)
data_copy_3 = Lambda(lambda x: x[:, :, 0:292, :])(data)
data_processed = concatenate([data_copy_1, data_copy_2])
data_processed = concatenate([data_processed, data_copy_3])
data_processed = Reshape((60, 292, 4, 3))(data_processed)
data_processed = Permute((1, 4, 3, 2))(data_processed)
data_processed = Flatten()(data_processed)
data_processed = RepeatVector(60)(data_processed)
data_processed = Reshape((60, 60, 292, 4, 3))(data_processed)
data_transpose = Permute((2, 1, 3, 4, 5))(data_processed)
data_combined = concatenate([data_processed, data_transpose]) 
data_combined = Reshape((1051200, 24))(data_combined)
data_combined = Conv1D(30, 1, padding='valid', activation='selu')(data_combined)
data_combined = Conv1D(8, 1, padding='valid', activation='selu')(data_combined)
#row-invariant average pooling
data_combined = AveragePooling1D(pool_size=17520, strides=None, padding='valid')(data_combined)
#concatenate data and tag&strand and perform row-invaraint convolution
input_combined = concatenate([tag_and_strand, data_combined])
input_combined = Conv1D(13, 1, padding = 'valid', activation = 'selu')(input_combined)
input_combined = Conv1D(8, 1, padding = 'valid', activation = 'selu')(input_combined)
input_combined = Flatten()(input_combined)
input_combined = Dense(500, activation = 'selu')(input_combined)
input_combined = Dense(80, activation = 'selu')(input_combined)
input_combined = Dropout(0.2)(input_combined)
predictions = Dense(10, activation = 'softmax')(input_combined)

model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='adam',
              loss='mean_absolute_error',
              metrics=['accuracy'])
model.summary()
model.fit(train_input, train_output, validation_data = (test_input, test_output), epochs = 200, batch_size = 4, verbose = 1)
scores = model.evaluate(test_input, test_output, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
'''