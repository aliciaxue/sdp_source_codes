import numpy as np
import time
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

num_of_files = 2130

for n in range(1, num_of_files+1):
	#load the input and output data
	input_name = "/nas7/yfxue/data" + str(n) + ".npy"
	data = np.load(input_name)
	output_name = "../data_process/result/result" + str(n) + ".csv"
	result = np.genfromtxt(output_name, delimiter = ',', dtype = np.float32)

	#pre-process the data
	max_read_length = 148
	max_read_count = 60
	num_of_sites = data.shape[0]

	processed_data = np.zeros(((num_of_sites, max_read_count, 4*(2*max_read_length - 1) + 1)), dtype = np.float32)
	mapq = np.zeros((num_of_sites, max_read_count), dtype = np.uint8)
	baseq = np.zeros(((num_of_sites, max_read_count, 2*max_read_length - 1)), dtype = np.uint8)

	#get mapq
	mapq = data[:,:,-1]%128
	mapq = (mapq.repeat(2*max_read_length-1)).reshape(num_of_sites,max_read_count,2*max_read_length-1)

	#get baseq
	baseq = 2*data[:,:,:2*max_read_length - 1]%64

	#get strand
	processed_data[:,:,-1] = np.floor(data[:,:,-1]/128) 


	#get nucleotides
	k, b = 0.018, -0.08
	processed_data[:,:,:4*(2*max_read_length-1):4] = np.absolute((k*baseq+b)*(k*mapq+b)*(np.sign(baseq-10.1)+1)/2*(np.sign(mapq-10.1)+1)/2*np.floor(data[:,:,:2*max_read_length-1]/64)*(np.floor(data[:,:,:2*max_read_length-1]/64)-1)*(np.floor(data[:,:,:2*max_read_length-1]/64)-2))
	#(k*baseq+b)*(k*mapq+b): equivalent to quality_score_scaling() in variant_calling_v2.py
	#(np.sign(baseq-10.1)+1)/2: check if baseq > 10, otherwise discard the nucleotide; I used 10.1 instead of 10 to eliminate the case when np.sign() = 0
	#(np.sign(mapq-10.1)+1)/2: check if mapq > 10, otherwise discard the nucleotide
	#np.floor(data[:,:,:2*max_read_length-1]/64)*(np.floor(data[:,:,:2*max_read_length-1]/64)-1)*(np.floor(data[:,:,:2*max_read_length-1]/64)-2):
	#the upper two bits of data[:,:,:2*max_read_length-1] were used to represent the nucleotides
	# 0-A, 1-C, 2-G, 3-T
	#the terms are to check whether the two upper bits represent zero (!= 1, 2 and 3)
	processed_data[:,:,1:4*(2*max_read_length-1):4] = np.absolute((k*baseq+b)*(k*mapq+b)*(np.sign(baseq-10.1)+1)/2*(np.sign(mapq-10.1)+1)/2*np.floor(data[:,:,:2*max_read_length-1]/64)*(np.floor(data[:,:,:2*max_read_length-1]/64)-1)*(np.floor(data[:,:,:2*max_read_length-1]/64)-3))
	processed_data[:,:,2:4*(2*max_read_length-1):4] = np.absolute((k*baseq+b)*(k*mapq+b)*(np.sign(baseq-10.1)+1)/2*(np.sign(mapq-10.1)+1)/2*np.floor(data[:,:,:2*max_read_length-1]/64)*(np.floor(data[:,:,:2*max_read_length-1]/64)-2)*(np.floor(data[:,:,:2*max_read_length-1]/64)-3))
	processed_data[:,:,3:4*(2*max_read_length-1):4] = np.absolute((k*baseq+b)*(k*mapq+b)*(np.sign(baseq-10.1)+1)/2*(np.sign(mapq-10.1)+1)/2*(np.floor(data[:,:,:2*max_read_length-1]/64)-1)*(np.floor(data[:,:,:2*max_read_length-1]/64)-2)*(np.floor(data[:,:,:2*max_read_length-1]/64)-3))

	#change the positions of variant site and the last read base
	temp = np.copy(processed_data[:,:,588:592])
	processed_data[:,:,588:592] = processed_data[:,:,1176:1180]
	processed_data[:,:,1176:1180] = temp

	#Divide the data into training and test sets
	size_of_training_set = int(num_of_sites*0.9)
	train_input = processed_data[:size_of_training_set, :, :]
	train_output = result[:size_of_training_set, :]
	test_input = processed_data[size_of_training_set:, :, :]
	test_output = result[size_of_training_set:, :]

	#Network
	if n == 1:

		inputs = Input(shape = (60, 1181), dtype = "float32")
		#Separate the inputs into data and tag & strand
		tag_and_strand = Lambda(lambda x: x[:, :, 1176:])(inputs)
		data = Lambda(lambda x: x[:, :, :1176])(inputs)
		#row-to-row convolution on data
		data_processed = Flatten()(data)
		data_processed = RepeatVector(60)(data_processed)
		data_processed = Reshape((60, 60, 294, 4, 1))(data_processed)
		data_transpose = Permute((2, 1, 3, 4, 5))(data_processed)
		data_combined = concatenate([data_processed, data_transpose]) 
		data_combined = Reshape((1058400, 8))(data_combined)
		data_combined = Conv1D(20, 1, padding='valid', activation='selu')(data_combined)
		data_combined = Conv1D(10, 1, padding='valid', activation='selu')(data_combined)
		data_combined = Conv1D(8, 1, padding='valid', activation='selu')(data_combined)
		#row-invariant average pooling
		data_combined = AveragePooling1D(pool_size=17640, strides=None, padding='valid')(data_combined)
		#concatenate data and tag&strand and perform row-invaraint convolution
		input_combined = concatenate([tag_and_strand, data_combined])
		input_combined = Conv1D(15, 1, padding = 'valid', activation = 'selu')(input_combined)
		input_combined = Conv1D(8, 1, padding = 'valid', activation = 'selu')(input_combined)
		input_combined = Flatten()(input_combined)
		input_combined = Dense(400, activation = 'selu')(input_combined)
		input_combined = Dense(200, activation = 'selu')(input_combined)
		input_combined = Dense(80, activation = 'selu')(input_combined)
		input_combined = Dropout(0.2)(input_combined)
		predictions = Dense(10, activation = 'softmax')(input_combined)

		model = Model(inputs=inputs, outputs=predictions)
		model.compile(optimizer='adam',
		              loss='mean_absolute_error',
		              metrics=['accuracy'])
		model.summary()

	model.fit(train_input, train_output, validation_data = (test_input, test_output), epochs = 100, batch_size = 10, verbose = 1)
	scores = model.evaluate(test_input, test_output, verbose=0)
	print("File " + str(n) + ":")
	print("Baseline Error: %.2f%%" % (100-scores[1]*100))
	model.save("variant_calling_network.h5")