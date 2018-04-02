#fetch data at the variant positions

import numpy as np
import pandas as pd
import pysam
import queue
import sys

read_queue = queue.Queue() #read_queue stores the reads to be processed at the next variant site
current_index = 0 
current_file = 1 
df = None #stores variant sites
default_read_length = 148

def cigarstring_process(cigarstring, seq, qual):
	last_index = 0
	seq_index = 0
	new_seq_index = 0
	new_seq = []
	new_qual = []
	for i in range(len(cigarstring)):
		if cigarstring[i] >= 'A' and cigarstring[i] <= 'Z':
			#print(i)
			#print(last_index)
			letter = cigarstring[i]
			number = int(cigarstring[last_index:i])
			if letter == 'M': #match, copy seq and qual
				new_seq[new_seq_index:new_seq_index + number] = seq[seq_index:seq_index + number]
				new_qual[new_seq_index:new_seq_index + number] = qual[seq_index:seq_index + number]
				new_seq_index += number
				seq_index += number
			elif letter == 'D':  #deletion; inserting zeros to seq and qual
				for j in range(number):
					new_seq.append("O")
					new_qual.append(0)
				new_seq_index += number
			elif letter == 'I':  #insertion; do not copy the portion to seq and qual
				seq_index += number
			elif letter == 'S':  #softcut; do nothing
				dummy = 1
			else:  #unknown command; report an error 
				print("ERROR: don't no how to handle " + str(number) + str(letter))
				return None, None
			last_index = i + 1
	return new_seq, new_qual

#check if a read is useful
def check_read(start_index, read_chr, exp_pos, exp_chr):
	global default_read_length
	if read_chr == exp_chr and start_index <= exp_pos and start_index + default_read_length > exp_pos: #read length is 148
		return True
	return False


def process_a_read(read, read_chr):
	global read_queue, current_file, current_index
	exp_chr = int(df.values[current_index][0])
	exp_pos = int(df.values[current_index][1])
	start_index = read.reference_start
	if start_index == None or read_chr == None:
		return
	#print(check_read(start_index, read_chr, exp_pos, exp_chr))
	if check_read(start_index, read_chr, exp_pos, exp_chr) == True:
		read_queue.put(read)
		#print("put the read in the queue")
	elif start_index > exp_pos or read_chr > exp_chr:
		process_relevant_sites(read, read_chr, 0)


def process_relevant_sites(next_read, next_read_chr, flag):
	global read_queue, current_index, current_file, df, size_of_file, max_read_count, max_read_length
	global data

	read_queue_temp = queue.Queue()

	while True:

		min_quality_score = 255
		min_read_index = 255
		read_count = 0
		
		exp_chr = int(df.values[current_index][0])
		exp_pos = int(df.values[current_index][1])
#		print("current index: " + str(current_index))

		read_queue_temp.queue.clear()

		quality_score_array = np.zeros(max_read_count, dtype = np.uint8)

		#processed the reads
		#start_time = time.time()
		while(read_queue.empty() == False):
			read = read_queue.get()
			read_queue_temp.put(read)

			new_read_index = read_count

			#put the data in the numpy file
			#get quality score and update the minimum
			quality_score = int(read.mapping_quality)
			if quality_score == None:
				continue
			if quality_score <= min_quality_score and read_count >= max_read_count:
				continue
			if quality_score > min_quality_score and read_count >= max_read_count:
				quality_score_array[min_read_index] = quality_score
				new_read_index = min_read_index  #replace the read with the lowest quality score
				min_read_index = np.argmin(quality_score_array)
				min_quality_score = min(quality_score_array)
				data[current_index][new_read_index][-1] = quality_score #an error here, should be [:, :, -1]
			else:
				quality_score_array[new_read_index] = quality_score
				data[current_index][new_read_index][-1] = quality_score #an error here, should be [:, :, -1]
				if quality_score < min_quality_score:
					min_quality_score = quality_score
					min_read_index = new_read_index 

			#get start and end point
			start_index = read.reference_start
			if start_index == None:
				continue
			ref_start_index = (start_index - exp_pos) + max_read_length - 1

			end_index = read.reference_end
			if end_index == None:
				continue
			if end_index > exp_pos + max_read_length:
				end_index = exp_pos + max_read_length

			#get strand
			#use the last byte in per line for strand & mapping quality
			#First bit indicates the strand, last 7 bits the mapping quality
			data[current_index][new_read_index][-1] += int(read.mate_is_reverse)*128
			
			#fetch the sequence & qualities
			seq = read.query_alignment_sequence
			qual = read.query_alignment_qualities
			if seq == None or qual == None:
				continue
			cigarstring = read.cigarstring
			if cigarstring == None:
				continue

			if cigarstring != "148M": #if the read has insertion, deletion or softcut
				seq, qual = cigarstring_process(cigarstring, seq, qual)

			if seq == None: #if has error, discard the read
				continue
	
			#Use a byte to represent a nucleotide and the corresponding quality score
			#first two bits indicate the nucleotide: A-00, C-01, G-10, T-11
			#last six bits indicate the quality score, which < 63
			for i in range(end_index - start_index):
				if seq[i] == 'T':
					data[current_index][new_read_index][ref_start_index + i] += 192
				elif seq[i] == 'G':
					data[current_index][new_read_index][ref_start_index + i] += 128
				elif seq[i] == 'C':
					data[current_index][new_read_index][ref_start_index + i] += 64
				elif seq[i] == 'A':
					data[current_index][new_read_index][ref_start_index + i] += 0
				#else:
				#	data[current_index][new_read_index][ref_start_index + i] += 63 #use 00111111 to indicate a null nucleotide
				data[current_index][new_read_index][ref_start_index + i] += qual[i]

			read_count += 1

		current_index += 1

		if current_index == size_of_file:
			return

		#update read_queue
		next_chr = int(df.values[current_index][0])
		next_pos = int(df.values[current_index][1])

		read_queue.queue.clear()
		while(read_queue_temp.empty() == False):
			read = read_queue_temp.get()
			start_index = read.reference_start
			if check_read(start_index, exp_chr, next_pos, next_chr) == True:
				read_queue.put(read)

		if flag == 1:
			continue

		start_index = next_read.reference_start
		if check_read(start_index, next_read_chr, next_pos, next_chr) == True:
			read_queue.put(next_read)
			return
		if start_index + default_read_length <= next_pos or next_chr > next_read_chr:
			return


#import bam file
samfile = pysam.AlignmentFile("/dev/shm/data/HG002.bam","rb")

#read the variant indice from the csvs
num_of_files = 2130
num_of_copies = 24 #run 24 processes

program_copy = int(sys.argv[1])

current_file = program_copy


while current_file <= num_of_files:

	filename = "/dev/shm/" + str(current_file) + ".csv"
	df = pd.read_csv(filename, sep = ',', header = None)
	size_of_file = df.values.shape[0]

	current_index = 0

	start_pos = int(df.values[0][1])
	end_pos = int(df.values[-1][1])
	start_chr = int(df.values[0][0])
	end_chr = int(df.values[-1][0])

	read_queue.queue.clear()

	count = 0

	savefilename = "/raid/yfxue/data" + str(current_file) + ".npy"
	f = open(savefilename, 'ab')

	max_read_count = 60
	max_read_length = 148

	data = np.zeros((size_of_file,max_read_count, 2*max_read_length), dtype = np.uint8)

	#if the indices in the file are all from the same chromosome
	if start_chr == end_chr:
		for read in samfile.fetch(str(start_chr), start_pos, end_pos + 1):
			process_a_read(read, start_chr)
	else: #if from different chrs
		for read in samfile.fetch(str(start_chr), start_pos,):
			process_a_read(read, start_chr)
		for read in samfile.fetch(str(end_chr), 1, end_pos + 1):
			process_a_read(read, end_chr)

	process_relevant_sites(None, None, 1)

	np.save(savefilename, data)

	f.close()

	total_time = time.time() - start_time

	current_file += num_of_copies

samfile.close()
