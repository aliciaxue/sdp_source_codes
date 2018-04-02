#fetch data at the variant positions

import numpy as np
import pandas as pd
import pysam

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

#import bam file
samfile = pysam.AlignmentFile("HG002.bam","rb")

#read the variant indice from the csvs
num_of_files = 2130

for n in range(1, num_of_files + 1):

	filename = str(n) + ".csv"
	df = pd.read_csv(filename, sep = ',', header = None)
	size_of_file = df.values.shape[0]

	for m in range(size_of_file):

		exp_chr = int(df.values[m][0])
		exp_pos = int(df.values[m][1])

		min_quality_score = 255
		min_read_index = 255
		read_count = 0
		max_read_count = 100
		max_read_length = 148

		data = np.zeros((max_read_count, (2*max_read_length-1)*5+2), dtype = np.uint8)
		quality_score_array = np.zeros(max_read_count, dtype = np.uint8)

		for read in samfile.fetch(str(exp_chr), exp_pos, exp_pos + 1):
			new_read_index = read_count

			#get quality score and update the minimum
			quality_score = read.mapping_quality
			if quality_score == None:
				continue

			if quality_score <= min_quality_score and read_count >= max_read_count:
				continue
			if quality_score > min_quality_score and read_count >= max_read_count:
				quality_score_array[min_read_index] = quality_score
				new_read_index = min_read_index  #replace the read with the lowest quality score
				min_read_index = np.argmin(quality_score_array)
				min_quality_score = min(quality_score_array)
				data[new_read_index][-2] = quality_score
			else:
				quality_score_array[new_read_index] = quality_score
				data[new_read_index][-2] = quality_score
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
			data[new_read_index][-1] = int(read.mate_is_reverse)

			#fetch the sequence & qualities
			seq = read.query_alignment_sequence
			qual = read.query_alignment_qualities
			if seq == None || qual == None:
				continue

			cigarstring = read.cigarstring
			if cigarstring == None:
				continue
			seq, qual = cigarstring_process(cigarstring, seq, qual)

			if seq == None: #if has error, discard the read
				continue
	
			for i in range(end_index - start_index):
				if seq[i] == 'T':
					data[new_read_index][5*ref_start_index + 5*i] = 1
				if seq[i] == 'G':
					data[new_read_index][5*ref_start_index + 5*i + 1] = 1
				if seq[i] == 'C':
					data[new_read_index][5*ref_start_index + 5*i + 2] = 1
				if seq[i] == 'A':
					data[new_read_index][5*ref_start_index + 5*i + 3] = 1
				data[new_read_index][5*ref_start_index + 5*i + 4] = qual[i]

			read_count += 1

		savefilename = "data" + str(n) + ".npy"
		f = open(savefilename, 'ab')
		np.savetxt(f, data)
		f.close()

samfile.close()
