import random
import os

filename="data/big_data.txt"
f=open(filename,"w")
for i in range(10000):
    my_str=""
    if i%1000==0:
        print(str(i/100)+" % progress...")
    for i in range(30000):
        my_str+=random.choice(["apple","banana","orange","grape","pear","peach"])+" "
    print(my_str,file=f)
f.close()
file_size = os.path.getsize(filename)
print("File Size is :", file_size/1024/1024, "MB")


# import sys because we need to read and write data to STDIN and STDOUT
import sys

# reading entire line from STDIN (standard input)
for line in sys.stdin:
	# to remove leading and trailing whitespace
	line = line.strip()
	# split the line into words
	words = line.split()
	
	# we are looping over the words array and printing the word
	# with the count of 1 to the STDOUT
	for word in words:
		# write the results to STDOUT (standard output);
		# what we output here will be the input for the
		# Reduce step, i.e. the input for reducer.py
		print (word+"\t 1")