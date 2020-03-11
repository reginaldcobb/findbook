#!/usr/bin/python
import collections
import os

def remove_newlines(fname):
    flist = open(fname).readlines()
    return [s.rstrip('\n') for s in flist]

wordcount = collections.Counter()
# with open('text_book4png_3200_3200_0.65_0.5_1_10_02_26_01_24_54.txt') as f:
#     for line in f:
#         wordcount.update(line.split())
# # print(wordcount)
# file1 = open("/home/rcobb/Downloads/text_filename.txt","a") 
# file1.write(str(wordcount))
# file1.close() 

# for k,v in wordcount.items():
#     print (k, v)

file1 = open("/home/rcobb/Downloads/text_filename.txt","a") 
with os.scandir('./') as entries:
    for entry in entries:
        wordcount.clear()
        # if str.find(str(entry.name))
        if entry.name.find(".txt") > -1:
            # print(entry.name)
            with open(entry.name) as f:
                for line in f:
                    wordcount.update(line.split())
            # wordcount['filename'] = entry.name
            # print(wordcount)
            # print(entry.name)
            # file1.write(entry.name)
            # file1.write(str(wordcount))
                # file1.write("%%%")
                file1.write("\n")
                file1.write(entry.name)
                file1.write(";")
                print(entry.name)
                for k,v in wordcount.items():
                    # print (k, v)
                    for i in range (1,(int(v)+1)):
                        # print (k)
                        file1.write(k.upper())
                        file1.write(";")
                    # file1.write("\n")
file1.close() 

# remove_newlines("/home/rcobb/Downloads/text_filename.txt")
# print ("Wordcoutn =", wordcount)
# print ("opening to remove CR")
# flist = open("/home/rcobb/Downloads/text_filename.txt").readlines()
# for s in flist:
#     s.rstrip('\n')
# print ("finished removing CR")
# flist.close() 
