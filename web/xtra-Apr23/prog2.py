import sys, os
import re

if len(sys.argv) != 2:
    print("Usage: {} [arquivo]".format(sys.argv[0]))
    sys.exit(0)

fp = open(sys.argv[1])
content = fp.read()

def printContent(content, idx):
    # First find which element it is
    idx2 = idx+1
    while content[idx2] != ">" and content[idx2] != " ":
        if content[idx2] == "/":
            return
        idx2 += 1

    substr = content[idx+1:idx2]
    print(substr)

for i in range(len(content)):
    if content[i] == "<":
        printContent(content, i)

