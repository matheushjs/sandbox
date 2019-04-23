import sys, os
import re

if len(sys.argv) != 2:
    print("Usage: {} [arquivo]".format(sys.argv[0]))
    sys.exit(0)

fp = open(sys.argv[1])
content = fp.read()
content = content.replace("\n", "")

tabLevel = 0

i = 0
while i != len(content):
    if content[i:i+2] == "<!" or content[i:i+5] == "<meta":
        while content[i] != ">":
            print(content[i], sep="", end="")
            i += 1
        i += 1
        print(">\n", sep="", end="")
        print("  " * tabLevel, sep="", end="")
        while content[i] == " ":
            i += 1
    elif content[i:i+2] == "</":
        tabLevel -= 1
        print("\n", sep="", end="")
        print("  " * tabLevel, sep="", end="")
        while content[i] != ">":
            print(content[i], sep="", end="")
            i += 1
        i += 1
        print(">\n", sep="", end="")
        print("  " * tabLevel, sep="", end="")
        while i < len(content) and content[i] == " ":
            i += 1
    elif content[i] == "<":
        while content[i] != ">":
            print(content[i], sep="", end="")
            i += 1
        i += 1
        tabLevel += 1
        print(">\n", sep="", end="")
        print("  " * tabLevel, sep="", end="")
        while content[i] == " ":
            i += 1
    else:
        print(content[i], sep="", end="")
        i += 1

fp.close()
