#!/usr/bin/python
import sys 
import re
import numpy as np

def readValuesIntoList(line):
  out = []
  for num in line:
    num = re.split('[][]', num)
    for val in num: 
      if val: out.append(float(val))
  return np.asarray(out)


def parseLogFile(filename):
  # List of tuples of the form (str(tensorName), np.array(tensorValues))
  tensorList = []
  tensorCount = 0 
  with open(filename, 'r') as f:
    foundTensor = False
    for line in f:
      # Process only output tensors
      #if "output" in line and "contents" in line:
      # Process all tensors
      if "contents" in line:
        foundTensor = True
        tensorName = line.split(' contents')[0]
        continue
      if foundTensor:
        tensorCount += 1
        tensorValuesArray = readValuesIntoList(line.split())
        tensorTuple = (tensorName, tensorValuesArray)
        tensorList.append(tensorTuple)
        if tensorCount == 10: # Process only 10 tensors.
          break
    return tensorList


def compareLists(eigenList, mklList, rtol=1e-02, atol=1e-04):
  #TODO: Enable logging
  length = len(eigenList)
  for idx in range(length):
    allclose_res = np.allclose(eigenList[idx][1], mklList[idx][1], rtol, atol, equal_nan=True)
    if allclose_res:
      print eigenList[idx][0], "and", mklList[idx][0], "are equal."
      continue
    print eigenList[idx][0], "and", mklList[idx][0], "are not equal."
    isclose_res = np.isclose(eigenList[idx][1], mklList[idx][1], rtol, atol, equal_nan=True)
    print "Tensor name: " + str(eigenList[idx][0])
    print "Eigen tensor values: " + str(eigenList[idx][1])
    print "MKL tensor values: " + str(mklList[idx][1])
    #print "np.isclose result:", isclose_res

 
def main():
  eigenValuesList = parseLogFile("eigen_log") 
  mklValuesList = parseLogFile("mkl_log")
  assert(len(eigenValuesList) == len(mklValuesList)), "eigenValuesList and mklValuesList do not have the same lengths."
  compareLists(eigenValuesList, mklValuesList)


if __name__ == "__main__":
  main()
