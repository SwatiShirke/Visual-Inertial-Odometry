"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import os
import cv2
import numpy as np
import random
import skimage
import PIL
import sys
import csv
import torch

# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupAll(BasePath, CheckPointPath):
    """
    Inputs:
    BasePath is the base path where Images are saved without "/" at the end
    CheckPointPath - Path to save checkpoints/model
    Outputs:
    DirNamesTrain - Variable with Subfolder paths to train files
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    ImageSize - Size of the image
    NumTrainSamples - length(Train)
    NumTestRunsPerEpoch - Number of passes of Val data with MiniBatchSize
    Trainabels - Labels corresponding to Train
    NumClasses - Number of classes
    """
    # Setup DirNames
    DirNamesTrain = SetupDirNames("Phase2/Data/Train_synthetic")

    # Read and Setup Labels
    LabelsPathTrain = "Phase2/Data/Train_synthetic/H4.csv"
    TrainLabels = ReadLabels(LabelsPathTrain)

    # If CheckPointPath doesn't exist make the path
    if not (os.path.isdir(CheckPointPath)):
        os.makedirs(CheckPointPath)

    # Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    SaveCheckPoint = 100
    # Number of passes of Val data with MiniBatchSize
    NumTestRunsPerEpoch = 5

    # Image Input Shape
    ImageSize = [128, 128, 1]
    NumTrainSamples = len(DirNamesTrain)

    # Number of classes
    NumClasses = 10

    return (
        DirNamesTrain,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        TrainLabels,
        NumClasses,
    )

def ReadLabels(LabelsPathTrain):
    if not (os.path.isfile(LabelsPathTrain)):
        print("ERROR: Train Labels do not exist in " + LabelsPathTrain)
        sys.exit()
    labels_images = {}

    with open(LabelsPathTrain, 'r') as labels_file:
        csv_reader = csv.reader(labels_file)
        for row in csv_reader:
            image_name = row[0]
            labels = [float(label) for label in row[1:]]
            labels_tensor = torch.tensor(labels, dtype=torch.float32)
            labels_images[image_name] = labels_tensor

    return labels_images

def SetupDirNames(BasePath):
    """
    Inputs:
    BasePath is the base path where Images are saved without "/" at the end
    Outputs:
    Writes a file ./TxtFiles/DirNames.txt with full path to all image files without extension
    """
    # DirNamesTrain = ReadDirNames("Phase2/Code/TxtFiles/DirNamesTrain.txt")
    DirNamesTrain = os.listdir(BasePath)

    return DirNamesTrain


def ReadDirNames(ReadPath):
    """
    Inputs:
    ReadPath is the path of the file you want to read
    Outputs:
    DirNames is the data loaded from ./TxtFiles/DirNames.txt which has full path to all image files without extension
    """
    # Read text files
    DirNames = open(ReadPath, "r")
    DirNames = DirNames.read()
    DirNames = DirNames.split()
    return DirNames