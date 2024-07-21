"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch.nn as nn
import sys
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import datetime
import matplotlib.ticker as mticker
import ipdb
# Don't generate pyc codes
sys.dont_write_bytecode = True

device = 'cuda'

def loss_fn(pred, target):
    # Define individual loss functions for position and orientation
    loss_pos = nn.MSELoss()
    loss_angle = nn.CosineEmbeddingLoss()

    # Compute position loss
    pred_pos, target_pos = pred[:, :3], target[:, :3]
    loss_pos_value = torch.sqrt(loss_pos(pred_pos, target_pos))

    # Compute orientation loss
    pred_quat, target_quat = pred[:, 3:], target[:, 3:]
    # Normalize quaternion predictions and targets
    pred_quat = pred_quat / torch.norm(pred_quat, dim=1, keepdim=True)
    target_quat = target_quat / torch.norm(target_quat, dim=1, keepdim=True)
    # Cosine embedding loss takes normalized inputs
    loss_angle_value = loss_angle(pred_quat, target_quat, torch.ones(target_quat.shape[0], device=target_quat.device))

    # Weighted combination of position and orientation losses
    loss = 0.4 * loss_pos_value + 0.6 * loss_angle_value

    return loss


class Visual_encoder(nn.Module):
    def __init__(self):
        super(Visual_encoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.lstm = nn.LSTM(512, 256, 1, batch_first=True)
        self.linear = nn.Linear(256, 6)


    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)   
        x = self.conv3(x)
        x = self.conv3_1(x)
        x = self.conv4(x)

        x = x.view(x.size(0)[0],x.size(0)[2] * x.size(0)[3], x.size(0)[1])
        x, (_, _) = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x

    
    def validation_step(self, Img_test_batch, pose_test_batch):
        
        prediction = self.forward(Img_test_batch)
        loss_val = loss_fn(prediction, pose_test_batch)
        return loss_val


class Visual_encoder2(nn.Module):
    def __init__(self):
        super(Visual_encoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2) 
        )
        self.conv2 = nn.Sequential( 
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2)
        )

        self.conv3_1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2)
        )

        self.lstm = nn.LSTM(512, 256, 1, batch_first=True)
        self.linear = nn.Linear(256, 6)


        
class Inertial_encoder2(nn.Module):
    def __init__(self):
        super(Inertial_encoder, self).__init__()

        # Define convolutional layers
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.1)
        )

        # Define LSTM layer
        self.lstm = nn.LSTM(256, 64, 2, batch_first=True)

        # Define linear layer
        self.linear = nn.Linear(64, 6)


class Inertial_encoder(nn.Module):
    def __init__(self):
        super(Inertial_encoder, self).__init__()

        # Define convolutional layers
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.1),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.1),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.1),
        )

        # Define LSTM layer
        self.lstm = nn.LSTM(128, 64, 2, batch_first=True, dropout=0.1)

        # Define linear layers
        self.linear1 = nn.Linear(64, 32)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(32, 6)

    def forward(self, x):
        """
        Forward pass of the model
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output tensor
        """
        # Pass through convolutional layers
        x = self.encoder_conv(x)
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Pass through LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Get last output of LSTM
        lstm_out = lstm_out[:, -1, :]
        
        # Pass through linear layers with dropout
        x = self.linear1(lstm_out)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.linear2(x)
        
        return output


    def validation_step(self, IMU_test_batch, pose_test_batch):
        
        prediction = self.forward(IMU_test_batch)
        loss_val = loss_fn(prediction, pose_test_batch)
        return loss_val
    
class Visual_Inertial_encoder(nn.Module):
    def __init__(self):
        super(Visual_Inertial_encoder, self).__init__()

        # Instantiate visual encoder and inertial encoder
        self.visual_encoder = Visual_encoder().to(device)
        self.inertial_encoder = Inertial_encoder().to(device)

        # Define layers for fusion
        self.conv1d = nn.Conv1d(2, 64, kernel_size=3)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.linear = nn.Linear(256, 6)

    def forward(self, img_batch, imu_batch):
        """
        Forward pass of the model
        
        Args:
            img_batch (torch.Tensor): Batch of image data
            imu_batch (torch.Tensor): Batch of inertial data
        
        Returns:
            torch.Tensor: Predicted pose
        """
        # Obtain pose predictions from visual and inertial encoders
        img_pose = self.visual_encoder(img_batch)
        inertial_pose = self.inertial_encoder(imu_batch)
        
        # Concatenate pose predictions along the channel dimension
        x = torch.cat((img_pose.unsqueeze(1), inertial_pose.unsqueeze(1)), dim=1).to(device)
        
        # Pass through fusion layer
        x = self.conv1d(x)
        x = self.relu(x)
        
        # Flatten and pass through linear layer
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        
        return out
    
    def validation_step(self, img_batch, imu_batch, pose_batch):
        """
        Validation step
        
        Args:
            img_batch (torch.Tensor): Batch of image data
            imu_batch (torch.Tensor): Batch of inertial data
            pose_batch (torch.Tensor): Ground truth pose
        
        Returns:
            torch.Tensor: Validation loss
        """
        # Perform forward pass
        prediction = self.forward(img_batch, imu_batch)
        
        # Calculate validation loss
        loss_val = loss_fn(prediction, pose_batch)
        
        return loss_val
