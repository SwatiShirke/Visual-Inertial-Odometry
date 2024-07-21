import torch
from torch.optim import lr_scheduler
import os
import cv2
import numpy as np
import ipdb
import csv 
import argparse
import matplotlib.pyplot as plt
from Misc.MiscUtils import FindLatestModel
from Network import LossFn, Inertial_encoder
import scipy.io
import transforms3d.euler as euler

device = 'cuda'

def GenerateBatch(base_path, MiniBatchSize, start_index, IMU_data, pose_data):
    """
    Base path: path to data folder which contains Imgs, IMU data and ground truth        
    """   
      
    IMU_test_batch = []
    pose_test_batch = []
    
    ImageNum = 0
    while ImageNum < MiniBatchSize and  start_index < 500:
        RandIndex = start_index  
         
        IMU_batch = torch.from_numpy( IMU_data[RandIndex * 10: RandIndex * 10 + 10])
        pose_batch = torch.from_numpy( pose_data[RandIndex * 10])
        
        IMU_test_batch.append(IMU_batch )
        pose_test_batch.append(pose_batch)  
        start_index +=1  
        ImageNum += 1

    IMU_test_batch = torch.stack(IMU_test_batch) if len(IMU_test_batch) !=0 else None
    pose_test_batch =  torch.stack(pose_test_batch) if len(pose_test_batch) !=0 else None

    return IMU_test_batch.to(device), pose_test_batch.to(device), start_index

def read_csv(file_path):
    with open(file_path, mode='r') as file:
    # Create a CSV reader object
        reader = csv.reader(file)
        file_data = []
        row_count = 0
        for data in reader:
            float_row = [float(value) for value in data]
            
            file_data.append(float_row)
    file_data = np.array(file_data)
    return file_data

def plot_and_save_metrics(loss_vs_iteration, CheckPointPath, title, filename):

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(loss_vs_iteration) + 1), loss_vs_iteration, label='Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'{title} Loss vs Iteration')
    plt.legend()

    save_path = os.path.join(CheckPointPath, filename)
    plt.savefig(save_path)
    print(f"Figure saved at: {save_path}")
    plt.show()

def TestOperation(
        MiniBatchSize,
        CheckPointPath,
        LatestFile,
        BasePath
    ):
    # Predict output with forward pass
    model = Inertial_encoder().to(device)

    if LatestFile is not None:
        full_path = "Checkpoints_inertial/" + LatestFile + ".ckpt"
        CheckPoint = torch.load(full_path)
        #CheckPoint = torch.load(LatestFile + ".ckpt")
        # Extract only numbers from the name
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        print("New model initialized....")
   
    names = os.listdir(BasePath)

    for name in names:
        Loss_pose_test = []
        pose_train_predicted_list = []
        initial_combined_data = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        Basepath = os.path.join(BasePath,name)
        IMU_path = f"{Basepath}/test2/IMU_data_file.csv"
        pose_path = f"{Basepath}/test2/Pose_data.csv"
        IMU_data = read_csv(IMU_path)
        pose_data = read_csv(pose_path)
        states_data= f"{Basepath}/test2/states.mat"
        mat = scipy.io.loadmat(states_data)
        time_data = mat['time'][0][:-2]
        time_10 = time_data[::10]
        gt = mat['state'][:-2]
        gt_10 = gt[::10]
        gt_position = gt_10[:, :3]
        gt_quat = gt_10[:, 6:10]
        combined_data = np.hstack((gt_position, gt_quat))
        gt_file = np.column_stack((time_10, combined_data))
        initial_entry = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        gt_data = np.vstack((initial_entry, gt_file))
        # gt_file = np.diff(gt_data, axis=0)
        output_file_path_gt = f"{name}_pose_gt_abs_inertial.txt"
        with open(output_file_path_gt, 'w') as f:
            for gt_pose in gt_file:
                f.write(','.join(map(str, gt_pose.flatten())) + '\n')
        
        print(f"{name} Testing Started")
        start_index = 0
        for i in range(0, 500):

            IMU_test_batch, pose_test_batch, start_index = GenerateBatch(Basepath, MiniBatchSize, start_index, IMU_data, pose_data)

            if IMU_test_batch is not None and pose_test_batch is not None:
                model.eval()
                with torch.no_grad():
                    
                    pose_train_predicted = model(IMU_test_batch.float()).float()
                    pose_train_predicted = pose_train_predicted.to(device)
                    rpy = pose_train_predicted[:, 3:].cpu().numpy()
                    quats = euler.euler2quat(rpy[:, 0], rpy[:, 1], rpy[:, 2])
                    combined_data = np.hstack((time_10[i], pose_train_predicted[:, :3].cpu().numpy()[0], quats))
                    # relative_combined_data = combined_data - initial_combined_data
                    Loss_pose_train = LossFn(pose_train_predicted, pose_test_batch.float()).float()
                    Loss_pose_train = Loss_pose_train.to(device)
                    Loss_pose_test.append(Loss_pose_train.detach().cpu().numpy())
                    pose_train_predicted_list.append(combined_data)
                    # pose_train_predicted_list.append(relative_combined_data)
                    # initial_combined_data = relative_combined_data

        print(f"{name} Testing Ended")
        output_file_path = f"{name}_pose_test_predicted_abs_inertial.txt"
        with open(output_file_path, 'w') as f:
            for predicted_pose in pose_train_predicted_list:
                f.write(','.join(map(str, predicted_pose.flatten())) + '\n')

        plot_and_save_metrics(Loss_pose_test, CheckPointPath, 'Testing', f'{name}_test_metrics.png')


def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default="./Data",
        help="Base path of images, Default:../Data",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="Checkpoints_inertial/",
        help="Path to save Checkpoints, Default: Checkpoints/",
    )

    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=1,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=1,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )


    Args = Parser.parse_args()
    BasePath = Args.BasePath
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath

    if not os.path.exists(CheckPointPath):
        os.makedirs(CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
        print("Latest Checkpoint Found")
    else:
        LatestFile = None
    
    TestOperation(
        MiniBatchSize,
        CheckPointPath,
        LatestFile,
        BasePath
    )
    

if __name__ == "__main__":
    main()