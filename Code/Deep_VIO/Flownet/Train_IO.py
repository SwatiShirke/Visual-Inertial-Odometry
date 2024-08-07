import torch
# import torchvision
# #from torch.utils.tensorboard import SummaryWriter
# from torchvision import datasets, transforms
from torch.optim import lr_scheduler
import os
import cv2
import numpy as np
import ipdb
import csv 
import argparse
import matplotlib.pyplot as plt
from Misc.MiscUtils import FindLatestModel
from Network import loss_fn, Inertial_encoder
import wandb
device = "cuda"
wan = True
if wan:
    wandb.init(
        # set the wandb project where this run will be logged
        project="Deep-IO-FLowNet",
        name="m1_5e-5_adamW_45",

        # track hyperparameters and run metadata
        config={
        "learning_rate": 5e-5,
        "architecture": "Flownet-IO-2",
        "dataset": "Custom",
        "epochs": 30000,
        }
    )
def GenerateBatch(base_path, MiniBatchSize, TrainProbability,start_index, IMU_data, pose_data):
    """
    Base path: path to data folder which contains Imgs, IMU data and ground truth        
    """   
    
    # Img_train_batch = []
    # Img_test_batch = []
    IMU_train_batch = []    
    IMU_test_batch = []
    pose_train_batch = []
    pose_test_batch = []
    # png_files = [file for file in Dirfile_namesTrain if file.endswith('.png')]
    train_count = TrainProbability * MiniBatchSize 
    ImageNum = 0
    while ImageNum < MiniBatchSize and  start_index < 500:
        RandIndex = start_index  
         
        # img1_path = base_path + os.sep + str(RandIndex) + ".png"        
        # img2_path = base_path + os.sep + str(RandIndex + 1) + ".png"
        IMU_batch = torch.from_numpy( IMU_data[RandIndex * 10: RandIndex * 10 + 10])
        pose_batch = torch.from_numpy( pose_data[RandIndex * 10])
        
        
        if ImageNum < train_count:
            # Img_train_batch.append(torch.from_numpy(stacked_img))  
            IMU_train_batch.append(IMU_batch )  ##   
            pose_train_batch.append(pose_batch)       
        else:
            # Img_test_batch.append(torch.from_numpy(stacked_img))  
            IMU_test_batch.append(IMU_batch )  ##  
            pose_test_batch.append(pose_batch)  
        start_index +=1  
        ImageNum += 1

    # Img_train_batch = torch.stack(Img_train_batch) if len(Img_train_batch) !=0 else None
    # Img_test_batch = torch.stack(Img_test_batch) if len(Img_test_batch) !=0 else None
    IMU_train_batch = torch.stack(IMU_train_batch) if len(IMU_train_batch) !=0 else None
    IMU_test_batch = torch.stack(IMU_test_batch) if len(IMU_test_batch) !=0 else None
    pose_train_batch =  torch.stack(pose_train_batch) if len(pose_train_batch) !=0 else None
    pose_test_batch =  torch.stack(pose_test_batch) if len(pose_test_batch) !=0 else None

    return IMU_train_batch.to(device), IMU_test_batch.to(device), pose_train_batch.to(device), pose_test_batch.to(device), start_index


def read_csv(file_path):
    with open(file_path, mode='r') as file:
    # Create a CSV reader object
        reader = csv.reader(file)                   
    # Iterate over each row in the CSV file
        file_data = []
        for data in reader:
            # if row_count == 0:
            #     row_count += 1
            #     continue
            float_row = [float(value) for value in data]
            
            file_data.append(float_row) 
    #ipdb.set_trace()
    file_data = np.array(file_data)
    return file_data

def debug(TotalEpochs, MiniBatchSize, latest_chkpt):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(TotalEpochs))
    print("Mini Batch Size " + str(MiniBatchSize))
    # print("Number of Training Images " + str(NumTrainSamples))
    if latest_chkpt is not None:
        print("Loading latest checkpoint with the name " + latest_chkpt)

def plot_and_save_metrics(loss_vs_iteration, loss_vs_epoch, CheckPointPath, title, filename):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(loss_vs_iteration) + 1), loss_vs_iteration, label='Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'{title} Loss vs Iteration')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(loss_vs_epoch) + 1), loss_vs_epoch, label='Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{title} Loss vs Epoch')
    plt.legend()
    plt.tight_layout()

    # Define the path to save the figure
    save_path = os.path.join(CheckPointPath, filename)
    # Save the figure
    plt.savefig(save_path)
    print(f"Figure saved at: {save_path}")
    # Display the figure
    plt.show()




def TrainOperation(
    TotalEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    latest_chkpt,
    BasePath,
    LogsPath,
    splitPerc
):
    
    # Define the model
    model = Inertial_encoder().to(device)
    # Define the optimizer
    Optimizer1 = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=5e-6)
    Optimizer2 = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=5e-6)

    # Define the learning rate scheduler
    scheduler = lr_scheduler.StepLR(Optimizer1, step_size=10, gamma=0.1)
    # Define the loss function


    # Load the latest checkpoint    
    if latest_chkpt is not None:
        CheckPoint = torch.load(latest_chkpt + ".ckpt")
        # Extract only numbers from the name
        Start = int("".join(c for c in latest_chkpt.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + latest_chkpt + "....")
    else:
        Start = 0
        print("New model initialized....")
   

    epoch_loss_train = []
    epoch_loss_val = []
    file_names = os.listdir(BasePath)

    for Epochs in range(Start, TotalEpochs):
        
        if Epochs<200:
            Optimizer = Optimizer1
        else:
            Optimizer = Optimizer2  
            

        for file in file_names:
            Basepath = os.path.join(BasePath,file)
            IMU_path = f"{Basepath}/IMU_data_file.csv"
            pose_path = f"{Basepath}/Pose_data.csv"
            IMU_data = read_csv(IMU_path)   
            pose_data = read_csv(pose_path)
            print(f"{file} Training Started")
            NumTrainSamples = len([filename for filename in os.listdir(Basepath) if filename.endswith('.png')])
            
            NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize)
            start_index = 0
            for PerEpochCounter in range(NumIterationsPerEpoch):
                IMU_train_batch , IMU_test_batch, pose_train_batch, pose_test_batch, start_index = GenerateBatch(Basepath, MiniBatchSize, splitPerc, start_index, IMU_data, pose_data)
                # Predict output with forward pass
                if IMU_train_batch is not None and pose_train_batch is not None:
                    # pose_train_predicted = model(Img_train_batch, IMU_train_batch)
                    model.train()
                    pose_train_predicted = model(IMU_train_batch.float()).float()
                    pose_train_predicted = pose_train_predicted.to(device)

                    Loss_pose_train = loss_fn(pose_train_predicted, pose_train_batch.float()).float()
                    Loss_pose_train = Loss_pose_train.to(device)
                    loss_xyz = Loss_pose_train.detach().cpu().numpy()
                    epoch_loss_train.append(loss_xyz)

                    Optimizer.zero_grad()
                    Loss_pose_train.mean().backward()
                    Optimizer.step()
                    if wan:
                        wandb.log({"iter loss train": Loss_pose_train.item()})
                    
                    # scheduler.step()

                    # Save checkpoint every some SaveCheckPoint's iterations
                    if PerEpochCounter % SaveCheckPoint == 0:
                        # Save the Model learnt in this epoch
                        SaveName = (
                            CheckPointPath
                            + str(Epochs)
                            + "a"
                            + str(PerEpochCounter)
                            + "model.ckpt"
                        )

                        torch.save(
                            {
                                "epoch": Epochs,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": Optimizer.state_dict(),
                                "loss": Loss_pose_train,
                            },
                            SaveName,
                        )
                        print("\n" + SaveName + " Model Saved...")

                if IMU_test_batch is not None and pose_test_batch is not None:
                    # ipdb.set_trace()
                    model.eval()
                    with torch.no_grad():
                        Loss_pose_val = model.validation_step(IMU_test_batch.float(), pose_test_batch.float())
                    epoch_loss_val.append(Loss_pose_val.detach().cpu().numpy())

            average_epoch_loss_train = sum(epoch_loss_train[-NumIterationsPerEpoch:]) / NumIterationsPerEpoch
            if wan:
                wandb.log({"avg epochloss train": average_epoch_loss_train})
            
            average_epoch_loss_val = sum(epoch_loss_val[-NumIterationsPerEpoch:]) / NumIterationsPerEpoch
            if wan:
                wandb.log({"avg epochloss val": average_epoch_loss_val})
            

            # Save model every epoch
            SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
            torch.save(
                {
                    "epoch": Epochs,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": Optimizer.state_dict(),
                    "loss_train": Loss_pose_train,
                    "loss_val": Loss_pose_val,
                },
                SaveName,
            )
            print("\n" + SaveName + " Model Saved...")
        print(f"{file} Training Ended")



def main():
    """
    Main Function
    Parameters are defined in the main itself
    
    This function is the entry point of the program. It parses command line arguments, sets up necessary paths,
    and starts the training process. It also prints the values of important variables at the start of the training.
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default="./data",
        help="Base path of images, Default:../Data",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="Checkpoints_inertial_2/",
        help="Path to save Checkpoints, Default: Checkpoints/",
    )

    Parser.add_argument(
        "--TotalEpochs",
        type=int,
        default=3000000,
        help="Number of Epochs to Train for, Default:30",
    )

    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=45,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="Logs_inertial/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )
    Parser.add_argument(
        "--Split",
        type=int,
        default=0.7,
        help="Train Validation Split",
    )

    Args = Parser.parse_args()
    TotalEpochs = Args.TotalEpochs
    BasePath = Args.BasePath
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    splitPerc = Args.Split

    if not os.path.exists(CheckPointPath):
        os.makedirs(CheckPointPath)

    if not os.path.exists(LogsPath):
        os.makedirs(LogsPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        latest_chkpt = FindLatestModel(CheckPointPath)
        print("Latest Checkpoint Found")
    else:
        latest_chkpt = None
    
    # NumTrainSamples = len([filename for filename in os.listdir(BasePath) if filename.endswith('.png')])
    SaveCheckPoint = 100

    #  print all variables at the start of the training
    debug(TotalEpochs, MiniBatchSize, latest_chkpt)
    
    TrainOperation(
        TotalEpochs,
        MiniBatchSize,
        SaveCheckPoint,
        CheckPointPath,
        latest_chkpt,
        BasePath,
        LogsPath,
        splitPerc
    )

if __name__ == "__main__":
    main()
