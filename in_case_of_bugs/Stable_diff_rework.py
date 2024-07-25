from dataclasses import dataclass
import matplotlib.pyplot as plt

from PIL import Image

import math
import os
import numpy as np

import diffusers
import accelerate

from huggingface_hub import HfFolder, Repository, whoami

import torchvision
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from pathlib import Path
from itertools import islice
from mpl_toolkits.axes_grid1 import ImageGrid

# Check if GPU is used in the code
if torch.cuda.is_available():
    print("CUDA is available. Using the GPU.")
    device = torch.device("cuda")

# Access token for upload to huggingface repository
# access_token = "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"

# Parameters for training
class configuration:
    # Image resolution
    img_dim = 64
    # Number of images sampled when training the model
    size_of_the_train_batch = 128
    # Number of images sampled when evaluating the model
    size_of_the_evaluation_batch = 128
    total_epochs = 75
    # Number of epochs between each evaluation (1 = off)
    gradient_acc_steps = 1
    # Learning rate
    lr = 1e-4
    # Warmup steps for the learning rate scheduler (lr increases linearly from 0 to lr for the first warmup_steps steps)
    warmup_for_lr = 500
    # Number of epochs between each time the model is saved
    saving_image_and_model_epochs = 1
    # Precision to use on the GPU (fp16 has automatic precision)
    precision = "fp16"
    # Directory where the model is saved (also the name of the private repository)
    directory_of_output = "/data/s3287297/diff_baseline"

    # Whether to push the model to huggingface and its settings
    push_huggingface = False
    private_repo = True
    overwrite_old_repo = True

    # seed for algorithm
    seed = 0

# Class to load in images from the dataset
class LoadImages:
    # Initialize the class
    def __init__(self, directory_of_images, file_with_labels, preprocessing):
        # Set the directory of images
        self.directory_of_images = directory_of_images
        # Set the preprocessing function
        self.preprocessing = preprocessing
        # Initialize the list of data locations
        self.data = []

        # Open the file with labels open as read
        with open(file_with_labels, "r") as current_file:
            # Read all lines in the file
            lines_in_file = current_file.readlines()
            # Loop through all lines in the file
            for current_line in lines_in_file:
                # Split the current line by the last underscore
                current_line = current_line.strip().rsplit("_", 0)
                # Add the current line to the list of image data locations
                self.data.append(current_line)
    
    # Get the length of the data list
    def __len__(self):
        # Return the number of images in the dataset
        return len(self.data)
    
    # Get the item at the index
    def __getitem__(self, idx):
        # Get the image location
        image_name = self.directory_of_images + "/" + self.data[idx][0]
        # Open the image with index idx
        curr_image = Image.open(image_name)\
        
        # Check if the image is grayscale and normalized (check that transform is applied)
        if self.preprocessing is not None:
            curr_image = self.preprocessing(curr_image)
        
        return curr_image


class train:
    def __init__(self):
        # Define the directory of images and the location of the text file (which contains the list of images to load in)
        os.chdir('..')
        os.chdir('..')
        self.directory_of_images =r'/data/s3287297/NIH_data/images'
        self.location_of_txt_file =r'/data/s3287297/NIH_data/train_val_list.txt'
        self.train_conf = configuration()

        # Define the transformation for the images (resize, grayscale, from PIL Image to tensor, normalize with mean 0,5 and standard deviation 0,5)
        self.transform_images = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((self.train_conf.img_dim, self.train_conf.img_dim)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.Grayscale(num_output_channels=1),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5], [0.5]),
            ]
        )
        # Create the noise creator scheduler
        self.noise_creator_scheduler = diffusers.DDPMScheduler(num_train_timesteps=1000)

        # Create custom dataset instance
        training_dataset = LoadImages(self.directory_of_images, self.location_of_txt_file, self.transform_images)

        self.model = diffusers.UNet2DModel(
            # Resolution of the images
            sample_size=self.train_conf.img_dim,
            # Amount of input channels, 1 for greyscale images 
            in_channels=1,
            # Amount of output channels, 1 for greyscale images  
            out_channels=1,
            # Number of UNet blocks 
            layers_per_block=2,
            # Number of output channels for each UNet block (for 64 x 64 and higher 2, lower 1 is recommended) 
            block_out_channels=(self.train_conf.img_dim * 2, self.train_conf.img_dim * 2, self.train_conf.img_dim * 4 , self.train_conf.img_dim * 4, self.train_conf.img_dim * 8, self.train_conf.img_dim * 8),  
            down_block_types=(
                # This block downsamples the input
                "DownBlock2D",  
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                # This block has spatial self-attention for improved spatial performance
                "AttnDownBlock2D",  
                "DownBlock2D",
            ),
            up_block_types=(
                # This block upsamples the input
                "UpBlock2D",
                # This block has spatial self-attention for improved spatial performance
                "AttnUpBlock2D",  
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )


        # Create DataLoaders for training and testing sets
        self.loader_for_train = torch.utils.data.DataLoader(training_dataset, batch_size=self.train_conf.size_of_the_train_batch, shuffle=True)

    def make_grid_numpy(self, images, rows, cols):
        # Assuming all images are the same size
        width, height = images[0].size
        
        # Create an empty array for the grid
        grid_array = np.zeros((rows * height, cols * width, 1), dtype=np.uint8)
        
        # Populate the grid with images
        for index, img in enumerate(images):
            # Convert image to numpy array and add an extra dimension
            img_array = np.array(img)[:, :, np.newaxis]
            
            # Calculate row and column in the grid
            row = index // cols
            col = index % cols
            
            # Calculate start positions for this image in the grid
            start_y = row * height
            start_x = col * width
            
            # Place the image in the grid
            grid_array[start_y:start_y+height, start_x:start_x+width, :] = img_array
        
        # Convert the grid array back to an image
        grid_image = Image.fromarray(grid_array.squeeze())  # Use squeeze() to remove singleton dimensions for display/save
        return grid_image



    def model_evaluation(self, epoch, pipeline):
        # Generate images from the pipeline by using a manual seed
        img = pipeline(
            batch_size=self.train_conf.size_of_the_evaluation_batch,
            generator=torch.manual_seed(self.train_conf.seed),
        ).images
        # Create a grid of images
        img_grid = self.make_grid_numpy(img, rows=16, cols=8)

        # Save the images
        sample_directory = os.path.join(self.train_conf.directory_of_output, "samples")
        os.makedirs(sample_directory, exist_ok=True)
        img_grid.save(f"/data/s3287297/diff_baseline/{epoch:04d}.png")

    def noise_creator(self):
        # # Create a figure with 4 subplots
        # f, axis = plt.subplots(1, 4, figsize=(16, 4))
        # Get the first 4 batches of images
        first_four_batches = []
        for i, batch in enumerate(self.loader_for_train):
            first_four_batches.append(batch)
            if i == 1: 
                break
        # Get the first image from the first batch
        image_from_batch = first_four_batches[0]

        # print("Input shape:", image_from_batch.shape)

        # print("Output shape:", self.model(image_from_batch, timestep=0).sample.shape)
        # # Load the state dict previously saved
        state_dict = torch.load('/data/s3287297/diff_baseline/model_weights_stable_diff_epoch_49.pth')
        print("continuing from epoch 50")


        # Update the model's weights
        self.model.load_state_dict(state_dict)

        # Generate random noise
        random_noise = torch.randn(image_from_batch.shape)
        # How many steps are needed to reach the target noise level
        steps_needed = torch.LongTensor([50])
        # Add noise to the image
        image_with_noise = self.noise_creator_scheduler.add_noise(image_from_batch, random_noise, steps_needed)
    
        # print("Noisy image shape:", image_with_noise.shape)
        # [0] because we only have one image in the batch
        # permutes, normalizes, scales and converts the image to numpy
        permute = image_with_noise[0].permute(1, 2, 0)
        image = (((permute + 1.0) * 127.5).type(torch.uint8).numpy()[0])
        # Convert the numpy array to a PIL image
        Image.fromarray(image)
        

        # Add noise to the image
        predicted_noise = self.model(image_with_noise, steps_needed).sample
        # Calculate the mean squared error loss
        calculate_MSEloss_function = F.mse_loss(predicted_noise, random_noise)
        # Calculate length of the dataset
        length_of_dataset = len(self.loader_for_train)

        # Initialize the Adam optimizer for more stable learning
        self.Adam_optim = torch.optim.Adam(self.model.parameters(), lr=self.train_conf.lr)
        
        # Initialize the scheduler for more stable learning
        self.scheduler_for_stable_learning = diffusers.optimization.get_cosine_schedule_with_warmup(
            optimizer=self.Adam_optim,
            num_warmup_steps=self.train_conf.warmup_for_lr,
            num_training_steps=(length_of_dataset * self.train_conf.total_epochs),
        )
    def model_training(self):
        # Initialize accelerator for more efficient training
        huggingface_train_package = accelerate.Accelerator(
            mixed_precision=self.train_conf.precision,
            # Number of gradient accumulation steps (1 = off, if the GPU runs out of memory, increase this number may slow down learning)
            # if this number is increased it takes more time to update the weights (it averages the gradients over the number of steps and then updates the weights)
            gradient_accumulation_steps=self.train_conf.gradient_acc_steps,
            project_dir=os.path.join(self.train_conf.directory_of_output, "logs"),
        )
        # Initialize the repository for the model
        if huggingface_train_package.is_main_process:
            if self.train_conf.push_huggingface:
                name_of_repository = "DMR2/NIH_diffusion_model"
                repo = Repository(self.train_conf.directory_of_output, clone_from=name_of_repository)
            elif self.train_conf.directory_of_output is not None:
                os.makedirs(self.train_conf.directory_of_output, exist_ok=True)
            huggingface_train_package.init_trackers("train_diffusion_model")

        # Prepare the model for training, order does not matter
        self.model, self.Adam_optim, self.loader_for_train, self.scheduler_for_stable_learning = huggingface_train_package.prepare(
            self.model, self.Adam_optim, self.loader_for_train, self.scheduler_for_stable_learning
        )

        # Initialize the global step for logging
        global_step_for_logs = 0

        # Train the model
        for epoch in range(49,self.train_conf.total_epochs):
            print(f"Epoch {epoch}/{self.train_conf.total_epochs}")

            # Loop through the training data
            for idx, current_batch in enumerate(self.loader_for_train):
                # Get first images from the current batch
                clean_images = current_batch
                # add noise to the images
                noise_random = torch.randn(clean_images.shape).to(clean_images.device)
                batch_size_of_clean_images = clean_images.shape[0]

                # Randomly sample timesteps for each image in the batch
                timestep_random = torch.randint(
                    0, self.noise_creator_scheduler.config.num_train_timesteps, (batch_size_of_clean_images,), device=clean_images.device
                ).long()

                # Apply noise based on the timesteps generated
                images_with_noise = self.noise_creator_scheduler.add_noise(clean_images, noise_random, timestep_random)

                with huggingface_train_package.accumulate(self.model):
                    # Predict the noise residual
                    pred_of_noise = self.model(images_with_noise, timestep_random, return_dict=False)[0]
                    # Calculate the mean squared error loss
                    MSE_loss = F.mse_loss(pred_of_noise, noise_random)
                    # Backward pass to update the weights
                    huggingface_train_package.backward(MSE_loss)

                    # Clip the gradients to prevent exploding gradients
                    huggingface_train_package.clip_grad_norm_(self.model.parameters(), 1.0)
                    # Update the weights
                    self.Adam_optim.step()
                    # Update the learning rate
                    self.scheduler_for_stable_learning.step()
                    # Zero the gradients
                    self.Adam_optim.zero_grad()
                # Log the loss, learning rate and step
                logs = {"loss": MSE_loss.detach().item(), "lr": self.scheduler_for_stable_learning.get_last_lr()[0], "step": global_step_for_logs}
                huggingface_train_package.log(logs, step=global_step_for_logs)
                global_step_for_logs += 1
                print(f"Loss: {MSE_loss.item()} current_batch: {idx}/{len(self.loader_for_train)}")

            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if huggingface_train_package.is_main_process:
                DDPM = diffusers.DDPMPipeline(unet=huggingface_train_package.unwrap_model(self.model), scheduler=self.noise_creator_scheduler)

                if (epoch + 1) % self.train_conf.saving_image_and_model_epochs == 0 or epoch == self.train_conf.total_epochs - 1:
                    self.model_evaluation(epoch, DDPM)

                if (epoch + 1) % self.train_conf.saving_image_and_model_epochs == 0 or epoch == self.train_conf.total_epochs - 1:
                    if self.train_conf.push_huggingface:
                        train_conf = self.train_conf
                        repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                    else:
                        DDPM.save_pretrained(self.train_conf.directory_of_output)
            # Save the model
            huggingface_train_package.save(self.model.state_dict(), f'/data/s3287297/diff_baseline/model_weights_stable_diff_epoch_{epoch}.pth')
    




def main():
    launch_train = train()
    launch_train.noise_creator()
    launch_train.model_training()

if __name__ == "__main__":
    main()

    