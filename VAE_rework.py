import os
from PIL import Image

import torch
import torch.nn as nn

import numpy as np
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torchvision.utils import save_image

from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from torch.optim import Adam
import matplotlib.pyplot as plt
from tkinter.filedialog import askdirectory

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
    

# Define the encoder and decoder classes
# The encoder produces the mean and log variance of the latent variables
class Encoder(nn.Module):
    # Initialize the class
    def __init__(self, dimensions_input, dimensions_hidden, dimensions_latent, img_size):
        # Initialize the superclass
        nn.Module.__init__(self)
        self.img_size = img_size

        # Set the input layer
        self.layer_input = nn.Linear(dimensions_input, dimensions_hidden)
        # Set the hidden layers
        self.second_layer = nn.Linear(dimensions_hidden, dimensions_hidden)
        self.third_layer = nn.Linear(dimensions_hidden, dimensions_hidden)
        if img_size != 32:
            self.fourth_layer = nn.Linear(dimensions_hidden, dimensions_hidden)
        # Set the mean layer
        self.average_layer  = nn.Linear(dimensions_hidden, dimensions_latent)
        # Set the variance layer
        self.var_layer   = nn.Linear (dimensions_hidden, dimensions_latent)
        # Set the leaky ReLU activation function
        self.LeakyReLU = nn.LeakyReLU(0.2)
        # Set the training flag
        self.training = True
        
    # Forward pass
    def forward(self, x):
        # Set the input layer
        layer_       = self.LeakyReLU(self.layer_input(x))
        # Set the hidden layers
        layer_       = self.LeakyReLU(self.second_layer(layer_))
        layer_       = self.LeakyReLU(self.third_layer(layer_))
        if self.img_size != 32:
            layer_       = self.LeakyReLU(self.fourth_layer(layer_))
        # Set the mean and variance layers
        mean     = self.average_layer(layer_)
        # Set the log variance
        log_var  = self.var_layer(layer_)
        
        return mean, log_var
    
# The decoder produces the output from the latent variables
class Decoder(nn.Module):
    # Initialize the class
    def __init__(self, dimensions_latent, dimensions_hidden, output_dimensions, IMG_SIZE):
        # Initialize the superclass
        nn.Module.__init__(self)
        self.img_size = IMG_SIZE
        # Set the input layer
        self.input = nn.Linear(dimensions_latent, dimensions_hidden)
        # Set the hidden layers
        self.second_layer = nn.Linear(dimensions_hidden, dimensions_hidden)
        self.third_layer = nn.Linear(dimensions_hidden, dimensions_hidden)
        if IMG_SIZE != 32:
            self.fourth_layer = nn.Linear(dimensions_hidden, dimensions_hidden)
        # Set the output layer
        self.output_layer = nn.Linear(dimensions_hidden, output_dimensions)
        # Set the leaky ReLU activation function
        self.LeakyReLU = nn.LeakyReLU(0.2)
    
    # Forward pass
    def forward(self, input):
        # Set the input layer
        layer     = self.LeakyReLU(self.input(input))
        # Set the hidden layers
        layer     = self.LeakyReLU(self.second_layer(layer))
        layer     = self.LeakyReLU(self.third_layer(layer))
        if self.img_size != 32:
            layer     = self.LeakyReLU(self.fourth_layer(layer))
        # Set the output layer
        x = self.output_layer(layer)
        # Return the output after going through the activation function
        x_hat = torch.sigmoid(x)
        # Return the output
        return x_hat
    
# The model class combines the encoder and decoder
class Model(nn.Module):
    # Initialize the class
    def __init__(self, class_of_Encoder, class_of_Decoder, DEVICE):
        # Initialize the superclass
        nn.Module.__init__(self)
        # Set the encoder
        self.Encoder = class_of_Encoder
        # Set the decoder
        self.Decoder = class_of_Decoder
        # Set the device
        self.DEVICE = DEVICE

    # Reparameterization trick    
    def reparameterization(self, average, x):
        # Set the epsilon (random noise)
        epsilon = torch.randn_like(x).to(self.DEVICE)
        # add the mean (average) with the standard deviation (x) multiplied by the random noise (epsilon)                
        trick = average + x*epsilon
        # Return the reparameterization trick                          
        return trick
        
    # Forward pass
    def forward(self, input):
        # Set the mean and log variance
        average, log_x = self.Encoder(input)
        # Set the reparameterization trick, it takes the exponent of the log variance to get the variance
        trick = self.reparameterization(average, torch.exp(0.5 * log_x))
        # Set the output
        x            = self.Decoder(trick)
        # Return the output
        return x, average, log_x
    

# Initialize the loss function (used to compare two distributions)
def loss_function(x, x_hat, mean, log_var):
    # Set the reproduction loss
    loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    # Set the Kullback-Leibler Divergence
    calc = 1 + log_var - mean**2 - torch.exp(log_var)
    Kullback = torch.sum(calc)
    final_Kullback = -0.5 * Kullback

    # Return the reproduction loss and KLD
    return loss + final_Kullback


def main():
    os.chdir("..")
    # If cuda is available, use it
    CUDA = True
    seed = 1
    CUDA = CUDA and torch.cuda.is_available()
    print("PyTorch version: {}".format(torch.__version__))
    if CUDA:
        print("CUDA version: {}\n".format(torch.version.cuda))

    if CUDA:
        torch.cuda.manual_seed(seed)
    DEVICE = torch.device("cuda:0" if CUDA else "cpu")

    # Set batch size (for GPU's with low memory, reduce the batch size)
    # Amount of images that are processed at the same time
    batch_size = 128
    # Set the amount of hidden dimensions
    hidden_dim = 400
    # Set the amount of latent dimensions
    latent_dim = 200
    # Set the learning rate
    lr = 1e-4
    # Set the amount of epochs
    epochs = 2

    version = input("Which version of the model do you want to train? 1 = learning rate, 2 = Image size, 3 = hidden dimensions, 4 = Baseline")

    path_baseline = input("Please select the path where the images and model saves should be stored of the baseline")
    path_mod_1 = input("Please select the path where the images and model saves should be stored of the second model (lr = 1e-5, hidden_dim = 300, or img dim = 32)")
    path_mod_2 = input("Please select the path where the images and model saves should be stored of the third model(lr = 1e-6, hidden_dim = 500, or img dim = 128)")
    directory_of_images = input("Please select the directory of images")
    location_of_txt_file_train = input("the location of the training txt files")
    location_of_txt_file_test =input("the location of the test txt files") 

    # Set the dimensions of the images
    if version == "2":
        img_dim_list = [64, 32, 128]
    else:
        img_dim_list = [64]
    
    for img_dim in img_dim_list:
        # Set the learning rate
        if version == "1":
            lr_list = [1e-4, 1e-5, 1e-6]
        else:
            lr_list = [1e-4]
        for lr in lr_list:
            # Set the amount of hidden dimensions
            if img_dim == 32:
                hidden_dim = 200
            if img_dim == 128:
                hidden_dim = 400
            elif img_dim == 64:
                hidden_dim = 400
            if version == "3":
                hidden_dim_list = [400, 300, 500]
            else:
                hidden_dim_list = [400]
            for hidden_dim in hidden_dim_list:
                # Set the path  
                if version == "1" and lr == 1e-5:
                    path = path_mod_1
                elif version == "1" and lr == 1e-6:
                    path = path_mod_2
                elif version == "2" and img_dim == 32:
                    path = path_mod_1
                elif version == "2" and img_dim == 128:
                    path = path_mod_2
                elif version == "3" and hidden_dim == 300:
                    path = path_mod_1
                elif version == "3" and hidden_dim == 500:
                    path = path_mod_2
                else:
                    path = path_baseline
                # Define the transformation for the images (resize, grayscale, from PIL Image to tensor, normalize with mean 0,5 and standard deviation 0,5)
                # Set the dimensions of the images
                X_DIM = img_dim
                # Set the dimensions of the input
                x_dim  = X_DIM * X_DIM
                transform = transforms.Compose([
                    transforms.Resize((X_DIM)),  
                    transforms.Grayscale(num_output_channels=1),  
                    transforms.ToTensor(),  
                ])

                # Create instance of dataset
                training_dataset = LoadImages(directory_of_images, location_of_txt_file_train, transform)
                testing_dataset = LoadImages(directory_of_images, location_of_txt_file_test, transform)
                # Create DataLoader (needed to create batches, and shuffle them accordingly)
                loader_train = torch.utils.data.DataLoader(training_dataset, batch_size=128, shuffle=True)
                test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=128, shuffle=True)

                # Initialize the encoder
                encoder = Encoder(x_dim, hidden_dim, latent_dim, img_dim)
                # Initialize the decoder
                decoder = Decoder(latent_dim, hidden_dim, x_dim, img_dim)

                # Initialize the model
                model = Model(encoder, decoder, DEVICE).to(DEVICE)

                # Initialize the optimizer
                loss_func = nn.BCELoss()

                # Initialize the optimizer
                Adam_optim = Adam(model.parameters(), lr=lr)

                # Start training the model
                print("Start training VAE...")

                # Set the model to training mode
                model.train()

                # Loop through the epochs
                for epoch in range(epochs):
                    # Set the total loss to 0
                    Total_loss = 0
                    # Loop through the batches
                    for index, batch in enumerate(loader_train):
                        # Flatten the image
                        batch = batch.view(batch.size(0), x_dim)
                        # Set the batch to the device  
                        batch = batch.to(DEVICE)
                        # Set the gradients to zero
                        Adam_optim.zero_grad()
                        # calculate the mean (average), log variance (log_variance) and output (output)
                        output, average, log_variance = model(batch)
                        # Calculate the loss, send the loss to the device
                        loss_between_distribs = loss_function(batch, output, average, log_variance)
                        # Add the loss to the total loss
                        Total_loss += loss_between_distribs.item()
                        # Backpropagate the loss
                        loss_between_distribs.backward()
                        # Update the weights by performing a step of the optimizer
                        Adam_optim.step()
                        # Print statistics
                        print("Batch index:", index,"\tCurrent loss: ", loss_between_distribs.item())
                        
                    print("\tEpoch", epoch + 1, "\tAverage Loss over all batches: ", Total_loss / (index*batch_size))

                    # Save images
                    # torch.no_grad() is used to prevent the model from updating the weights
                    with torch.no_grad():
                        # Set the noise
                        noise = torch.randn(batch_size, latent_dim).to(DEVICE)
                        # Generate images from the decoder by using the generated noise
                        generated_images = decoder(noise)
                        # Save the images
                        save_image(generated_images.view(batch_size, 1, X_DIM, X_DIM), f'{path}/generated_images_epoch_{epoch+1}'+ '_lr='+str(lr)+ 'img_dim='+str(img_dim)+'.png')
                    
                    # Save the model weights
                    torch.save(model.state_dict(), f'{path}/model_weights_VAE_total_epoch_{epoch+1}_'+ '_lr='+str(lr)+ 'img_dim='+str(img_dim)+ 'hidden_dim = ' + str(hidden_dim)+ '.pth')

                    # Save the encoder and decoder weights
                    torch.save(encoder.state_dict(), f'{path}/model_weights_VAE_encoder_epoch_{epoch+1}_'+ '_lr='+str(lr)+ 'img_dim='+str(img_dim)+ 'hidden_dim = ' + str(hidden_dim)+ '.pth')
                    torch.save(decoder.state_dict(), f'{path}/model_weights_VAE_decoder_epoch_{epoch+1}_'+ '_lr='+str(lr)+ 'img_dim='+str(img_dim)+ 'hidden_dim = ' + str(hidden_dim)+ '.pth')

                print("Training has concluded")
    
if __name__ == "__main__":
    main()
