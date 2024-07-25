import os
import sys
import math
import tqdm
import numpy as np
import torch.nn.functional as F
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import save_image
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
    

class Discriminator(nn.Module):
    def __init__(self, IMAGE_CHANNEL, D_HIDDEN, img_size):
        super(Discriminator, self).__init__()
        if img_size == 64:
            self.main = nn.Sequential(
                # 1st layer
                nn.Conv2d(IMAGE_CHANNEL, D_HIDDEN, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # 2nd layer
                nn.Conv2d(D_HIDDEN, D_HIDDEN * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(D_HIDDEN * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # 3rd layer
                nn.Conv2d(D_HIDDEN * 2, D_HIDDEN * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(D_HIDDEN * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # 4th layer
                nn.Conv2d(D_HIDDEN * 4, D_HIDDEN * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(D_HIDDEN * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # Sigmoid is commonly used for binary classification problems, as it squashes the output to be between 0 and 1. 1 = real 0 = fake.
                nn.Conv2d(D_HIDDEN * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
        elif img_size == 128:
            self.main = nn.Sequential(
                # 1st layer
                nn.Conv2d(IMAGE_CHANNEL, D_HIDDEN, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # 2nd layer
                nn.Conv2d(D_HIDDEN, D_HIDDEN * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(D_HIDDEN * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # 3rd layer
                nn.Conv2d(D_HIDDEN * 2, D_HIDDEN * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(D_HIDDEN * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # 4th layer
                nn.Conv2d(D_HIDDEN * 4, D_HIDDEN * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(D_HIDDEN * 8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(D_HIDDEN * 8, D_HIDDEN * 16, 4, 2, 1, bias=False),
                nn.BatchNorm2d(D_HIDDEN * 16),
                nn.LeakyReLU(0.2, inplace=True),
                # Sigmoid is commonly used for binary classification problems, as it squashes the output to be between 0 and 1. 1 = real 0 = fake.
                nn.Conv2d(D_HIDDEN * 16, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
        else:
            self.main = nn.Sequential(
                # 1st layer
                nn.Conv2d(IMAGE_CHANNEL, D_HIDDEN, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # 2nd layer
                nn.Conv2d(D_HIDDEN, D_HIDDEN * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(D_HIDDEN * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # # 3rd layer
                nn.Conv2d(D_HIDDEN * 2, D_HIDDEN * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(D_HIDDEN * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # Sigmoid is commonly used for binary classification problems, as it squashes the output to be between 0 and 1. 1 = real 0 = fake.
                nn.Conv2d(D_HIDDEN * 4, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)
    
def weights_init(m):
    # Custom weights initialization called on netG and netD to help the models converge
    classname = type(m).__name__
    # Initialize the weights of the convolutional and batch normalization layers
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.03)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.03)
        m.bias.data.fill_(0)

def diversity_loss(img):
    # Calculate the diversity loss of the generated images
    res = img.size(0)
    # Flatten the images to calculate the pairwise distances
    img = img.view(res, -1)
    total_loss = 0

    # Manually compute pairwise distances
    for i in range(res):
        for j in range(i + 1, res):
            distance_btwn_images = torch.norm(img[i] - img[j], p=2)
            total_loss += distance_btwn_images
    # Negative diversity loss, because we want to promote diversity
    div_loss = -total_loss / (res * (res - 1) / 2)  
    return div_loss


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
    

    

def main():
    os.chdir("..")
    version = input("Which version of the model do you want to make a graph of? 1 = learning rate, 2 = Image size, 3 = hidden dimension, 4 = Baseline")
    baseline_path = input("what is the checkpoint location of the baseline model?")
    if version == '1' or version == '2' or version == '3':
        input_model_1 = input("what is the checkpoint location of the second model? (for learning rate 1e-5, for image size 32, for hidden dimension 300)")
        input_model_2 = input("what is the checkpoint location of the third model? (for learning rate 1e-6, for image size 64, for hidden dimension 500)")
    location_of_discriminator = input("What is the location of the discriminator? (128 x 128)")
    if version == "2":
        location_of_discriminator_32 = input("What is the location of the (32 x 32) discriminator?")
        location_of_discriminator_64 = input("What is the location of the (64 x 64) discriminator?")
    # Flag to use CUDA
    CUDA = True
    # Batch size for training
    BATCH_SIZE = 128
    # Number of channels in the RGB image (1 channel = grayscale)
    RGB_CHANNEL = 1
    # Size of the input noise vector
    INPUT_NOISE_VECTOR = 100
    # Number of hidden layers in the generator
    HIDDEN_DIM_GEN = 64
    # Number of hidden layers in the discriminator
    HIDDEN_DIM_DISCR = 64
    # Number of epochs to train the GAN
    EPOCH_NUM = 1
    # Real and fake labels
    REAL_LABEL = 1
    FAKE_LABEL = 0
    # Learning rate for the GAN	
    lr = 1e-4
    seed = 1

    # check if GPU is available
    CUDA = CUDA and torch.cuda.is_available()
    print("PyTorch version: {}".format(torch.__version__))
    if CUDA:
        print("CUDA version: {}\n".format(torch.version.cuda))

    if CUDA:
        torch.cuda.manual_seed(seed)
    DEVICE = torch.device("cuda:0" if CUDA else "cpu")


    directory_of_images = input("What is the directory of the images? (all together not in different folders)")
    location_of_txt_file = input("What is the location of the txt file with the train value names?")
    location_of_txt_file_test = input("What is the location of the txt file with the test value names?")
    if version == "3":
        hidden_dim_list = [400, 300, 500]
    else:
        hidden_dim_list = [400]
    if version == "2":
        image_size_list = [128, 32, 64]
    else:
        image_size_list = [128]
    if version == "1":
        lr_list = [1e-4, 1e-5, 1e-6]
    else:
        lr_list = [1e-4]
    for lr in lr_list:
        for image_size in image_size_list:
            for hidden_dim in hidden_dim_list:
                # Define the transformation for the images (resize, grayscale, from PIL Image to tensor, normalize with mean 0,5 and standard deviation 0,5)
                transform = transforms.Compose([
                    transforms.Resize((image_size)),  
                    transforms.Grayscale(num_output_channels=1),  
                    transforms.ToTensor(),  
                    transforms.Normalize((0.5,), (0.5,))
                ])

                # Create instance of dataset
                training_dataset = LoadImages(directory_of_images, location_of_txt_file, transform)
                testing_dataset = LoadImages(directory_of_images, location_of_txt_file_test, transform)

                # Create DataLoaders for training and testing sets
                train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
                test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=BATCH_SIZE, shuffle=True)


                # Create the discriminator
                Disc_net = Discriminator(RGB_CHANNEL, HIDDEN_DIM_DISCR, image_size).to(DEVICE)
                # Load the weights of the discriminator
                if image_size == 128:
                    Disc_net.load_state_dict(torch.load(location_of_discriminator))
                elif image_size == 32:
                    Disc_net.load_state_dict(torch.load(location_of_discriminator_32))
                elif image_size == 64:
                    Disc_net.load_state_dict(torch.load(location_of_discriminator_64))

                # Setting hyperparameters
                batch_size = 128
                X_DIM = image_size
                x_dim  = X_DIM * X_DIM
                hidden_dim = 400
                latent_dim = 200
                # create the encoder and decoder
                encoder = Encoder(x_dim, hidden_dim, latent_dim, image_size)
                decoder = Decoder(latent_dim, hidden_dim, x_dim, image_size)
                # create the model
                model = Model(encoder, decoder, DEVICE).to(DEVICE)

                # Initialize BCELoss function
                BCE_crit = nn.BCELoss()

                def loss_function(x, x_hat, mean, log_var):
                    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
                    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

                    return reproduction_loss + KLD



                print("Starting Training Loop...")
                print("Starting Testing Loop...")
                # Create a list to store the loss of the VAE
                list_of_disc_errors_500 = np.zeros(EPOCH_NUM)
                list_of_disc_errors_300 = np.zeros(EPOCH_NUM)
                list_of_disc_errors_baseline = np.zeros(EPOCH_NUM)
                list_of_epochs = np.arange(0, EPOCH_NUM, 1)
                # Create plots to show the loss of the Discriminator per epoch
                fig, plt1 = plt.subplots()
                plt1.set_xlabel('Epoch')
                plt1.set_ylabel('Discriminator Loss')      
                plt1.set_title("Discriminator Loss vs epoch Variational Autoencoder")
                if version == "1" or version == "2" or version == "3":
                    paths = [baseline_path, input_model_1, input_model_2]
                else:
                    path = [baseline_path]
                for path in paths:
                    for epoch in range(EPOCH_NUM):
                        for i, data in enumerate(train_loader, 0):
                            # Create the discriminator made for the 64x64 images
                            Disc_net = Discriminator(RGB_CHANNEL, HIDDEN_DIM_DISCR, image_size).to(DEVICE)
                            # Load the weights of the discriminator
                            if image_size == 64:
                                Disc_net.load_state_dict(torch.load(location_of_discriminator_64))
                            elif image_size == 32:
                                Disc_net.load_state_dict(torch.load(location_of_discriminator_32))
                            elif image_size == 128:
                                Disc_net.load_state_dict(torch.load(location_of_discriminator))
                            # Load the images and labels 
                            moving_images = data.to(DEVICE)
                            batch_size = moving_images.size(0)
                            label = torch.full((batch_size,), REAL_LABEL, dtype=torch.float, device=DEVICE)
                            
                            # Forward pass real batch through Discriminator
                            output = Disc_net(moving_images).view(-1)
                            real_error_disc = BCE_crit(output, label)     
                            # Generate batch of latent vectors
                            noise = torch.randn(batch_size, INPUT_NOISE_VECTOR, 1, 1, device=DEVICE)
                            # Generate fake image batch with G
                            model.eval()

                            with torch.no_grad():
                                # Set parameters for discriminator
                                Disc_net = Discriminator(RGB_CHANNEL, HIDDEN_DIM_GEN, image_size).to(DEVICE)
                                if image_size == 64:
                                    Disc_net.load_state_dict(torch.load(location_of_discriminator_64))
                                elif image_size == 32:
                                    Disc_net.load_state_dict(torch.load(location_of_discriminator_32))
                                elif image_size == 128:
                                    Disc_net.load_state_dict(torch.load(location_of_discriminator))

                                # Create the encoder and decoder
                                encoder = Encoder(x_dim, hidden_dim, latent_dim, image_size)
                                decoder = Decoder(latent_dim,  hidden_dim, x_dim, image_size)

                                # Create the model and load the weights
                                model = Model(encoder, decoder, DEVICE).to(DEVICE)
                                model.load_state_dict(torch.load(str(path) + '/model_weights_VAE_total_epoch_' + str(epoch+1) + '__lr='+str(lr)+ 'img_dim='+str(image_size)+ 'hidden_dim = ' + str(hidden_dim)+ '.pth'))\
                                
                                # Generate the fake images
                                noise = torch.randn(batch_size, latent_dim).to(DEVICE)
                                fake = decoder(noise)
                                fake = fake.view(batch_size, 1, X_DIM, X_DIM)
                            
                            with torch.no_grad():
                                # Set the noise
                                noise = torch.randn(batch_size, latent_dim).to(DEVICE)
                                # Generate images from the decoder by using the generated noise
                                generated_images = decoder(noise)
                                # Save the images
                                save_image(generated_images.view(batch_size, 1, image_size, image_size), str(path) + 'generated_images_epoch_'+ str(epoch) + '_lr='+str(lr)+ '.png')
                                        
                            label.fill_(FAKE_LABEL)

                            # Use the discriminator to classify the fake images
                            class_output = Disc_net(fake.detach()).view(-1)
                            # Calculate the loss of the discriminator
                            Disc_loss_fake = BCE_crit(class_output, label)
                            # Calculate the mean of the discriminator predictions over the generated images
                            D_G_z1 = class_output.mean().item()
                            # Compute error of the discriminator as sum over the fake and the real batches
                            errD = real_error_disc + Disc_loss_fake
                            if path == baseline_path:
                                list_of_disc_errors_baseline[epoch] = errD.item()
                            elif path == input_model_1:
                                list_of_disc_errors_500[epoch] = errD.item()
                            else:
                                list_of_disc_errors_300[epoch] = errD.item()
                            print(epoch)
                            print(errD.item())
                            break
                    if path == baseline_path:
                        plt1.plot(list_of_epochs,list_of_disc_errors_baseline,label="Discriminator Loss of the baseline")
                    elif path == input_model_1 and version == "1":
                        plt1.plot(list_of_epochs,list_of_disc_errors_500,label="Discriminator Loss of the VAE model with lr = 1e-5")
                    elif path == input_model_1 and version == "2":
                        plt1.plot(list_of_epochs,list_of_disc_errors_500,label="Discriminator Loss of the VAE model with image size = 32")
                    elif path == input_model_1 and version == "3":
                        plt1.plot(list_of_epochs,list_of_disc_errors_500,label="Discriminator Loss of the VAE model with hidden dimension = 300")
                    elif path == input_model_2 and version == "1":
                        plt1.plot(list_of_epochs,list_of_disc_errors_300,label="Discriminator Loss of the VAE model with lr = 1e-6")
                    elif path == input_model_2 and version == "2":
                        plt1.plot(list_of_epochs,list_of_disc_errors_300,label="Discriminator Loss of the VAE model with image size = 64")
                    elif path == input_model_2 and version == "3":
                        plt1.plot(list_of_epochs,list_of_disc_errors_300,label="Discriminator Loss of the VAE model with hidden dimension = 500")
    plt1.legend()
    if version == "1":
        fig.savefig("Discriminator_vs_VAE_lr.png",dpi=300)
    if version == "2":
        fig.savefig("Discriminator_vs_VAE_image_size.png",dpi=300)
    if version == "3":
        fig.savefig("Discriminator_vs_VAE_hidden_dim.png",dpi=300)
    if version == "4":
        fig.savefig("Discriminator_vs_VAE_baseline.png",dpi=300)    

if __name__=="__main__":
	main()