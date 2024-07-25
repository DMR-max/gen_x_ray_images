import os
import numpy as np
import torch
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
        # The amount of images found in the list
        return len(self.data)
    
    # Get the item at the index
    def __getitem__(self, idx):
        # Get the image location
        image_name = self.directory_of_images + "/" + self.data[idx][0]
        # Open the image with index idx
        curr_img = Image.open(image_name)\
        
        # Check if the image is grayscale and normalized (check that transform is applied)
        if self.preprocessing is not None:
            curr_img = self.preprocessing(curr_img)
        
        return curr_img
    
# Main function for training GAN    
def main():
    # Print that the GAN training is starting
    print("Starting GAN training...")
    # Flag to use CUDA
    CUDA = True
    # Path to the data
    DATA_PATH = './data'
    # Batch size for training
    BATCH_SIZE = 128
    # Number of channels in the RGB image (1 channel = grayscale)
    RGB_CHANNEL = 1
    # Size of the input noise vector
    INPUT_NOISE_VECTOR = 100
    # Number of hidden dimensions in the generator
    HIDDEN_DIM_GEN = 128
    # Dimensions of the image
    DIM_IMAGE = 64
    # Number of hidden dimensions in the discriminator
    HIDDEN_DIM_DISCR = 128
    # Number of epochs to train the GAN
    EPOCH_NUM = 50
    # Real and fake labels
    REAL_LABEL = 1
    FAKE_LABEL = 0
    # Seed for reproducibility
    seed = 1

    # check if GPU is available
    CUDA = CUDA and torch.cuda.is_available()
    print("PyTorch version: {}".format(torch.__version__))
    if CUDA:
        print("CUDA version: {}\n".format(torch.version.cuda))

    if CUDA:
        torch.cuda.manual_seed(seed)
    device = torch.device("cuda:0" if CUDA else "cpu")
    cudnn.benchmark = True

    
    version = input("Which version of the model do you want to train? 1 = learning rate, 2 = Image size, 3 = hidden dimensions, 4 = Baseline")

    # Define the directory of images and the location of the text file with all image names for training 
    path_baseline = input("Please select the path where the images and model saves should be stored of the baseline")
    if version != "4":
        path_mod_1 = input("Please select the path where the images and model saves should be stored of the second model (lr = 1e-5, hidden_dim = 32, or img dim = 32)")
        path_mod_2 = input("Please select the path where the images and model saves should be stored of the third model (lr = 1e-3, hidden_dim = 128, or img dim = 128)")
    
    directory_of_images = input("Please select the directory of images")
    location_of_txt_file = input("the location of the training txt files")

    if version == "2":
        img_dim_list = [32, 64, 128]
    else:
        img_dim_list = [64]
    
    for DIM_IMAGE in img_dim_list:
        if version == "1":
            lr_list = [1e-4, 1e-3, 1e-6]
        else:
            lr_list = [1e-4]
        for lr in lr_list:
            if lr_list == 1e-3:
                HIDDEN_DIM_DISCR = 128
                HIDDEN_DIM_GEN = 128
            elif lr_list == 1e-4:
                HIDDEN_DIM_DISCR = 64
                HIDDEN_DIM_GEN = 64
            elif lr_list == 1e-5:
                HIDDEN_DIM_DISCR = 64
                HIDDEN_DIM_GEN = 64
            if version == "3":
                hidden_dim_list = [64, 32, 128]
            else:
                hidden_dim_list = [64]
            for hidden_dim in hidden_dim_list:
                if version == "4" or lr == 1e-4 and hidden_dim == 64 and DIM_IMAGE == 64:
                    path = path_baseline
                    HIDDEN_DIM_DISCR = 64
                    HIDDEN_DIM_GEN = 64
                if version == "1" and lr == 1e-5:
                    path = path_mod_1
                elif version == "1" and lr == 1e-3:
                    path = path_mod_2
                elif version == "2" and DIM_IMAGE == 32:
                    path = path_mod_1
                elif version == "2" and DIM_IMAGE == 128:
                    path = path_mod_2
                elif version == "3" and hidden_dim == 32:
                    HIDDEN_DIM_DISCR = 32
                    HIDDEN_DIM_GEN = 32
                    path = path_mod_1
                elif version == "3" and hidden_dim == 128:
                    HIDDEN_DIM_DISCR = 128
                    HIDDEN_DIM_GEN = 128
                    path = path_mod_2
                # Define the transformation for the images (resize, grayscale, from PIL Image to tensor, normalize with mean 0,5 and standard deviation 0,5)
                transform = transforms.Compose([
                    transforms.Resize((DIM_IMAGE)),  
                    transforms.Grayscale(num_output_channels=1),  
                    transforms.ToTensor(),  
                    transforms.Normalize((0.5,), (0.5,))
                ])

                # Create instance of dataset
                training_dataset = LoadImages(directory_of_images, location_of_txt_file, transform)

                # Create DataLoader
                loader_train = torch.utils.data.DataLoader(training_dataset, batch_size=128, shuffle=True)


                # Create the generator
                Gen_net = Generator(INPUT_NOISE_VECTOR, HIDDEN_DIM_GEN, RGB_CHANNEL, DIM_IMAGE).to(device)
                # Initialize the weights of the generator
                Gen_net.apply(weights_init)
                print(Gen_net)

                # Create the discriminator
                Disc_net = Disc(RGB_CHANNEL, HIDDEN_DIM_DISCR, DIM_IMAGE).to(device)
                # Initialize the weights of the discriminator
                Disc_net.apply(weights_init)
                print(Disc_net)

                # Initialize BCELoss function
                BCE_crit = nn.BCELoss()

                # Create batch of latent vectors
                Create_noise = torch.randn(BATCH_SIZE, INPUT_NOISE_VECTOR, 1, 1, device=device)

                # Setup Adam optimizers to help with the training of the GAN to stabilize the training
                D_optim = optim.Adam(Disc_net.parameters(), lr=lr, betas=(0.5, 0.999))
                G_optim = optim.Adam(Gen_net.parameters(), lr=lr, betas=(0.5, 0.999))

                # Lists to keep track of progress
                list_of_images = []
                loss_of_gen = []
                loss_of_disc = []
                iterations = 0

                # Create a fixed batch of latent vectors 1 to track the progression
                noise_for_progression = torch.randn(128, INPUT_NOISE_VECTOR, 1, 1, device=device)

                print("Starting Training Loop...")

                # For each epoch
                for epoch in range(EPOCH_NUM):
                    # For each batch in the dataloader
                    for i, data in enumerate(loader_train, 0):

                        # Step 1: feed the discriminator real images and update the discriminator
                        # To ensure that the discriminator is updated with real images, the gradients of the discriminator are zeroed out using Disc_net.zero_grad(). 
                        # This is done to prevent the gradients from accumulating between batches.
                        Disc_net.zero_grad()

                        # Moves images to the GPU or CPU
                        moving_images = data.to(device)
                        # Get the batch size (dimension 0 of the tensor is the batch size)
                        batch_size = moving_images.size(0)
                        # Creates a tensor of batch size filled with 1's as that is the label for real images. 
                        # Consists of labels of the real images, this tensor is used as target for the discriminator.
                        label = torch.full((batch_size,), REAL_LABEL, dtype=torch.float, device=device)
                        # Forward real image batch by using the discriminator 
                        output = Disc_net(moving_images).view(-1)
                        # Calculate current loss on all-real batch by discriminator
                        real_error_disc = BCE_crit(output, label)
                        # Calculate Discriminator gradient by backpropagating through the network
                        real_error_disc.backward()
                        # Calculate the average of the predictions of the discriminator over real images
                        avg_disc_pred_real = output.mean().item()

                        # Step 2: feed the discriminator fake images and update the discriminator
                        # Generate noise so that the GAN can generate fake images
                        latent_vectors = torch.randn(batch_size, INPUT_NOISE_VECTOR, 1, 1, device=device)
                        # Generate fake images with the generator by using the noise
                        fake_img = Gen_net(latent_vectors)
                        label.fill_(FAKE_LABEL)
                        # Use the discriminator to identify the fake images
                        class_output = Disc_net(fake_img.detach()).view(-1)
                        # Calculate the loss of the discriminator on the fake images by using the BCE loss function
                        Disc_loss_fake = BCE_crit(class_output, label)
                        # Calculate Discriminator gradient by backpropagating through the network 
                        Disc_loss_fake.backward()
                        # Calculate the average of the predictions of the discriminator over fake images
                        D_G_z1 = class_output.mean().item()
                        # Calculate the complete discriminator error
                        errD = real_error_disc + Disc_loss_fake
                        # Update the discriminator with the new gradients from the backpropagation

                        D_optim.step()

                        # Step 3: feed the discriminator fake images and update the generator
                        # Reset the gradients of the generator
                        Gen_net.zero_grad()
                        # To calculate the loss of the generator, the discriminator is used to identify the fake images generated by the generator.
                        # So it is assumed that the fake images are real images, and the discriminator is used to identify them.
                        label.fill_(REAL_LABEL)
                        # Perform forward pass
                        output = Disc_net(fake_img).view(-1)
                        # Calculate the generators loss by using the BCE loss function
                        GEN_error = BCE_crit(output, label)
                        # Add diversity loss, this has been halved to improve performance
                        GEN_error += 0.5 * diversity_loss(fake_img)
                        # Calculate Generator gradient by backpropagating through the network
                        GEN_error.backward()
                        # Calculate the average of the predictions of the discriminator over fake images
                        D_G_z2 = output.mean().item()
                        # Update the generator with the new gradients from the backpropagation
                        G_optim.step()

                        # print training statistics
                        if i % 50 == 0:
                            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                                % (epoch, EPOCH_NUM, i, len(loader_train),
                                    errD.item(), GEN_error.item(), avg_disc_pred_real, D_G_z1, D_G_z2))

                        # Measure generator performance by saving the generators output on fixed noise
                        if (iterations % 500 == 0) or ((epoch == EPOCH_NUM-1) and (i == len(loader_train)-1)):
                            with torch.no_grad():
                                fake = Gen_net(Create_noise).detach().cpu()
                            list_of_images.append(vutils.make_grid(fake, padding=2, normalize=True))

                        # Generate images if the epoch is coming to an end
                        if (i + 1) == len(loader_train):
                            with torch.no_grad():
                                fake = Gen_net(noise_for_progression).detach().cpu()
                            list_of_images.append(vutils.make_grid(fake, padding=2, normalize=True))

                            # Saving images
                            vutils.save_image(fake, str(path) + '/output_'+ str(epoch) +'.png', normalize=True)

                        iterations += 1

                    dictionairy = Disc_net.state_dict()
                    torch.save(dictionairy, str(path) + '/discriminator_'+ str(epoch) +'lr ' + str(lr) + 'img '+str(DIM_IMAGE)+'.pth')
                    torch.save(Gen_net.state_dict(), str(path) + '/generator_'+ str(epoch) +'lr ' + str(lr) + 'img '+str(DIM_IMAGE)+'.pth')


def weights_init(m):
    # Custom weights initialization called on netG and netD to help the models converge
    classname = m.__class__.__name__
    # Initialize the weights of the convolutional and batch normalization layers
    if 'Conv' in classname:
        m.weight.data.normal_(mean=0.0, std=0.03)
    elif 'BatchNorm' in classname:
        m.weight.data.normal_(mean=1.0, std=0.03)
        m.bias.data.zero_()

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


class Generator(torch.nn.Module):
    def __init__(self, INPUT_NOISE_VECTOR, HIDDEN_DIM_GEN, RGB_CHANNEL, img_size):
        torch.nn.Module.__init__(self)
        if img_size == 128:
            self.net = torch.nn.Sequential(
                # # input layer
                # # This line creates a convolutional layer it is made to upsample a tensor. The parameters of this layer are:
                torch.nn.ConvTranspose2d(INPUT_NOISE_VECTOR, HIDDEN_DIM_GEN * 16, 4, 1, 0, bias=False),
                # # INPUT_NOISE_VECTOR: means the amount of channels the input has. This is based on the input noise vector size. 
                # # HIDDEN_DIM_GEN * 8: The size of feature maps that will be used in the network.
                # # 4: Kernel size the dimensions of the convolutional window.
                # # 1: The step size for moving the kernel across the input tensor.
                # # 0: The padding. number of pixels added to the input tensor on each side.
                # # bias=False: Removes additive bias, helps with flexibility and pattern recognition of the model.
                # # nn.BatchNorm2d(G_HIDDEN * 8): normalizes the output of the previous layer to have a mean of 0 and a standard deviation of 1.
                torch.nn.BatchNorm2d(HIDDEN_DIM_GEN * 16),
                # # nn.LeakyReLU(0.2, inplace=True): Leaky ReLU activation function. It is used to introduce non-linearity to the model, it helped with stabilizing the GAN versus a ReLU activation function.
                torch.nn.LeakyReLU(0.2, inplace=True),

                torch.nn.ConvTranspose2d(HIDDEN_DIM_GEN * 16, HIDDEN_DIM_GEN * 8, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(HIDDEN_DIM_GEN * 8),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # hidden layer 1
                torch.nn.ConvTranspose2d(HIDDEN_DIM_GEN * 8, HIDDEN_DIM_GEN * 4, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(HIDDEN_DIM_GEN * 4),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # hidden layer 2
                torch.nn.ConvTranspose2d(HIDDEN_DIM_GEN * 4, HIDDEN_DIM_GEN * 2, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(HIDDEN_DIM_GEN * 2),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # hidden layer 3
                torch.nn.ConvTranspose2d(HIDDEN_DIM_GEN * 2, HIDDEN_DIM_GEN, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(HIDDEN_DIM_GEN),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # ending in a Tanh activation function, tanh squashes the output to be between -1 and 1 which is the range of the real images.
                # therefore the tanh is a good choice for the output layer of the generator.
                torch.nn.ConvTranspose2d(HIDDEN_DIM_GEN, RGB_CHANNEL, 4, 2, 1, bias=False),
                torch.nn.Tanh()
            )
        if img_size == 64:
            self.net = torch.nn.Sequential(
                # input layer
                torch.nn.ConvTranspose2d(INPUT_NOISE_VECTOR, HIDDEN_DIM_GEN * 8, 4, 1, 0, bias=False),
                torch.nn.BatchNorm2d(HIDDEN_DIM_GEN * 8),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # hidden layer 1
                torch.nn.ConvTranspose2d(HIDDEN_DIM_GEN * 8, HIDDEN_DIM_GEN * 4, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(HIDDEN_DIM_GEN * 4),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # hidden layer 2
                torch.nn.ConvTranspose2d(HIDDEN_DIM_GEN * 4, HIDDEN_DIM_GEN * 2, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(HIDDEN_DIM_GEN * 2),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # hidden layer 3
                torch.nn.ConvTranspose2d(HIDDEN_DIM_GEN * 2, HIDDEN_DIM_GEN, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(HIDDEN_DIM_GEN),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # output layer
                torch.nn.ConvTranspose2d(HIDDEN_DIM_GEN, RGB_CHANNEL, 4, 2, 1, bias=False),
                torch.nn.Tanh()
            )
        if img_size == 32:
            self.net = torch.nn.Sequential(
                # input layer
                torch.nn.ConvTranspose2d(INPUT_NOISE_VECTOR, HIDDEN_DIM_GEN * 4, 4, 1, 0, bias=False),
                torch.nn.BatchNorm2d(HIDDEN_DIM_GEN * 4),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # hidden layer 1
                torch.nn.ConvTranspose2d(HIDDEN_DIM_GEN * 4, HIDDEN_DIM_GEN * 2, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(HIDDEN_DIM_GEN * 2),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # hidden layer 2
                torch.nn.ConvTranspose2d(HIDDEN_DIM_GEN * 2, HIDDEN_DIM_GEN, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(HIDDEN_DIM_GEN),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # output layer
                torch.nn.ConvTranspose2d(HIDDEN_DIM_GEN, RGB_CHANNEL, 4, 2, 1, bias=False),
                torch.nn.Tanh()
            )
    def forward(self, input):
        return self.net(input)

class Disc(torch.nn.Module):
    def __init__(self, RGB_CHANNEL, HIDDEN_DIM_DISCR, resolution):
        torch.nn.Module.__init__(self)
        if resolution == 64:
            self.net = torch.nn.Sequential(
                # first layer
                torch.nn.Conv2d(RGB_CHANNEL, HIDDEN_DIM_DISCR, 4, 2, 1, bias=False),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # second layer
                torch.nn.Conv2d(HIDDEN_DIM_DISCR, HIDDEN_DIM_DISCR * 2, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(HIDDEN_DIM_DISCR * 2),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # third layer
                torch.nn.Conv2d(HIDDEN_DIM_DISCR * 2, HIDDEN_DIM_DISCR * 4, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(HIDDEN_DIM_DISCR * 4),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # fourth layer
                torch.nn.Conv2d(HIDDEN_DIM_DISCR * 4, HIDDEN_DIM_DISCR * 8, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(HIDDEN_DIM_DISCR * 8),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # Sigmoid is for binary classification problems, as the output is between 0 and 1. 1 = real 0 = fake.
                torch.nn.Conv2d(HIDDEN_DIM_DISCR * 8, 1, 4, 1, 0, bias=False),
                torch.nn.Sigmoid()
            )
        elif resolution == 128:
            self.net = torch.nn.Sequential(
                # first layer
                torch.nn.Conv2d(RGB_CHANNEL, HIDDEN_DIM_DISCR, 4, 2, 1, bias=False),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # second layer
                torch.nn.Conv2d(HIDDEN_DIM_DISCR, HIDDEN_DIM_DISCR * 2, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(HIDDEN_DIM_DISCR * 2),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # third layer
                torch.nn.Conv2d(HIDDEN_DIM_DISCR * 2, HIDDEN_DIM_DISCR * 4, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(HIDDEN_DIM_DISCR * 4),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # fourth layer
                torch.nn.Conv2d(HIDDEN_DIM_DISCR * 4, HIDDEN_DIM_DISCR * 8, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(HIDDEN_DIM_DISCR * 8),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # fifth layer
                torch.nn.Conv2d(HIDDEN_DIM_DISCR * 8, HIDDEN_DIM_DISCR * 16, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(HIDDEN_DIM_DISCR * 16),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # Sigmoid is for binary classification problems, as it squashes the output to be between 0 and 1. 1 = real 0 = fake.
                torch.nn.Conv2d(HIDDEN_DIM_DISCR * 16, 1, 4, 1, 0, bias=False),
                torch.nn.Sigmoid()
            )
        else:
            self.net = torch.nn.Sequential(
                # first layer
                torch.nn.Conv2d(RGB_CHANNEL, HIDDEN_DIM_DISCR, 4, 2, 1, bias=False),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # second layer
                torch.nn.Conv2d(HIDDEN_DIM_DISCR, HIDDEN_DIM_DISCR * 2, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(HIDDEN_DIM_DISCR * 2),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # third layer
                torch.nn.Conv2d(HIDDEN_DIM_DISCR * 2, HIDDEN_DIM_DISCR * 4, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(HIDDEN_DIM_DISCR * 4),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # Sigmoid is for binary classification problems, as it squashes the output to be between 0 and 1. 1 = real 0 = fake.
                torch.nn.Conv2d(HIDDEN_DIM_DISCR * 4, 1, 4, 1, 0, bias=False),
                torch.nn.Sigmoid()
            )

    def forward(self, obj):
        return self.net(obj).view(-1, 1).squeeze(1)
if __name__ == '__main__':
    main()
