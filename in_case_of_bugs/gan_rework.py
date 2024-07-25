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
    # Number of hidden layers in the generator
    HIDDEN_LAYERS_GEN = 128
    # Dimensions of the image
    DIM_IMAGE = 64
    # Number of hidden layers in the discriminator
    HIDDEN_LAYERS_DISCR = 128
    # Number of epochs to train the GAN
    EPOCH_NUM = 10
    # Real and fake labels
    REAL_LABEL = 1
    FAKE_LABEL = 0
    # Learning rate for the GAN (for both discriminator and generator)
    lr_gan = 1e-4
    lr_disc = 1e-4
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


    # Change the directory to the data folder (needed for GPU server Leiden)
    # os.chdir('..')
    # os.chdir('..')
    # Define the directory of images and the location of the text file with all image names for training
    directory_of_images =r'NIH_data/images'
    location_of_txt_file =r'NIH_data/train_val_list.txt'
    # directory_of_images =r'images'
    # location_of_txt_file =r'train_val_list.txt'

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
    Gen_net = Generator(INPUT_NOISE_VECTOR, HIDDEN_LAYERS_GEN, RGB_CHANNEL).to(device)
    # Initialize the weights of the generator
    Gen_net.apply(weights_init)
    print(Gen_net)

    # Create the discriminator
    Disc_net = Discriminator(RGB_CHANNEL, HIDDEN_LAYERS_DISCR).to(device)
    # Initialize the weights of the discriminator
    Disc_net.apply(weights_init)
    print(Disc_net)

    # # Load discriminator
    # discriminator_path = r'/data/s3287297/GAN_baseline/discriminator_59.pth'
    # discriminator_state_dict = torch.load(discriminator_path)
    # # Assuming you have an instance of your discriminator model
    # Disc_net.load_state_dict(discriminator_state_dict)

    # # Load generator
    # generator_path = r'/data/s3287297/GAN_baseline/generator_59.pth'
    # generator_state_dict = torch.load(generator_path)
    # Gen_net.load_state_dict(generator_state_dict)

    # print("Continuing from epoch 59...")

    # Initialize BCELoss function
    BCE_crit = nn.BCELoss()

    # Create batch of latent vectors
    Create_noise = torch.randn(BATCH_SIZE, INPUT_NOISE_VECTOR, 1, 1, device=device)

    # Setup Adam optimizers to help with the training of the GAN to stabilize the training
    D_optim = optim.Adam(Disc_net.parameters(), lr=lr_gan, betas=(0.5, 0.999))
    G_optim = optim.Adam(Gen_net.parameters(), lr=lr_disc, betas=(0.5, 0.999))

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
                vutils.save_image(fake, r'GAN_hidden_layers_128/images/output_'+ str(epoch) +'.png', normalize=True)

            iterations += 1

        dictionairy = Disc_net.state_dict()
        torch.save(dictionairy, r'GAN_hidden_layers_128/discriminator_'+ str(epoch) +'.pth')
        torch.save(Gen_net.state_dict(), r'GAN_hidden_layers_128/generator_'+ str(epoch) +'.pth')
    # # Grab real images from the dataset
    # real_batch = next(iter(loader_train))


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


class Generator(nn.Module):
    def __init__(self, INPUT_NOISE_VECTOR, HIDDEN_LAYERS_GEN, RGB_CHANNEL):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # # input layer
            # # This line creates a convolutional layer it is made to upsample a tensor. The parameters of this layer are:
            # nn.ConvTranspose2d(INPUT_NOISE_VECTOR, HIDDEN_LAYERS_GEN * 16, 4, 1, 0, bias=False),
            # # Z_DIM: means the amount of channels the input has. This is based on the input noise vector size. 
            # # HIDDEN_LAYERS_GEN * 8: The number of feature maps that will be produced.
            # # 4: Kernel size the dimensions of the convolutional window.
            # # 1: The step size for moving the kernel across the input tensor.
            # # 0: The padding. number of pixels added to the input tensor on each side.
            # # bias=False: Removes additive bias, helps with flexibility and pattern recognition of the model.
            # # nn.BatchNorm2d(G_HIDDEN * 8): normalizes the output of the previous layer to have a mean of 0 and a standard deviation of 1.
            # nn.BatchNorm2d(HIDDEN_LAYERS_GEN * 16),
            # # nn.LeakyReLU(0.2, inplace=True): Leaky ReLU activation function. It is used to introduce non-linearity to the model, it helped with stabilizing the GAN versus a ReLU activation function.
            # nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(INPUT_NOISE_VECTOR, HIDDEN_LAYERS_GEN * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(HIDDEN_LAYERS_GEN * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # # hidden layer 1
            nn.ConvTranspose2d(HIDDEN_LAYERS_GEN * 8, HIDDEN_LAYERS_GEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(HIDDEN_LAYERS_GEN * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # hidden layer 2
            nn.ConvTranspose2d(HIDDEN_LAYERS_GEN * 4, HIDDEN_LAYERS_GEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(HIDDEN_LAYERS_GEN * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # hidden layer 3
            nn.ConvTranspose2d(HIDDEN_LAYERS_GEN * 2, HIDDEN_LAYERS_GEN, 4, 2, 1, bias=False),
            nn.BatchNorm2d(HIDDEN_LAYERS_GEN),
            nn.LeakyReLU(0.2, inplace=True),
            # output layer
            nn.ConvTranspose2d(HIDDEN_LAYERS_GEN, RGB_CHANNEL, 4, 2, 1, bias=False),
            # ending in a Tanh activation function, tanh squashes the output to be between -1 and 1 which is the range of the real images.
            # therefore the tanh is a good choice for the output layer of the generator.
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, IMAGE_CHANNEL, HIDDEN_LAYERS_DISCR):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 1st layer
            nn.Conv2d(IMAGE_CHANNEL, HIDDEN_LAYERS_DISCR, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 2nd layer
            nn.Conv2d(HIDDEN_LAYERS_DISCR, HIDDEN_LAYERS_DISCR * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(HIDDEN_LAYERS_DISCR * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 3rd layer
            nn.Conv2d(HIDDEN_LAYERS_DISCR * 2, HIDDEN_LAYERS_DISCR * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(HIDDEN_LAYERS_DISCR * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 4th layer
            nn.Conv2d(HIDDEN_LAYERS_DISCR * 4, HIDDEN_LAYERS_DISCR * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(HIDDEN_LAYERS_DISCR * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 5th layer
            # nn.Conv2d(HIDDEN_LAYERS_DISCR * 8, HIDDEN_LAYERS_DISCR * 16, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(HIDDEN_LAYERS_DISCR * 16),
            # nn.ReLU(inplace=True),
            #output layer
            nn.Conv2d(HIDDEN_LAYERS_DISCR * 8, 1, 4, 1, 0, bias=False),
            # Sigmoid is commonly used for binary classification problems, as it squashes the output to be between 0 and 1. 1 = real 0 = fake.
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)
if __name__ == '__main__':
    main()
