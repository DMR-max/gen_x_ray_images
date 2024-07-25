import os
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import diffusers
import accelerate
from huggingface_hub import HfFolder, Repository, whoami
import torch
import torch.nn.functional as F
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
    
class Generator(torch.nn.Module):
    def __init__(self, INPUT_NOISE_VECTOR, GEN_FEATURE_MAP_SIZE, RGB_CHANNEL, img_size):
        torch.nn.Module.__init__()
        if img_size == 128:
            self.net = torch.nn.Sequential(
                # # input layer
                # # This line creates a convolutional layer it is made to upsample a tensor. The parameters of this layer are:
                torch.nn.ConvTranspose2d(INPUT_NOISE_VECTOR, GEN_FEATURE_MAP_SIZE * 16, 4, 1, 0, bias=False),
                # # Z_DIM: means the amount of channels the input has. This is based on the input noise vector size. 
                # # GEN_FEATURE_MAP_SIZE * 8: The number of feature maps that will be produced.
                # # 4: Kernel size the dimensions of the convolutional window.
                # # 1: The step size for moving the kernel across the input tensor.
                # # 0: The padding. number of pixels added to the input tensor on each side.
                # # bias=False: Removes additive bias, helps with flexibility and pattern recognition of the model.
                # # nn.BatchNorm2d(G_HIDDEN * 8): normalizes the output of the previous layer to have a mean of 0 and a standard deviation of 1.
                torch.nn.BatchNorm2d(GEN_FEATURE_MAP_SIZE * 16),
                # # nn.LeakyReLU(0.2, inplace=True): Leaky ReLU activation function. It is used to introduce non-linearity to the model, it helped with stabilizing the GAN versus a ReLU activation function.
                torch.nn.LeakyReLU(0.2, inplace=True),
                
                # hidden layer 1
                torch.nn.ConvTranspose2d(GEN_FEATURE_MAP_SIZE * 8, GEN_FEATURE_MAP_SIZE * 4, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(GEN_FEATURE_MAP_SIZE * 4),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # hidden layer 2
                torch.nn.ConvTranspose2d(GEN_FEATURE_MAP_SIZE * 4, GEN_FEATURE_MAP_SIZE * 2, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(GEN_FEATURE_MAP_SIZE * 2),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # hidden layer 3
                torch.nn.ConvTranspose2d(GEN_FEATURE_MAP_SIZE * 2, GEN_FEATURE_MAP_SIZE, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(GEN_FEATURE_MAP_SIZE),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # ending in a Tanh activation function, tanh squashes the output to be between -1 and 1 which is the range of the real images.
                # therefore the tanh is a good choice for the output layer of the generator.
                torch.nn.ConvTranspose2d(GEN_FEATURE_MAP_SIZE, RGB_CHANNEL, 4, 2, 1, bias=False),
                torch.nn.Tanh()
            )
        if img_size == 64:
            self.net = nn.Sequential(
                # input layer
                torch.nn.ConvTranspose2d(INPUT_NOISE_VECTOR, GEN_FEATURE_MAP_SIZE * 8, 4, 1, 0, bias=False),
                torch.nn.BatchNorm2d(GEN_FEATURE_MAP_SIZE * 8),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # hidden layer 1
                torch.nn.ConvTranspose2d(GEN_FEATURE_MAP_SIZE * 8, GEN_FEATURE_MAP_SIZE * 4, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(GEN_FEATURE_MAP_SIZE * 4),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # hidden layer 2
                torch.nn.ConvTranspose2d(GEN_FEATURE_MAP_SIZE * 4, GEN_FEATURE_MAP_SIZE * 2, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(GEN_FEATURE_MAP_SIZE * 2),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # hidden layer 3
                torch.nn.ConvTranspose2d(GEN_FEATURE_MAP_SIZE * 2, GEN_FEATURE_MAP_SIZE, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(GEN_FEATURE_MAP_SIZE),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # output layer
                torch.nn.ConvTranspose2d(GEN_FEATURE_MAP_SIZE, RGB_CHANNEL, 4, 2, 1, bias=False),
                torch.nn.Tanh()
            )
        if img_size == 32:
            self.net = nn.Sequential(
                # input layer
                torch.nn.ConvTranspose2d(INPUT_NOISE_VECTOR, GEN_FEATURE_MAP_SIZE * 4, 4, 1, 0, bias=False),
                torch.nn.BatchNorm2d(GEN_FEATURE_MAP_SIZE * 4),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # hidden layer 1
                torch.nn.ConvTranspose2d(GEN_FEATURE_MAP_SIZE * 4, GEN_FEATURE_MAP_SIZE * 2, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(GEN_FEATURE_MAP_SIZE * 2),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # hidden layer 2
                torch.nn.ConvTranspose2d(GEN_FEATURE_MAP_SIZE * 2, GEN_FEATURE_MAP_SIZE, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(GEN_FEATURE_MAP_SIZE),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # output layer
                torch.nn.ConvTranspose2d(GEN_FEATURE_MAP_SIZE, RGB_CHANNEL, 4, 2, 1, bias=False),
                torch.nn.Tanh()
            )
    def forward(self, input):
        return self.net(input)
    
class Disc(torch.nn.Module):
    def __init__(self, RGB_CHANNEL, DISCR_FEATURE_MAP_SIZE, resolution):
        torch.nn.Module.__init__()
        if resolution == 64:
            self.net = torch.nn.Sequential(
                # first layer
                torch.nn.Conv2d(RGB_CHANNEL, DISCR_FEATURE_MAP_SIZE, 4, 2, 1, bias=False),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # second layer
                torch.nn.Conv2d(DISCR_FEATURE_MAP_SIZE, DISCR_FEATURE_MAP_SIZE * 2, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(DISCR_FEATURE_MAP_SIZE * 2),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # third layer
                torch.nn.Conv2d(DISCR_FEATURE_MAP_SIZE * 2, DISCR_FEATURE_MAP_SIZE * 4, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(DISCR_FEATURE_MAP_SIZE * 4),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # fourth layer
                torch.nn.Conv2d(DISCR_FEATURE_MAP_SIZE * 4, DISCR_FEATURE_MAP_SIZE * 8, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(DISCR_FEATURE_MAP_SIZE * 8),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # Sigmoid is for binary classification problems, as the output is between 0 and 1. 1 = real 0 = fake.
                torch.nn.Conv2d(DISCR_FEATURE_MAP_SIZE * 8, 1, 4, 1, 0, bias=False),
                torch.nn.Sigmoid()
            )
        elif resolution == 128:
            self.net = torch.nn.Sequential(
                # first layer
                torch.nn.Conv2d(RGB_CHANNEL, DISCR_FEATURE_MAP_SIZE, 4, 2, 1, bias=False),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # second layer
                torch.nn.Conv2d(DISCR_FEATURE_MAP_SIZE, DISCR_FEATURE_MAP_SIZE * 2, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(DISCR_FEATURE_MAP_SIZE * 2),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # third layer
                torch.nn.Conv2d(DISCR_FEATURE_MAP_SIZE * 2, DISCR_FEATURE_MAP_SIZE * 4, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(DISCR_FEATURE_MAP_SIZE * 4),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # fourth layer
                torch.nn.Conv2d(DISCR_FEATURE_MAP_SIZE * 4, DISCR_FEATURE_MAP_SIZE * 8, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(DISCR_FEATURE_MAP_SIZE * 8),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # fifth layer
                torch.nn.Conv2d(DISCR_FEATURE_MAP_SIZE * 8, DISCR_FEATURE_MAP_SIZE * 16, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(DISCR_FEATURE_MAP_SIZE * 16),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # Sigmoid is for binary classification problems, as it squashes the output to be between 0 and 1. 1 = real 0 = fake.
                torch.nn.Conv2d(DISCR_FEATURE_MAP_SIZE * 16, 1, 4, 1, 0, bias=False),
                torch.nn.Sigmoid()
            )
        else:
            self.net = torch.nn.Sequential(
                # first layer
                torch.nn.Conv2d(RGB_CHANNEL, DISCR_FEATURE_MAP_SIZE, 4, 2, 1, bias=False),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # second layer
                torch.nn.Conv2d(DISCR_FEATURE_MAP_SIZE, DISCR_FEATURE_MAP_SIZE * 2, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(DISCR_FEATURE_MAP_SIZE * 2),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # third layer
                torch.nn.Conv2d(DISCR_FEATURE_MAP_SIZE * 2, DISCR_FEATURE_MAP_SIZE * 4, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(DISCR_FEATURE_MAP_SIZE * 4),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # Sigmoid is for binary classification problems, as it squashes the output to be between 0 and 1. 1 = real 0 = fake.
                torch.nn.Conv2d(DISCR_FEATURE_MAP_SIZE * 4, 1, 4, 1, 0, bias=False),
                torch.nn.Sigmoid()
            )

    def forward(self, obj):
        return self.net(obj).view(-1, 1).squeeze(1)
    
def weights_init(gan_model):
    # Custom weights initialization called on netG and netD to help the models converge
    classname = gan_model.__class__.__name__
    # Initialize the weights of the convolutional and batch normalization layers
    if 'Conv' in classname:
        gan_model.weight.data.normal_(mean=0.0, std=0.03)
    elif 'BatchNorm' in classname:
        gan_model.weight.data.normal_(mean=1.0, std=0.03)
        gan_model.bias.data.zero_()

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


def model(epoch, baseline, dim_img, lr, dev, path):
    if baseline or lr == 5e-4 or lr == 1e-5:
        model = diffusers.UNet2DModel(
                    # Resolution of the images
                    sample_size=dim_img,
                    # Amount of input channels, 1 for greyscale images 
                    in_channels=1,
                    # Amount of output channels, 1 for greyscale images  
                    out_channels=1,
                    # Number of UNet blocks  
                    layers_per_block=2,
                    # Number of output channels for each UNet block (for 64 x 64 and higher 2, lower 1 is recommended)
                    block_out_channels=(dim_img * 2, dim_img * 2, dim_img * 4 , dim_img * 4, dim_img * 8, dim_img * 8), 
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
    else:
        model = diffusers.UNet2DModel(
            # Resolution of the images
            sample_size=dim_img,
            # Amount of input channels, 1 for greyscale images 
            in_channels=1,
            # Amount of output channels, 1 for greyscale images  
            out_channels=1,
            # Number of UNet blocks 
            layers_per_block=2, 
            # Number of output channels for each UNet block (for 64 x 64 and higher 2, lower 1 is recommended) 
            block_out_channels=(dim_img * 2, dim_img * 4, dim_img * 8),  
            down_block_types=(
                # This block downsamples the input
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
            ),
        )
    
    model.eval()
    model.to(dev)
    huggingface_train_package = accelerate.Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=1,
    )
    if baseline:
        # # Load the state dict previously saved
        state_dict = torch.load(str(path) + '/model_weights_stable_diff_epoch_'+ str(epoch) +'.pth')
        print("continuing from baseline epoch " + str(epoch))
    elif dim_img == 32:
        # # Load the state dict previously saved
        state_dict = torch.load(str(path) + '/model_weights_stable_diff_epoch_'+ str(epoch) +'_imgsize_32.pth')
        print("continuing from epoch " + str(epoch))
    elif lr == 5e-4:
        state_dict = torch.load(str(path) + '/model_weights_stable_diff_epoch_'+ str(epoch) +'.pth')
        print("continuing from epoch 5e-4 " + str(epoch))
    elif lr == 1e-5:
        state_dict = torch.load(str(path) + '/model_weights_stable_diff_epoch_'+ str(epoch) +'_learningrate_1e-5.pth')
        print("continuing from epoch 1e-5 " + str(epoch))

    # Generates noise for the diffusion model
    noise_creator = diffusers.DDPMScheduler(num_train_timesteps=1000)

    # Load weights from appropriate model
    model.load_state_dict(state_dict)
    pipeline = diffusers.DDPMPipeline(unet=huggingface_train_package.unwrap_model(model), scheduler=noise_creator)
    # Generate images from the pipeline by using a manual seed
    img = pipeline(
        batch_size=128,
        generator=torch.manual_seed(0),
    ).images
    return img
def main():
    version = input("Which version of the model do you want to make a graph of? 1 = learning rate, 2 = Image size, 3 = Baseline")
    baseline_path = input("what is the checkpoint location of the baseline model?")
    if version == '1' or version == '2':
        input_model_1 = input("what is the checkpoint location of the second model? (for learning rate 5e-4, for image size 32)")
        input_model_2 = input("what is the checkpoint location of the third model? (for learning rate 1e-5)")
    location_of_discriminator = input("What is the location of the discriminator?")
    if version == 2:
        location_of_discriminator_32 = input("What is the location of the (32 x 32) discriminator?")
    # Flag to use CUDA
    CUDA = True
    # training batch size
    B_SIZE = 128
    # Channels in the RGB image (1 channel = grayscale)
    RGB_CHANNEL = 1
    # Size of the input noise vector
    INPUT_NOISE_VECTOR = 100
    # Number of hidden layers in the generator
    HIDDEN_LAYERS_GEN = 64
    # Dimensions of the image
    DIM_IMAGE = 64
    # Number of hidden layers in the discriminator
    HIDDEN_LAYERS_DISCR = 64
    # Number of epochs to train the GAN
    EPOCH_NUM = 50
    # Real and fake labels
    R_LABEL = 1
    F_LABEL = 0
    # Learning rate for the GAN	
    lr = 1e-4


    # Define the device
    if torch.cuda.is_available() and CUDA:
        dev = torch.device("cuda:0")
    else:
        dev = torch.device("cpu")
    print(dev)


    directory_of_images = input("What is the directory of the images? (all together not in different folders)")
    location_of_txt_file = input("What is the location of the txt file with the train value names?")

    if version == 2:
        image_size_list = [32, 64]
    else:
        image_size_list = [64]
    if version == "1":
        lr_list = [1e-4, 5e-4, 1e-5]
    else:
        lr_list = [1e-4]
    for lr in lr_list:
        for image_size in image_size_list:
            DIM_IMAGE = image_size
            # Define the transformation for the images (resize, grayscale, from PIL Image to tensor, normalize with mean 0,5 and standard deviation 0,5)
            transform_params = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize((DIM_IMAGE, DIM_IMAGE)),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.Grayscale(num_output_channels=1),
                    torchvision.transforms.ToTensor(),
                ]
            )

            # Create instance of dataset
            training = LoadImages(directory_of_images, location_of_txt_file, transform_params)

            # Create DataLoaders for training and testing sets
            train_set = torch.utils.data.DataLoader(training, batch_size=B_SIZE, shuffle=True)


            # Lists to keep track of progress
            list_of_images = []
            loss_of_gen = []
            loss_of_disc = []

            print("Starting Testing Loop...")
            list_of_disc_errors_2 = np.zeros(EPOCH_NUM)
            list_of_disc_errors_1 = np.zeros(EPOCH_NUM)
            list_of_disc_errors_baseline = np.zeros(EPOCH_NUM)
            list_of_epochs = np.arange(0, EPOCH_NUM, 1)
            fig, plt1 = plt.subplots()
            plt1.set_xlabel('Epoch')
            plt1.set_ylabel('Discriminator Loss')      
            plt1.set_title("Discriminator Loss vs epoch Diffusion model")
            if version == "1" or version == "2":
                paths = [baseline_path, input_model_1, input_model_2]
            else:
                path = [baseline_path]
            for path in paths:
                for epoch in range(EPOCH_NUM):
                    batch_count = 0
                    for i, data in enumerate(train_set, 0): 
                        with torch.no_grad():
                            # move the images to the device (GPU) 
                            moving_images = data.to(dev)
                            # Calculate the batch size
                            b_size = moving_images.size(0)
                            # Create the labels for the real images
                            label_img = torch.full((b_size,), R_LABEL, dtype=torch.float, device=dev)

                            # Create the discriminator made for the 64x64 images
                            img_size = DIM_IMAGE
                            Disc_net = Disc(RGB_CHANNEL, HIDDEN_LAYERS_DISCR, img_size).to(dev)
                            # Load the weights of the discriminator
                            if DIM_IMAGE == 64:
                                Disc_net.load_state_dict(torch.load(location_of_discriminator))
                            elif DIM_IMAGE == 32:
                                Disc_net.load_state_dict(torch.load(location_of_discriminator_32))
                            # Initialize BCELoss function
                            BCE_crit = nn.BCELoss()

                            # Forward pass real batch through Discriminator with the real images
                            fwd_pass = Disc_net(moving_images).view(-1)
                            real_error_disc = BCE_crit(fwd_pass, label_img)     

                            # Define the resize transformation
                            image_dim = DIM_IMAGE
                            # Create the discriminator made for the 64x64 images
                            img_size = DIM_IMAGE
                            Disc_net = Disc(RGB_CHANNEL, HIDDEN_LAYERS_DISCR, img_size).to(dev)
                            # Load the weights of the discriminator
                            if DIM_IMAGE == 64:
                                Disc_net.load_state_dict(torch.load(location_of_discriminator))
                            elif DIM_IMAGE == 32:
                                Disc_net.load_state_dict(torch.load(location_of_discriminator_32))
                            # Generate the fake images from the diffusion model
                            if path == baseline_path:
                                fake = model(epoch, True, image_dim, lr, dev, path)
                            else:
                                fake = model(epoch, False, image_dim, lr, dev, path)


                            label_img.fill_(F_LABEL)
                            # Create a batch of 128 images
                            cutout = fake[0:128]
                            # Create a list of the fake images
                            list_fake_img = []
                            # Transform the fake images to tensors and move them to the device
                            for img in cutout:
                                fake_one_img = torchvision.transforms.ToTensor()(img).to(dev)
                                list_fake_img.append(fake_one_img)
                            # Stack the fake images 
                            list_fake_img = torch.stack(list_fake_img)
                            
                            # Forward pass batch through Discriminator with the fake images
                            fwd_pass = Disc_net(list_fake_img.detach()).view(-1)
                            # The discriminator loss on the all-fake batch
                            Disc_loss_fake = BCE_crit(fwd_pass, label_img)
                            # The mean of the output
                            average_pred_fake = fwd_pass.mean().item()
                            # Calculate the total discriminator error
                            errD = real_error_disc + Disc_loss_fake
                            if path == baseline_path:
                                list_of_disc_errors_baseline[epoch] = errD.item()
                            elif path == input_model_1:
                                list_of_disc_errors_1[epoch] = errD.item()
                            else:
                                list_of_disc_errors_2[epoch] = errD.item()
                            print(errD.item())

                            break
                if path == baseline_path:
                    plt1.plot(list_of_epochs,list_of_disc_errors_baseline,label="Discriminator Loss of the baseline")
                elif path == input_model_1 and version == "1":
                    plt1.plot(list_of_epochs,list_of_disc_errors_1,label="Discriminator Loss of the autoreg model with lr = 5e-4")
                elif path == input_model_1 and version == "2":
                    plt1.plot(list_of_epochs,list_of_disc_errors_1,label="Discriminator Loss of the autoreg model with image size = 32")
                elif path == input_model_2 and version == "1":
                    plt1.plot(list_of_epochs,list_of_disc_errors_2,label="Discriminator Loss of the autoreg model with lr = 1e-5")

        plt1.legend()
        if version == "1":
            fig.savefig("Discriminator_vs_autoreg_lr.png",dpi=300)
        if version == "2":
            fig.savefig("Discriminator_vs_autoreg_image_size.png",dpi=300)
        if version == "3":
            fig.savefig("Discriminator_vs_autoreg_baseline.png",dpi=300)

if __name__ == "__main__":
    main()