import os
import sys
import tqdm
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
import numpy as np
import diffusers
import accelerate
from huggingface_hub import HfFolder, Repository, whoami
import torch
import torch.nn.functional as F
import configparser
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

# Class to define the masked CNN
class convolutional_NN_msk(torch.nn.Conv2d):
	# Initialize the class
	def __init__(self, msk_type, *arguments, **keywordedarguments):
		# Set the mask type
		self.mask_type = msk_type
		torch.nn.Conv2d.__init__(self,*arguments, **keywordedarguments)
		self.register_buffer('buffer_msk', self.weight.data.clone())

		# Set the mask for the CNN
		_, _, h, w = self.weight.size()
		self.buffer_msk.fill_(1)
		if msk_type =='A':
			self.buffer_msk[:,:,h//2,w//2:] = 0
		else:
			self.buffer_msk[:,:,h//2,w//2+1:] = 0
		self.buffer_msk[:,:,h//2+1:,:] = 0

	# Propagation through the neural network
	def forward(self, input):
		self.weight.data*=self.buffer_msk
		return torch.nn.Conv2d.forward(self,input)

# Class to define the autoregressive CNN
class Model_pixel(torch.nn.Module):
	# Initialize the class
	def __init__(self, layer_numb=8, size_kernel = 7, ch_numb=64, dev=None):
		torch.nn.Module.__init__(self)
		# Set the amount of layers
		self.amount_of_layers = layer_numb
		# Set the kernel size
		self.kernel_size = size_kernel
		# Set the number of channels
		self.ch_numb = ch_numb
		# Set the device
		self.layer = {}
		# Set the device
		self.dev = dev

		# Set the first layer
		# A is so that the model does not have access to the future information the model is trying to predict
		self.Convolutional_first_layer = convolutional_NN_msk('A',1,ch_numb, size_kernel, 1, size_kernel//2, bias=False)
		self.Normalization_first_layer = torch.nn.BatchNorm2d(ch_numb)
		self.Activation_func_first_layer = torch.nn.ReLU(True)

		# Set the second layer
		# B is so that the model does have access to the future information the model is trying to predict
		# Should help in learning more complex patterns
		self.Convolutional_second_layer = convolutional_NN_msk('B',ch_numb,ch_numb, size_kernel, 1, size_kernel//2, bias=False)
		self.Normalization_second_layer = torch.nn.BatchNorm2d(ch_numb)
		self.Activation_func_second_layer = torch.nn.ReLU(True)

		# Set the third layer
		self.Convolutional_third_layer = convolutional_NN_msk('B',ch_numb,ch_numb, size_kernel, 1, size_kernel//2, bias=False)
		self.Normalization_third_layer = torch.nn.BatchNorm2d(ch_numb)
		self.Activation_func_third_layer = torch.nn.ReLU(True)

		# Set the fourth layer
		self.Convolutional_fourth_layer = convolutional_NN_msk('B',ch_numb,ch_numb, size_kernel, 1, size_kernel//2, bias=False)
		self.Normalization_fourth_layer = torch.nn.BatchNorm2d(ch_numb)
		self.Activation_func_fourth_layer = torch.nn.ReLU(True)

		# Set the fifth layer
		self.Convolutional_fifth_layer = convolutional_NN_msk('B',ch_numb,ch_numb, size_kernel, 1, size_kernel//2, bias=False)
		self.Normalization_fifth_layer = torch.nn.BatchNorm2d(ch_numb)
		self.Activation_func_fifth_layer = torch.nn.ReLU(True)

		# Set the sixth layer
		self.Convolutional_sixth_layer = convolutional_NN_msk('B',ch_numb,ch_numb, size_kernel, 1, size_kernel//2, bias=False)
		self.Normalization_sixth_layer = torch.nn.BatchNorm2d(ch_numb)
		self.Activation_func_sixth_layer = torch.nn.ReLU(True)

		# Set the seventh layer
		self.Convolutional_seventh_layer = convolutional_NN_msk('B',ch_numb,ch_numb, size_kernel, 1, size_kernel//2, bias=False)
		self.Normalization_seventh_layer = torch.nn.BatchNorm2d(ch_numb)
		self.Activation_func_seventh_layer = torch.nn.ReLU(True)

		# Set the eighth layer
		self.Convolutional_eight_layer = convolutional_NN_msk('B',ch_numb,ch_numb, size_kernel, 1, size_kernel//2, bias=False)
		self.Normalization_eight_layer = torch.nn.BatchNorm2d(ch_numb)
		self.Activation_func_eight_layer = torch.nn.ReLU(True)

		if self.amount_of_layers == 10:
			# Set the ninth layer
			self.Convolutional_nineth_layer = convolutional_NN_msk('B',ch_numb,ch_numb, size_kernel, 1, size_kernel//2, bias=False)
			self.Normalization_nineth_layer = torch.nn.BatchNorm2d(ch_numb)
			self.Activation_func_nineth_layer = torch.nn.ReLU(True)

			# Set the tenth layer
			self.Convolutional_tenth_layer = convolutional_NN_msk('B',ch_numb,ch_numb, size_kernel, 1, size_kernel//2, bias=False)
			self.Normalization_tenth_layer = torch.nn.BatchNorm2d(ch_numb)
			self.Activation_func_tenth_layer = torch.nn.ReLU(True)

		# Set the output layer
		self.out = torch.nn.Conv2d(ch_numb, 256, 1)

	def forward(self, input):
		# Forward pass of all the layers
		input = self.Convolutional_first_layer(input)
		input = self.Normalization_first_layer(input)
		input = self.Activation_func_first_layer(input)

		input = self.Convolutional_second_layer(input)
		input = self.Normalization_second_layer(input)
		input = self.Activation_func_second_layer(input)

		input = self.Convolutional_third_layer(input)
		input = self.Normalization_third_layer(input)
		input = self.Activation_func_third_layer(input)

		input = self.Convolutional_fourth_layer(input)
		input = self.Normalization_fourth_layer(input)
		input = self.Activation_func_fourth_layer(input)

		input = self.Convolutional_fifth_layer(input)
		input = self.Normalization_fifth_layer(input)
		input = self.Activation_func_fifth_layer(input)

		input = self.Convolutional_sixth_layer(input)
		input = self.Normalization_sixth_layer(input)
		input = self.Activation_func_sixth_layer(input)

		input = self.Convolutional_seventh_layer(input)
		input = self.Normalization_seventh_layer(input)
		input = self.Activation_func_seventh_layer(input)

		input = self.Convolutional_eight_layer(input)
		input = self.Normalization_eight_layer(input)
		input = self.Activation_func_eight_layer(input)

		if self.amount_of_layers == 10:

			input = self.Convolutional_nineth_layer(input)
			input = self.Normalization_nineth_layer(input)
			input = self.Activation_func_nineth_layer(input)

			input = self.Convolutional_tenth_layer(input)
			input = self.Normalization_tenth_layer(input)
			input = self.Activation_func_tenth_layer(input)

		# Output layer
		return self.out(input)



        
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


def model_diff(epoch, baseline, dim_img, lr, dev, path):
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
    path_disc_64 = input("Enter the path to the discriminator for 64x64 images: ")
    path_disc_32 = input("Enter the path to the discriminator for 32x32 images: ")
    path_diff = input("Enter the path to store the diffusion model: ")
    path_VAE = input("Enter the path to store the VAE model: ")
    path_GAN = input("Enter the path to store the GAN model: ")
    path_auto_reg = input("Enter the path to store the autoregressive model: ")
    path_save = input("Enter the path to save the discriminator vs all models graph: ")
    path_real = input("Enter the path to the NIH chest X-ray images: ")
    path_train = input("Enter the path to the NIH chest X-ray train list: ") 
    # os.chdir("..")
    # os.chdir("..")   
    CUDA = True
    DATA_PATH = './data'
    BATCH_SIZE = 128
    IMAGE_CHANNEL = 1
    Z_DIM = 100
    G_HIDDEN = 64
    X_DIM = 64
    HIDDEN_DIM_DISCR = 64
    EPOCH_NUM = 25
    REAL_LABEL = 1
    FAKE_LABEL = 0

    lr = 1e-4
    seed = 1

    CUDA = CUDA and torch.cuda.is_available()
    print("PyTorch version: {}".format(torch.__version__))
    if CUDA:
        print("CUDA version: {}\n".format(torch.version.cuda))

    if CUDA:
        torch.cuda.manual_seed(seed)
    DEVICE = torch.device("cuda:0" if CUDA else "cpu")


    root_dir = path_real
    txt_file = path_train


    # Define the transformation for the images (resize, grayscale, from PIL Image to tensor, normalize with mean 0,5 and standard deviation 0,5)
    transform = transforms.Compose([
        transforms.Resize((X_DIM)),  
        transforms.Grayscale(num_output_channels=1),  
        transforms.ToTensor(),  
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Create custom dataset instance
    custom_dataset = LoadImages(root_dir, txt_file, transform)


    # Define the size of your training and testing sets
    train_size = custom_dataset # 80% of the dataset for training

    # Create DataLoaders for training and testing sets
    train_loader = torch.utils.data.DataLoader(train_size, batch_size=BATCH_SIZE, shuffle=True)

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    # Create the discriminator
    img_size = 64
    Disc_net = Disc(IMAGE_CHANNEL, HIDDEN_DIM_DISCR, img_size).to(DEVICE)
    Disc_net.load_state_dict(torch.load(str(path_disc_64)))
    # Initialize BCELoss function
    BCE_crit = nn.BCELoss()

    # Create batch of latent vectors that I will use to visualize the progression of the generator
    viz_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=DEVICE)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(Disc_net.parameters(), lr=lr, betas=(0.5, 0.999))

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    INPUT_NOISE_VECTOR = 100
    HIDDEN_LAYERS_GEN = 64
    RGB_CHANNEL = 1
    # Create the generator
    Gen_net = Generator(INPUT_NOISE_VECTOR, HIDDEN_LAYERS_GEN, RGB_CHANNEL).to(DEVICE)
    Gen_net.apply(weights_init)
    print(Gen_net)
    batch_size = 128
    X_DIM = 32
    x_dim  = X_DIM * X_DIM
    hidden_dim = 200
    latent_dim = 200
    encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)

    model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)

    BCE_loss = nn.BCELoss()
    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(custom_dataset, batch_size=128, shuffle=True)
    print("Starting Testing Loop...")
    list_of_disc_errors_autoreg = np.zeros(EPOCH_NUM)
    list_of_disc_errors_VAE = np.zeros(EPOCH_NUM)
    list_of_disc_errors_GAN = np.zeros(EPOCH_NUM)
    list_of_disc_errors_diff = np.zeros(EPOCH_NUM)
    list_of_epochs = np.arange(0, EPOCH_NUM, 1)
    fig, plt1 = plt.subplots()
    plt1.set_xlabel('Epoch')
    plt1.set_ylabel('Discriminator Loss')      
    plt1.set_title("Discriminator Loss vs epoch all models")
    for epoch in range(EPOCH_NUM):
        batch_count = 0
        for i, data in enumerate(dataloader, 0): 
            with torch.no_grad(): 
                # Create the discriminator made for the 64x64 images
                Disc_net = Disc(RGB_CHANNEL, HIDDEN_DIM_DISCR, img_size).to(DEVICE)
                # Load the weights of the discriminator
                Disc_net.load_state_dict(torch.load(path_disc_64))
                
                # Load the images and labels 
                moving_images = data.to(DEVICE)
                batch_size = moving_images.size(0)
                label = torch.full((batch_size,), REAL_LABEL, dtype=torch.float, device=DEVICE)
                
                # Forward pass real batch through Discriminator
                output = Disc_net(moving_images).view(-1)
                real_error_disc = BCE_crit(output, label)     
                # Generate batch of latent vectors
                noise = torch.randn(batch_size, INPUT_NOISE_VECTOR, 1, 1, device=DEVICE)

                # Define the resize transformation
                image_dim = 32
                img_size = 32
                # Create the discriminator for the 32x32 images
                Disc_net = Disc(IMAGE_CHANNEL, HIDDEN_DIM_DISCR, img_size).to(DEVICE)
                Disc_net.load_state_dict(torch.load(str(path_disc_32)))
                lr = 1e-4
                fake = model_diff(epoch, False, image_dim, lr)

                # show_image(fake, idx = 0)
                F_LABEL = 0
                label.fill_(F_LABEL)

                # This will extract the region from each image in the batch
                cutout = fake[0:128]
                list_fake_img = []
                for img in cutout:
                    fake_one_img = transforms.ToTensor()(img).to(DEVICE)
                    list_fake_img.append(fake_one_img) 
                list_fake_img = torch.stack(list_fake_img)
                
                output = netD(list_fake_img.detach()).view(-1)
                # Calculate the discriminator loss
                errD_fake = BCE_crit(output, label)
                # Calculate the average of the predictions of the discriminator over fake
                errD = real_error_disc + errD_fake
                list_of_disc_errors_diff[epoch] += errD.item()
                print(errD.item())


                image_dim = 64
                img_size = 64
                netD = Disc(IMAGE_CHANNEL, HIDDEN_DIM_DISCR, img_size).to(DEVICE)
                netD.load_state_dict(torch.load(str(path_disc_64)))
                HIDDEN_DIM_GEN = 64
                Gen_net = Generator(INPUT_NOISE_VECTOR, HIDDEN_DIM_GEN, RGB_CHANNEL).to(DEVICE)
                Gen_net.apply(weights_init)
                generator_path = str(path_GAN) + '/generator_'+ str(epoch) +'lr ' + str(lr) + 'img '+str(X_DIM)+'.pth'
                generator_state_dict = torch.load(generator_path)
                Gen_net.load_state_dict(generator_state_dict)

                print("Continuing from epoch " + str(epoch) + "...")
                # Create the discriminator for the 64x64 images
                latent_vectors = torch.randn(batch_size, INPUT_NOISE_VECTOR, 1, 1, device=DEVICE)
                # Generate fake image batch with generator
                fake_img = Gen_net(latent_vectors)
                label.fill_(FAKE_LABEL)
                # classify all fake batch with discriminator
                class_output = Disc_net(fake_img.detach()).view(-1)
                # Calculate the discriminator loss
                Disc_loss_fake = BCE_crit(class_output, label)
                # Calculate the error of the discriminator
                errD = real_error_disc + Disc_loss_fake

                list_of_disc_errors_GAN[epoch] += errD.item()
                print(errD.item())

            with torch.no_grad():
                X_DIM = 32
                x_dim = 32*32
                img_size = 32
                Disc_net = Disc(IMAGE_CHANNEL, HIDDEN_DIM_DISCR, img_size).to(DEVICE)
                Disc_net.load_state_dict(torch.load(str(path_disc_32)))
                encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
                decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)

                model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)
                model.load_state_dict(torch.load(str(path_VAE)+'/model_weights_VAE_total_epoch_'+ str(epoch+1) +'_.pth'))
                noise = torch.randn(batch_size, latent_dim).to(DEVICE)
                fake = decoder(noise)

                # show_image(fake, idx = 0)
                label.fill_(FAKE_LABEL)
        
                fake = fake.view(batch_size, 1, X_DIM, X_DIM)
                


                print(torch.Tensor.size(fake))
                print(torch.Tensor.size(fake))
                print(torch.Tensor.size(label))
                output = netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                print(output.size())
                print(label.size())
                errD_fake = BCE_crit(output, label)
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = real_error_disc + errD_fake
                list_of_disc_errors_VAE[epoch] = errD.item()
                print(errD.item())

            with torch.no_grad():
                img_size = 64
                Disc_net = Disc(IMAGE_CHANNEL, HIDDEN_DIM_DISCR, img_size).to(DEVICE)
                Disc_net.load_state_dict(torch.load(str(path_disc_64)))
                
                load_path = str(path_auto_reg) + '/Model_Checkpoint_'+ str(epoch) +'.pt'
                if EPOCH_NUM == 24:
                    load_path =  str(path_auto_reg) + '/Model_Checkpoint_Last.pt'
                assert os.path.exists(load_path), 'Saved Model File Does not exist!'
                no_images = 128
                images_size = 64
                images_channels = 1
                

                #Define and load model
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                net = Model_pixel().to(device)
                net.load_state_dict(torch.load(load_path))
                net.eval()



                gen_img_tensor = torch.Tensor(128, images_channels, images_size, images_size).to(dev)
                gen_img_tensor.fill_(0)

                #Generating images pixel by pixel
                for rows in range(images_size):
                    for cols in range(images_size):
                        out = net(gen_img_tensor)
                        probability = F.softmax(out[:,:,rows,cols], dim=-1).data
                        gen_img_tensor[:,:,rows,cols] = torch.multinomial(probability, 1)
                        gen_img_tensor[:,:,rows,cols] = gen_img_tensor[:,:,rows,cols].float()
                        gen_img_tensor[:,:,rows,cols] = gen_img_tensor[:,:,rows,cols] / 255.0


            #Saving images row wise
            # torchvision.utils.save_image(fake, 'auto_reg_epoch_'+ str(epoch)+'.png', nrow=12, padding=0)
            output = Disc_net(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            print(output.size())
            print(label.size())
            errD_fake = BCE_crit(output, label)
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = real_error_disc + errD_fake
            list_of_disc_errors_autoreg[epoch] = errD.item()
            print(epoch)
            print(errD.item())
                
            break
    plt1.plot(list_of_epochs,list_of_disc_errors_autoreg,label="Discriminator Loss of the autoregressive model")
    plt1.plot(list_of_epochs,list_of_disc_errors_diff,label="Discriminator Loss of the diffusion model (32 x 32)")
    plt1.plot(list_of_epochs,list_of_disc_errors_VAE,label="Discriminator Loss of the VAE (32 x 32)")
    plt1.plot(list_of_epochs,list_of_disc_errors_GAN,label="Discriminator Loss of the GAN (baseline)")
    plt1.legend()
    fig.savefig(str(path_save) + "Discriminator_vs_all.png",dpi=300)