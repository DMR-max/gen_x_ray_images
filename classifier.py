import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import sys
import configparser
import diffusers
import accelerate

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
class model_pixel(torch.nn.Module):
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

# for loading in the training images
def load_image_paths(folder_path):
    img_paths = []
    for root, dirs, files in os.walk(folder_path):
        for curr_img in files:
            if curr_img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img_paths.append(os.path.join(root, curr_img))
    return img_paths


# class to define the dataset
class XrayDataset(Dataset):
    def __init__(self, image_pth, labels, transform=None):
        # Set the image paths
        self.image_pth = image_pth
        # Set the labels
        self.labels = labels
        # Set the transform
        self.trans = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Set the current image path
        curr_image_pth = self.image_pth[idx]
        # Open the image and convert to grayscale
        image = Image.open(curr_image_pth).convert('L')
        # Set the label
        label = self.labels[idx]
        # Apply the transformations
        if self.trans:
            image = self.trans(image)
        # Return the image and label
        return image, torch.tensor(label, dtype=torch.float32)



class SimpleCNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleCNN, self).__init__()
        # layers to extract features
        self.Convolutional_first_layer = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.Convolutional_second_layer = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the feature maps after the conv and pooling layers
        if input_size == 32:
            fc_input_dim = 64 * 8 * 8
        elif input_size == 64:
            fc_input_dim = 64 * 16 * 16
        else:
            raise ValueError("Unsupported input size. Only 32x32 and 64x64 are supported.")

        # layers to classify the features
        self.lin_first = nn.Linear(fc_input_dim, 128)
        self.lin_second = nn.Linear(128, 1)

    def forward(self, input):
        input = self.pool(F.relu(self.Convolutional_first_layer(input)))
        input = self.pool(F.relu(self.Convolutional_second_layer(input)))
        input = input.view(input.size(0), -1)
        input = F.relu(self.lin_first(input))
        # Sigmoid activation function to squash the output between 0 and 1
        input = torch.sigmoid(self.lin_second(input))
        return input

def train(model, device, train_loader, adam_optim, criterion, epoch):
    # Set the model to training mode
    model.train()
    # Iterate over the training data
    for index, (image, label) in enumerate(train_loader):
        # Move the data to the device
        image = image.to(device)
        label = label.to(device)
        # Zero the gradients
        adam_optim.zero_grad()
        # Forward pass
        output = model(image).squeeze()
        # Calculate the loss
        loss_BCE = criterion(output, label)
        # Backward pass
        loss_BCE.backward()
        # Update the weights
        adam_optim.step()
        # Print the progress
        print("batch: ", index, "loss: ", loss_BCE.item())


def score_generated_images(class_model, DEVICE, INPUT_NOISE_VECTOR, RGB_CHANNEL, hidden_dim, latent_dim, batch_size, path_class_64, path_class_32, path_diff, path_VAE, path_GAN, path_autoreg, path_save):
    class_model.eval()
    scores = []
    EPOCH_NUM = 25
    print("Starting Testing Loop...")
    # Define the list of scores
    list_of_disc_errors_autoreg = np.zeros(EPOCH_NUM)
    list_of_disc_errors_VAE = np.zeros(EPOCH_NUM)
    list_of_disc_errors_GAN = np.zeros(EPOCH_NUM)
    list_of_disc_errors_diff = np.zeros(EPOCH_NUM)
    list_of_epochs = np.arange(0, EPOCH_NUM, 1)
    fig, plt1 = plt.subplots()
    plt1.set_xlabel('Epoch')
    plt1.set_ylabel('Classifier score')      
    plt1.set_title("Classifier score vs epoch all models")
    for epoch in range(EPOCH_NUM):

        with torch.no_grad():
            input_size = 32
            # Load the classification model
            class_model = SimpleCNN(input_size=input_size).to(DEVICE)
            class_model.load_state_dict(torch.load(str(path_class_32)))
            # Define the resize transformation
            image_dim = 32
            img_size = 32
            lr = 1e-4
            # Load the diffusion model
            fake = model_diff(epoch, False, image_dim, lr, DEVICE, path_diff)

            # Take the batch of fake images
            cutout = fake[0:128]
            list_fake_img = []
            # Transform the images to tensors
            for img in cutout:
                fake_one_img = transforms.ToTensor()(img).to(DEVICE)
                list_fake_img.append(fake_one_img)
            # Stack the images 
            list_fake_img = torch.stack(list_fake_img)
            # Classify the fake images
            list_of_disc_errors_diff[epoch] = class_model(list_fake_img).mean().item()

            # Load the GAN model
            input_size = 64
            class_model = SimpleCNN(input_size=input_size).to(DEVICE)
            class_model.load_state_dict(torch.load(str(path_class_64)))
            image_dim = 64
            img_size = 64
            HIDDEN_LAYERS_GEN = 64
            Gen_net = Generator(INPUT_NOISE_VECTOR, HIDDEN_LAYERS_GEN, RGB_CHANNEL).to(DEVICE)
            Gen_net.apply(weights_init)
            generator_path = str(path_GAN) +'/generator_'+ str(epoch) +'lr ' + str(lr) + 'img '+str(image_dim)+'.pth'
            generator_state_dict = torch.load(generator_path)
            Gen_net.load_state_dict(generator_state_dict)

            print("Continuing from epoch " + str(epoch) + "...")
            # Generate fake images
            latent_vectors = torch.randn(128, INPUT_NOISE_VECTOR, 1, 1, device=DEVICE)
            fake_img = Gen_net(latent_vectors)
            # Classify the fake images from the GAN
            list_of_disc_errors_GAN[epoch] = class_model(fake_img).mean().item()


        with torch.no_grad():
            input_size = 32
            class_model = SimpleCNN(input_size=input_size).to(DEVICE)
            class_model.load_state_dict(torch.load(str(path_class_32)))
            X_DIM = 32
            x_dim = 32*32
            img_size = 32
            encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
            decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)

            model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)
            model.load_state_dict(torch.load(str(path_VAE) + '/model_weights_VAE_total_epoch_'+ str(epoch+1) +'_.pth'))
            noise = torch.randn(batch_size, latent_dim).to(DEVICE)
            fake = decoder(noise)


            fake = fake.view(batch_size, 1, X_DIM, X_DIM)


            list_of_disc_errors_VAE[epoch] = class_model(fake).mean().item()


            input_size = 64
            class_model = SimpleCNN(input_size=input_size).to(DEVICE)
            class_model.load_state_dict(torch.load(str(path_class_64)))
            
            load_path = path_autoreg + '/Model_Checkpoint_'+str(epoch)+ '_lr='+str(lr)+ 'img_dim='+str(input_size)+'.pt'
            if EPOCH_NUM == 24:
                load_path = path_autoreg + '/Model_Checkpoint_'+'Last'+ '_lr='+str(lr)+ 'img_dim='+str(input_size)+'.pt'
            assert os.path.exists(load_path), 'Saved Model File Does not exist!'
            no_images = 128
            images_size = 64
            images_channels = 1
            

            #Define and load model
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            net = model_pixel().to(device)
            net.load_state_dict(torch.load(load_path))
            net.eval()



            gen_img_tensor = torch.Tensor(no_images, images_channels, images_size, images_size).to(device)
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
            list_of_disc_errors_autoreg[epoch] = class_model(gen_img_tensor).mean().item()
    plt1.plot(list_of_epochs,list_of_disc_errors_autoreg,label="Discriminator Loss of the autoregressive model")
    plt1.plot(list_of_epochs,list_of_disc_errors_diff,label="Discriminator Loss of the diffusion model (32 x 32)")
    plt1.plot(list_of_epochs,list_of_disc_errors_VAE,label="Discriminator Loss of the VAE (32 x 32)")
    plt1.plot(list_of_epochs,list_of_disc_errors_GAN,label="Discriminator Loss of the GAN (baseline)")
    plt1.legend()
    fig.savefig(str(path_save) +"/Classifier_vs_all.png",dpi=300)
    return scores


def main():
    path_class_64 = input("Enter the path to store the classifier model for 64x64 images: ")
    path_class_32 = input("Enter the path to store the classifier model for 32x32 images: ")
    path_diff = input("Enter the path to store the diffusion model: ")
    path_VAE = input("Enter the path to store the VAE model: ")
    path_GAN = input("Enter the path to store the GAN model: ")
    path_auto_reg = input("Enter the path to store the autoregressive model: ")
    path_save = input("Enter the path to save the classifier vs all models graph: ")
    path_real = input("Enter the path to the real images: ")
    path_fake = input("Enter the path to the fake (generated) images: ")  
    CUDA = True
    CUDA = CUDA and torch.cuda.is_available()
    print("PyTorch version: {}".format(torch.__version__))
    if CUDA:
        print("CUDA version: {}\n".format(torch.version.cuda))

    DEVICE = torch.device("cuda:0" if CUDA else "cpu") 
    INPUT_NOISE_VECTOR = 100
    HIDDEN_LAYERS_GEN = 64
    RGB_CHANNEL = 1
    # Create the generator
    Gen_net = Generator(INPUT_NOISE_VECTOR, HIDDEN_LAYERS_GEN, RGB_CHANNEL).to(DEVICE)
    Gen_net.apply(weights_init)
    print(Gen_net)

    BCE_loss = nn.BCELoss()
    batch_size = 128
    X_DIM = 32
    x_dim  = X_DIM * X_DIM
    hidden_dim = 200
    latent_dim = 200
    encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)

    model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)
    # Load images (replace these with your actual data loading code)
    # Specify the directories containing the real and generated images
    real_images_folder = str(path_real) 
    generated_images_folder = str(path_fake) 

    # Get the lists of image paths
    real_image_paths = load_image_paths(real_images_folder)
    generated_image_paths = load_image_paths(generated_images_folder)

    # Output the lists to verify
    print("Real Image Paths:")
    print(real_image_paths)

    print("\nGenerated Image Paths:")
    print(generated_image_paths)

    # Combine the image paths and labels
    image_paths = real_image_paths + generated_image_paths
    labels = [1] * len(real_image_paths) + [0] * len(generated_image_paths)

    # Split into training and test sets
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = XrayDataset(train_paths, train_labels, transform)
    test_dataset = XrayDataset(test_paths, test_labels, transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 64
    model = SimpleCNN(input_size=input_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    for epoch in range(1, 11):
        model_path = str(path_class_64)+'/classifier_model_64_' + str(epoch) + '.pth'
        if not os.path.exists(model_path):
            train(model, device, train_loader, optimizer, criterion, epoch)
            torch.save(model.state_dict(), str(path_class_64)+ 'classifier_model_64_' + str(epoch) + '.pth')

    for epoch in range(1, 11):
        model_path = str(path_class_64)+'classifier_model_32_' + str(epoch) + '.pth'
        if not os.path.exists(model_path):
            input_size = 32
            model = SimpleCNN(input_size=input_size).to(device)
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

            train_dataset = XrayDataset(train_paths, train_labels, transform)
            test_dataset = XrayDataset(test_paths, test_labels, transform)

            train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
            train(model, device, train_loader, optimizer, criterion, epoch)
            torch.save(model.state_dict(), str(path_class_32)+'classifier_model_32_' + str(epoch) + '.pth') 
      # New generated images
    score_generated_images(model, device, INPUT_NOISE_VECTOR, RGB_CHANNEL, hidden_dim, latent_dim, batch_size, path_class_64, path_class_32, path_diff, path_VAE, path_GAN, path_auto_reg, path_save)
