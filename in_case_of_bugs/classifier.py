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
os.chdir('..')
os.chdir('..')
CUDA = True
CUDA = CUDA and torch.cuda.is_available()
print("PyTorch version: {}".format(torch.__version__))
if CUDA:
    print("CUDA version: {}\n".format(torch.version.cuda))

DEVICE = torch.device("cuda:0" if CUDA else "cpu")

def parse_config(filename):
    config = configparser.ConfigParser()
    config.read(filename)
    output = {}
    for section in config.sections():
        output[section] = {}
        for key in config[section]:
            val_str = str(config[section][key])
            if(len(val_str)>0):
                val = parse_value_from_string(val_str) 
            else:
                val = None
            print(section, key,val_str, val)
            output[section][key] = val
    return output



def parse_value_from_string(val_str):
    if(is_int(val_str)):
        val = int(val_str)
    elif(is_float(val_str)):
        val = float(val_str)
    elif(is_list(val_str)):
        val = parse_list(val_str)
    elif(is_bool(val_str)):
        val = parse_bool(val_str)
    else:
        val = val_str
    return val

def is_int(val_str):
    start_digit = 0
    if(val_str[0] =='-'):
        start_digit = 1
    flag = True
    for i in range(start_digit, len(val_str)):
        if(str(val_str[i]) < '0' or str(val_str[i]) > '9'):
            flag = False
            break
    return flag

def is_float(val_str):
    flag = False
    if('.' in val_str and len(val_str.split('.'))==2):
        if(is_int(val_str.split('.')[0]) and is_int(val_str.split('.')[1])):
            flag = True
        else:
            flag = False
    elif('e' in val_str and len(val_str.split('e'))==2):
        if(is_int(val_str.split('e')[0]) and is_int(val_str.split('e')[1])):
            flag = True
        else:
            flag = False
    else:
        flag = False
    return flag 

def is_bool(var_str):
    if( var_str=='True' or var_str == 'true' or var_str =='False' or var_str=='false'):
        return True
    else:
        return False

def parse_bool(var_str):
    if(var_str=='True' or var_str == 'true' ):
        return True
    else:
        return False
    
def is_list(val_str):
    if(val_str[0] == '[' and val_str[-1] == ']'):
        return True
    else:
        return False

def parse_list(val_str):
    sub_str = val_str[1:-1]
    splits = sub_str.split(',')
    output = []
    for item in splits:
        item = item.strip()
        if(is_int(item)):
            output.append(int(item))
        elif(is_float(item)):
            output.append(float(item))
        elif(is_bool(item)):
            output.append(parse_bool(item))
        else:
            output.append(item)
    return output 
class MaskedCNN(nn.Conv2d):
	"""
	Implementation of Masked CNN Class as explained in A Oord et. al. 
	Taken from https://github.com/jzbontar/pixelcnn-pytorch
	"""

	def __init__(self, mask_type, *args, **kwargs):
		self.mask_type = mask_type
		assert mask_type in ['A', 'B'], "Unknown Mask Type"
		super(MaskedCNN, self).__init__(*args, **kwargs)
		self.register_buffer('mask', self.weight.data.clone())

		_, depth, height, width = self.weight.size()
		self.mask.fill_(1)
		if mask_type =='A':
			self.mask[:,:,height//2,width//2:] = 0
			self.mask[:,:,height//2+1:,:] = 0
		else:
			self.mask[:,:,height//2,width//2+1:] = 0
			self.mask[:,:,height//2+1:,:] = 0


	def forward(self, x):
		self.weight.data*=self.mask
		return super(MaskedCNN, self).forward(x)
	
class PixelCNN(nn.Module):
	"""
	Network of PixelCNN as described in A Oord et. al. 
	"""
	def __init__(self, no_layers=8, kernel = 7, channels=64, device=None, img_size = 64):
		super(PixelCNN, self).__init__()
		self.no_layers = no_layers
		self.kernel = kernel
		self.channels = channels
		self.layers = {}
		self.device = device
		self.img_size = img_size

		self.Conv2d_1 = MaskedCNN('A',1,channels, kernel, 1, kernel//2, bias=False)
		self.BatchNorm2d_1 = nn.BatchNorm2d(channels)
		self.ReLU_1= nn.ReLU(True)

		self.Conv2d_2 = MaskedCNN('B',channels,channels, kernel, 1, kernel//2, bias=False)
		self.BatchNorm2d_2 = nn.BatchNorm2d(channels)
		self.ReLU_2= nn.ReLU(True)

		self.Conv2d_3 = MaskedCNN('B',channels,channels, kernel, 1, kernel//2, bias=False)
		self.BatchNorm2d_3 = nn.BatchNorm2d(channels)
		self.ReLU_3= nn.ReLU(True)

		self.Conv2d_4 = MaskedCNN('B',channels,channels, kernel, 1, kernel//2, bias=False)
		self.BatchNorm2d_4 = nn.BatchNorm2d(channels)
		self.ReLU_4= nn.ReLU(True)

		self.Conv2d_5 = MaskedCNN('B',channels,channels, kernel, 1, kernel//2, bias=False)
		self.BatchNorm2d_5 = nn.BatchNorm2d(channels)
		self.ReLU_5= nn.ReLU(True)

		self.Conv2d_6 = MaskedCNN('B',channels,channels, kernel, 1, kernel//2, bias=False)
		self.BatchNorm2d_6 = nn.BatchNorm2d(channels)
		self.ReLU_6= nn.ReLU(True)

		self.Conv2d_7 = MaskedCNN('B',channels,channels, kernel, 1, kernel//2, bias=False)
		self.BatchNorm2d_7 = nn.BatchNorm2d(channels)
		self.ReLU_7= nn.ReLU(True)

		self.Conv2d_8 = MaskedCNN('B',channels,channels, kernel, 1, kernel//2, bias=False)
		self.BatchNorm2d_8 = nn.BatchNorm2d(channels)
		self.ReLU_8= nn.ReLU(True)

		if img_size == 128:
			self.Conv2d_9 = MaskedCNN('B',channels,channels, kernel, 1, kernel//2, bias=False)
			self.BatchNorm2d_9 = nn.BatchNorm2d(channels)
			self.ReLU_9= nn.ReLU(True)

			self.Conv2d_10 = MaskedCNN('B',channels,channels, kernel, 1, kernel//2, bias=False)
			self.BatchNorm2d_10 = nn.BatchNorm2d(channels)
			self.ReLU_10= nn.ReLU(True)

		self.out = nn.Conv2d(channels, 256, 1)

	def forward(self, x):
		x = self.Conv2d_1(x)
		x = self.BatchNorm2d_1(x)
		x = self.ReLU_1(x)

		x = self.Conv2d_2(x)
		x = self.BatchNorm2d_2(x)
		x = self.ReLU_2(x)

		x = self.Conv2d_3(x)
		x = self.BatchNorm2d_3(x)
		x = self.ReLU_3(x)

		x = self.Conv2d_4(x)
		x = self.BatchNorm2d_4(x)
		x = self.ReLU_4(x)

		x = self.Conv2d_5(x)
		x = self.BatchNorm2d_5(x)
		x = self.ReLU_5(x)

		x = self.Conv2d_6(x)
		x = self.BatchNorm2d_6(x)
		x = self.ReLU_6(x)

		x = self.Conv2d_7(x)
		x = self.BatchNorm2d_7(x)
		x = self.ReLU_7(x)

		x = self.Conv2d_8(x)
		x = self.BatchNorm2d_8(x)
		x = self.ReLU_8(x)

		if self.img_size == 128:
			x = self.Conv2d_9(x)
			x = self.BatchNorm2d_9(x)
			x = self.ReLU_9(x)

			x = self.Conv2d_10(x)
			x = self.BatchNorm2d_10(x)
			x = self.ReLU_10(x)

		return self.out(x)
        
class Generator(nn.Module):
    def __init__(self, INPUT_NOISE_VECTOR, HIDDEN_LAYERS_GEN, RGB_CHANNEL):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input layer
# These two lines are part of the definition of a generator network in a Generative Adversarial Network (GAN) using PyTorch.

# nn.ConvTranspose2d(Z_DIM, G_HIDDEN * 8, 4, 1, 0, bias=False): This line creates a transposed convolutional layer (also known as a deconvolutional layer). This layer is used to upsample the input tensor. The parameters are:

# Z_DIM: The number of input channels. This is typically the length of the noise vector that is input to the generator.
# G_HIDDEN * 8: The number of output channels. This is the number of feature maps the layer will produce.
# 4: The size of the kernel. This is the width and height of the convolutional window.
# 1: The stride of the convolution. This is the step size for moving the convolutional window across the input tensor.
# 0: The padding. This is the number of pixels added to the input tensor on each side.
# bias=False: This indicates that the layer will not learn an additive bias.
# nn.BatchNorm2d(G_HIDDEN * 8): This line creates a batch normalization layer. This layer normalizes the output of the previous layer to have a mean of 0 and a standard deviation of 1, which can help speed up training and improve the final performance of the model. The parameter G_HIDDEN * 8 is the number of input channels, which should match the number of output channels from the previous layer.
            # nn.ConvTranspose2d(INPUT_NOISE_VECTOR, HIDDEN_LAYERS_GEN * 16, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(HIDDEN_LAYERS_GEN * 16),
            # nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(INPUT_NOISE_VECTOR, HIDDEN_LAYERS_GEN * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(HIDDEN_LAYERS_GEN * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # hidden layer 1
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
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
    


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

                # nn.Conv2d(D_HIDDEN * 8, D_HIDDEN * 16, 4, 2, 1, bias=False),
                # nn.BatchNorm2d(D_HIDDEN * 16),
                # nn.LeakyReLU(0.2, inplace=True),
                # output layer
                nn.Conv2d(D_HIDDEN * 8, 1, 4, 1, 0, bias=False),
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
                # 3rd layer
                nn.Conv2d(D_HIDDEN * 2, D_HIDDEN * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(D_HIDDEN * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # 4th layer

                # nn.Conv2d(D_HIDDEN * 8, D_HIDDEN * 16, 4, 2, 1, bias=False),
                # nn.BatchNorm2d(D_HIDDEN * 16),
                # nn.LeakyReLU(0.2, inplace=True),
                # output layer
                nn.Conv2d(D_HIDDEN * 4, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
#The activation functions used in the output layers of the generator and discriminator in a Generative Adversarial Network (GAN) are chosen based on the nature of the data and the roles of the networks.

# Generator (Tanh): The generator's goal is to generate data that resembles the real data. If the real data is images, these are often normalized to be in the range [-1, 1]. The Tanh activation function also outputs in the range [-1, 1], so it's a natural choice for the generator's output layer. This way, the output of the generator is already in the correct range to match the real data.

# Discriminator (Sigmoid): The discriminator's goal is to classify its input as real or fake. This is a binary classification problem, and the Sigmoid activation function is commonly used in the output layer for such problems. The Sigmoid function outputs a value in the range [0, 1], which can be interpreted as the probability of the input being real. A value close to 0 indicates a fake classification, and a value close to 1 indicates a real classification.

# These are common choices, but they're not the only possible ones. The best activation functions to use can depend on the specific characteristics of your data and model.

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def diversity_loss(fake):
    n = fake.size(0)
    fake = fake.view(n, -1)  # Flatten the images
    dists = torch.cdist(fake, fake, p=2)
    div_loss = -torch.mean(dists)  # Negative because we want to maximize diversity
    return div_loss

# Function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

INPUT_NOISE_VECTOR = 100
HIDDEN_LAYERS_GEN = 64
RGB_CHANNEL = 1
# Create the generator
Gen_net = Generator(INPUT_NOISE_VECTOR, HIDDEN_LAYERS_GEN, RGB_CHANNEL).to(DEVICE)
Gen_net.apply(weights_init)
print(Gen_net)

def model_diff(epoch, baseline, X_DIM, lr):
    if baseline or lr == 5e-4 or lr == 1e-5:
        model = diffusers.UNet2DModel(
            sample_size=X_DIM,  # the target image resolution
            in_channels=1,  # the number of input channels, 3 for RGB images
            out_channels=1,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(X_DIM * 2, X_DIM * 2, X_DIM * 4 , X_DIM * 4, X_DIM * 8, X_DIM * 8),  # the number of output channels for each UNet block
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
    else:
        model = diffusers.UNet2DModel(
            sample_size=X_DIM,  # the target image resolution
            in_channels=1,  # the number of input channels, 3 for RGB images
            out_channels=1,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(X_DIM * 2, X_DIM * 4, X_DIM * 8),  # the number of output channels for each UNet block
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
            ),
        )
    
    model.eval()
    model.to(DEVICE)
    huggingface_train_package = accelerate.Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=1,
    )
    if baseline:
        # # Load the state dict previously saved
        state_dict = torch.load('/data/s3287297/diff_baseline/model_weights_stable_diff_epoch_'+ str(epoch) +'.pth')
        print("continuing from baseline epoch " + str(epoch))
    elif X_DIM == 32:
        # # Load the state dict previously saved
        state_dict = torch.load('/data/s3287297/NIH_diffusion_model/img_size_32/model_weights_stable_diff_epoch_'+ str(epoch) +'_imgsize_32.pth')
        print("continuing from epoch " + str(epoch))
    elif lr == 5e-4:
        state_dict = torch.load('/data/s3287297/diff_5e-4/model_weights_stable_diff_epoch_'+ str(epoch) +'.pth')
        print("continuing from epoch 5e-4 " + str(epoch))
    elif lr == 1e-5:
        state_dict = torch.load('/data/s3287297/NIH_diffusion_model/model_weights_stable_diff_epoch_'+ str(epoch) +'_learningrate_1e-5.pth')
        print("continuing from epoch 1e-5 " + str(epoch))

    noise_creator_scheduler = diffusers.DDPMScheduler(num_train_timesteps=1000)

    # length_of_dataset = len(train_loader)
    Adam_optim = torch.optim.Adam(model.parameters(), lr=1e-4)


    # Update the model's weights
    model.load_state_dict(state_dict)
    pipeline = diffusers.DDPMPipeline(unet=huggingface_train_package.unwrap_model(model), scheduler=noise_creator_scheduler)
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    img = pipeline(
        batch_size=128,
        generator=torch.manual_seed(0),
    ).images
    return img
class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_input3 = nn.Linear(hidden_dim, hidden_dim)  # Added this line
        self.FC_input4 = nn.Linear(hidden_dim, hidden_dim)  # Added this line
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        self.training = True
        
    def forward(self, x):
        h_       = self.LeakyReLU(self.FC_input(x))
        h_       = self.LeakyReLU(self.FC_input2(h_))
        h_       = self.LeakyReLU(self.FC_input3(h_))  # Added this line
        h_       = self.LeakyReLU(self.FC_input4(h_))  # Added this line
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     # encoder produces mean and log of variance 
                                                       #             (i.e., parateters of simple tractable normal distribution "q"
        
        return mean, log_var
    

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_hidden3 = nn.Linear(hidden_dim, hidden_dim)  # Added this line
        self.FC_hidden4 = nn.Linear(hidden_dim, hidden_dim)  # Added this line
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h     = self.LeakyReLU(self.FC_hidden(x))
        h     = self.LeakyReLU(self.FC_hidden2(h))
        h     = self.LeakyReLU(self.FC_hidden3(h))  # Added this line
        h     = self.LeakyReLU(self.FC_hidden4(h))  # Added this line
        
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat
    

class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(DEVICE)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat            = self.Decoder(z)
        
        return x_hat, mean, log_var
batch_size = 128
X_DIM = 32
x_dim  = X_DIM * X_DIM
hidden_dim = 400
latent_dim = 200
encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)

model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)
# # Function to show an image
# def show_and_save_image(img_tensor, filename):
#     img = img_tensor.cpu().numpy()
#     img = img.transpose(1, 2, 0)  # Move the channels to the last dimension
#     plt.imshow(img)
#     plt.axis('off')
#     plt.savefig(filename, bbox_inches='tight', pad_inches=0)
#     plt.show()

BCE_loss = nn.BCELoss()

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


def load_image_paths(folder_path):
    """
    Load all image file paths from the specified folder.
    
    Parameters:
    folder_path (str): Path to the folder containing images.

    Returns:
    list: List of image file paths.
    """
    image_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_paths.append(os.path.join(root, file))
    return image_paths



class XrayDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)



class SimpleCNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the feature maps after the conv and pooling layers
        if input_size == 32:
            fc_input_dim = 64 * 8 * 8
        elif input_size == 64:
            fc_input_dim = 64 * 16 * 16
        else:
            raise ValueError("Unsupported input size. Only 32x32 and 64x64 are supported.")

        self.fc1 = nn.Linear(fc_input_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# # Load images (replace these with your actual data loading code)
# # Specify the directories containing the real and generated images
# real_images_folder = "/data/s3287297/NIH_data/images"
# generated_images_folder = "/data/s3287297/fake_images"

# # Get the lists of image paths
# real_image_paths = load_image_paths(real_images_folder)
# generated_image_paths = load_image_paths(generated_images_folder)

# # Output the lists to verify
# print("Real Image Paths:")
# print(real_image_paths)

# print("\nGenerated Image Paths:")
# print(generated_image_paths)

# # Combine the image paths and labels
# image_paths = real_image_paths + generated_image_paths
# labels = [1] * len(real_image_paths) + [0] * len(generated_image_paths)

# # Split into training and test sets
# train_paths, test_paths, train_labels, test_labels = train_test_split(
#     image_paths, labels, test_size=0.2, random_state=42)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# train_dataset = XrayDataset(train_paths, train_labels, transform)
# test_dataset = XrayDataset(test_paths, test_labels, transform)

# train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# input_size = 64
# model = SimpleCNN(input_size=input_size).to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.BCELoss()

# def train(model, device, train_loader, optimizer, criterion, epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data).squeeze()
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % 100 == 0:
#             print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
#                   f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

for epoch in range(1, 11):
    model_path = '/data/s3287297/classifier_model_64_' + str(epoch) + '.pth'
    if not os.path.exists(model_path):
        torch.save(model.state_dict(), '/data/s3287297/classifier_model_64_' + str(epoch) + '.pth')
        train(model, device, train_loader, optimizer, criterion, epoch)

for epoch in range(1, 11):
    model_path = '/data/s3287297/classifier_model_32_' + str(epoch) + '.pth'
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
        torch.save(model.state_dict(), '/data/s3287297/classifier_model_32_' + str(epoch) + '.pth')
        train(model, device, train_loader, optimizer, criterion, epoch)

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data).squeeze()
            test_loss += criterion(output, target).item()
            pred = (output > 0.5).float()
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.0f}%)\n')

# test(model, device, test_loader, criterion)

def score_generated_images(class_model, DEVICE, generated_image_paths, transform):
    class_model.eval()
    scores = []
    EPOCH_NUM = 25
    # with torch.no_grad():
    #     for image_path in generated_image_paths:
    #         image = Image.open(image_path).convert('L')
    #         image = transform(image).unsqueeze(0).to(device)
    #         score = class_model(image).item()
    #         scores.append(score)
    print("Starting Testing Loop...")
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
            class_model = SimpleCNN(input_size=input_size).to(DEVICE)
            class_model.load_state_dict(torch.load('/data/s3287297/classifier_model_32_10.pth'))
                     # Define the resize transformation
            image_dim = 32
            img_size = 32
            lr = 1e-4
            fake = model_diff(epoch, False, image_dim, lr)

            # Extract the region using slicing
            # This will extract the region from each image in the batch
            cutout = fake[0:128]
            list_fake_img = []
            for img in cutout:
                fake_one_img = transforms.ToTensor()(img).to(DEVICE)
                # fake_one_img = fake_one_img.unsqueeze(1) # Reshape the fake images
                list_fake_img.append(fake_one_img) 
            list_fake_img = torch.stack(list_fake_img)
            
            list_of_disc_errors_diff[epoch] = class_model(list_fake_img).mean().item()

            input_size = 64
            class_model = SimpleCNN(input_size=input_size).to(DEVICE)
            class_model.load_state_dict(torch.load( '/data/s3287297/classifier_model_64_10.pth'))
            image_dim = 64
            img_size = 64
            HIDDEN_LAYERS_GEN = 64
            Gen_net = Generator(INPUT_NOISE_VECTOR, HIDDEN_LAYERS_GEN, RGB_CHANNEL).to(DEVICE)
            Gen_net.apply(weights_init)
            generator_path = '/data/s3287297/GAN_baseline/generator_' + str(epoch) + '.pth'
            generator_state_dict = torch.load(generator_path)
            Gen_net.load_state_dict(generator_state_dict)

            print("Continuing from epoch " + str(epoch) + "...")
            # (2) Update the discriminator with fake data
            # Generate batch of latent vectors
            latent_vectors = torch.randn(128, INPUT_NOISE_VECTOR, 1, 1, device=DEVICE)
            # Generate fake image batch with G
            fake_img = Gen_net(latent_vectors)
            # Classify all fake batch with D
            list_of_disc_errors_GAN[epoch] = class_model(fake_img).mean().item()


        with torch.no_grad():
            input_size = 32
            class_model = SimpleCNN(input_size=input_size).to(DEVICE)
            class_model.load_state_dict(torch.load( '/data/s3287297/classifier_model_32_10.pth'))
            X_DIM = 28
            x_dim = 28*28
            img_size = 32
            encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
            decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)

            model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)
            model.load_state_dict(torch.load('/data/s3287297/NIH_VAE/NIH_VAE_32_imgsize/model_weights_VAE_total_epoch_'+ str(epoch+1) +'_.pth'))
            noise = torch.randn(batch_size, latent_dim).to(DEVICE)
            fake = decoder(noise)

            # Extract the region using slicing
            # This will extract the region from each image in the batch
            fake = fake.view(batch_size, 1, X_DIM, X_DIM)
            # Upscale the fake images to 32x32
            fake = F.interpolate(fake, size=(32, 32), mode='bilinear', align_corners=True)
            # fake = fake[top:bottom, left:right]
            # list_of_tensors = torch.stack(torch.split(fake, 0, dim=0), dim=0)

            list_of_disc_errors_VAE[epoch] = class_model(fake).mean().item()


            input_size = 64
            class_model = SimpleCNN(input_size=input_size).to(DEVICE)
            class_model.load_state_dict(torch.load( '/data/s3287297/classifier_model_64_10.pth'))
            config_file = sys.argv[1]
            assert os.path.exists(config_file), "Configuration file does not exit!"
            config = parse_config(config_file)
            model = config['model']
            images = config['images']
            
            load_path = model.get('load_path', '/data/s3287297/models_baseline_try_3/Model_Checkpoint_'+ str(epoch) +'.pt')
            if EPOCH_NUM == 24:
                load_path = model.get('load_path', '/data/s3287297/models_baseline_try_3/Model_Checkpoint_Last.pt')
            assert os.path.exists(load_path), 'Saved Model File Does not exist!'
            no_images = images.get('no_images', 128)
            images_size = images.get('images_size', 64)
            images_channels = images.get('images_channels', 1)
            

            #Define and load model
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            net = PixelCNN().to(device)
            if torch.cuda.device_count() > 1: #Accelerate testing if multiple GPUs available
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                net = nn.DataParallel(net)
            net.load_state_dict(torch.load(load_path))
            net.eval()



            fake = torch.Tensor(no_images, images_channels, images_size, images_size).to(device)
            fake.fill_(0)

            #Generating images pixel by pixel
            for i in range(images_size):
                for j in range(images_size):
                    out = net(fake)
                    probs = F.softmax(out[:,:,i,j], dim=-1).data
                    fake[:,:,i,j] = torch.multinomial(probs, 1).float() / 255.0

            #Saving images row wise
            # torchvision.utils.save_image(fake, 'auto_reg_epoch_'+ str(epoch)+'.png', nrow=12, padding=0)
            list_of_disc_errors_autoreg[epoch] = class_model(fake).mean().item()
    plt1.plot(list_of_epochs,list_of_disc_errors_autoreg,label="Discriminator Loss of the autoregressive model")
    plt1.plot(list_of_epochs,list_of_disc_errors_diff,label="Discriminator Loss of the diffusion model (32 x 32)")
    plt1.plot(list_of_epochs,list_of_disc_errors_VAE,label="Discriminator Loss of the VAE (32 x 32)")
    plt1.plot(list_of_epochs,list_of_disc_errors_GAN,label="Discriminator Loss of the GAN (baseline)")
    plt1.legend()
    fig.savefig("/data/s3287297/Classifier_vs_all.png",dpi=300)
    return scores

new_generated_image_paths = ["path/to/new/generated/image1.png", "path/to/new/generated/image2.png"]  # New generated images
scores = score_generated_images(model, device, new_generated_image_paths, transform)

# for path, score in zip(new_generated_image_paths, scores):
#     print(f"Image: {path}, Score: {score:.4f}")