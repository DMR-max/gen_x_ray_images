import os
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
from PIL import Image
class LoadImages:
    def __init__(self, directory_of_images, file_with_labels, preprocessing):
        self.directory_of_images = directory_of_images
        self.preprocessing = preprocessing
        self.data = []

        with open(file_with_labels, "r") as current_file:
            lines_in_file = current_file.readlines()
            for current_line in lines_in_file:
                current_line = current_line.strip().rsplit("_", 0)
                self.data.append(current_line)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_name = self.directory_of_images + "/" + self.data[idx][0]
        curr_image = Image.open(image_name)\
        
        if self.preprocessing is not None:
            curr_image = self.preprocessing(curr_image)
        
        return curr_image
    
class Generator(nn.Module):
    def __init__(self, INPUT_NOISE_VECTOR, HIDDEN_DIM_GEN, RGB_CHANNEL, img_size):
        super(Generator, self).__init__()
        if img_size == 128:
            self.main = nn.Sequential(
                # # input layer
                # # This line creates a convolutional layer it is made to upsample a tensor. The parameters of this layer are:
                nn.ConvTranspose2d(INPUT_NOISE_VECTOR, HIDDEN_DIM_GEN * 16, 4, 1, 0, bias=False),
                # # Z_DIM: means the amount of channels the input has. This is based on the input noise vector size. 
                # # HIDDEN_DIM_GEN * 8: The size of feature maps that will be used in the network.
                # # 4: Kernel size the dimensions of the convolutional window.
                # # 1: The step size for moving the kernel across the input tensor.
                # # 0: The padding. number of pixels added to the input tensor on each side.
                # # bias=False: Removes additive bias, helps with flexibility and pattern recognition of the model.
                # # nn.BatchNorm2d(G_HIDDEN * 8): normalizes the output of the previous layer to have a mean of 0 and a standard deviation of 1.
                nn.BatchNorm2d(HIDDEN_DIM_GEN * 16),
                # # nn.LeakyReLU(0.2, inplace=True): Leaky ReLU activation function. It is used to introduce non-linearity to the model, it helped with stabilizing the GAN versus a ReLU activation function.
                nn.LeakyReLU(0.2, inplace=True),
                
                # hidden layer 1
                nn.ConvTranspose2d(HIDDEN_DIM_GEN * 8, HIDDEN_DIM_GEN * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(HIDDEN_DIM_GEN * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # hidden layer 2
                nn.ConvTranspose2d(HIDDEN_DIM_GEN * 4, HIDDEN_DIM_GEN * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(HIDDEN_DIM_GEN * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # hidden layer 3
                nn.ConvTranspose2d(HIDDEN_DIM_GEN * 2, HIDDEN_DIM_GEN, 4, 2, 1, bias=False),
                nn.BatchNorm2d(HIDDEN_DIM_GEN),
                nn.LeakyReLU(0.2, inplace=True),
                # ending in a Tanh activation function, tanh squashes the output to be between -1 and 1 which is the range of the real images.
                # therefore the tanh is a good choice for the output layer of the generator.
                nn.ConvTranspose2d(HIDDEN_DIM_GEN, RGB_CHANNEL, 4, 2, 1, bias=False),
                nn.Tanh()
            )
        if img_size == 64:
            self.main = nn.Sequential(
                # input layer
                nn.ConvTranspose2d(INPUT_NOISE_VECTOR, HIDDEN_DIM_GEN * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(HIDDEN_DIM_GEN * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # hidden layer 1
                nn.ConvTranspose2d(HIDDEN_DIM_GEN * 8, HIDDEN_DIM_GEN * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(HIDDEN_DIM_GEN * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # hidden layer 2
                nn.ConvTranspose2d(HIDDEN_DIM_GEN * 4, HIDDEN_DIM_GEN * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(HIDDEN_DIM_GEN * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # hidden layer 3
                nn.ConvTranspose2d(HIDDEN_DIM_GEN * 2, HIDDEN_DIM_GEN, 4, 2, 1, bias=False),
                nn.BatchNorm2d(HIDDEN_DIM_GEN),
                nn.LeakyReLU(0.2, inplace=True),
                # output layer
                nn.ConvTranspose2d(HIDDEN_DIM_GEN, RGB_CHANNEL, 4, 2, 1, bias=False),
                nn.Tanh()
            )
        if img_size == 32:
            self.main = nn.Sequential(
                # input layer
                nn.ConvTranspose2d(INPUT_NOISE_VECTOR, HIDDEN_DIM_GEN * 4, 4, 1, 0, bias=False),
                nn.BatchNorm2d(HIDDEN_DIM_GEN * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # hidden layer 1
                nn.ConvTranspose2d(HIDDEN_DIM_GEN * 4, HIDDEN_DIM_GEN * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(HIDDEN_DIM_GEN * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # hidden layer 2
                nn.ConvTranspose2d(HIDDEN_DIM_GEN * 2, HIDDEN_DIM_GEN, 4, 2, 1, bias=False),
                nn.BatchNorm2d(HIDDEN_DIM_GEN),
                nn.LeakyReLU(0.2, inplace=True),
                # output layer
                nn.ConvTranspose2d(HIDDEN_DIM_GEN, RGB_CHANNEL, 4, 2, 1, bias=False),
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
HIDDEN_DIM_GEN = 64
# Dimensions of the image
DIM_IMAGE = 64
# Number of hidden layers in the discriminator
HIDDEN_DIM_DISCR = 64
# Number of epochs to train the GAN
EPOCH_NUM = 75
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


directory_of_images =r'NIH_data/images'
location_of_txt_file =r'NIH_data/train_val_list.txt'
location_of_txt_file_test = r'NIH_data/test_list.txt'

# Define the transformation for the images (resize, grayscale, from PIL Image to tensor, normalize with mean 0,5 and standard deviation 0,5)
transform = transforms.Compose([
    transforms.Resize((DIM_IMAGE)),  
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
img_size = 64
Disc_net = Discriminator(RGB_CHANNEL, HIDDEN_DIM_DISCR, img_size).to(DEVICE)
Disc_net.load_state_dict(torch.load('disc_64/discriminator_49.pth'))
# Initialize BCELoss function
BCE_crit = nn.BCELoss()

# Create batch of latent vectors
Create_noise = torch.randn(BATCH_SIZE, INPUT_NOISE_VECTOR, 1, 1, device=DEVICE)

# Setup Adam optimizers to help with the training of the GAN to stabilize the training
optimizerD = optim.Adam(Disc_net.parameters(), lr=lr, betas=(0.5, 0.999))

# Lists to keep track of progress
list_of_images = []
loss_of_gen = []
loss_of_disc = []
iterations = 0
INPUT_NOISE_VECTOR = 100
HIDDEN_DIM_GEN = 64
RGB_CHANNEL = 1

# Create the generator
img_size = 64
Gen_net = Generator(INPUT_NOISE_VECTOR, HIDDEN_DIM_GEN, RGB_CHANNEL, img_size).to(DEVICE)
Gen_net.apply(weights_init)
print(Gen_net)

print("Starting Testing Loop...")
# Create a list to store the loss of the GAN
list_of_disc_errors_1 = np.zeros(EPOCH_NUM)
list_of_disc_errors_2 = np.zeros(EPOCH_NUM)
list_of_disc_errors_baseline = np.zeros(EPOCH_NUM)
list_of_epochs = np.arange(0, EPOCH_NUM, 1)
# Create plots to show the loss of the Discriminator per epoch
fig, plt1 = plt.subplots()
plt1.set_xlabel('Epoch')
plt1.set_ylabel('Discriminator Loss')      
plt1.set_title("Discriminator Loss vs epoch GAN model")
for epoch in range(EPOCH_NUM):

    batch_count = 0
    for i, data in enumerate(train_loader, 0): 
        with torch.no_grad():
            img_size = 64
            # Create the discriminator made for the 64x64 images
            Disc_net = Discriminator(RGB_CHANNEL, HIDDEN_DIM_DISCR, img_size).to(DEVICE)
            # Load the weights of the discriminator
            Disc_net.load_state_dict(torch.load('disc_64/discriminator_49.pth'))
            # Load the images and labels 
            moving_images = data.to(DEVICE)
            batch_size = moving_images.size(0)
            label = torch.full((batch_size,), REAL_LABEL, dtype=torch.float, device=DEVICE)
            
            # Forward pass real batch through Discriminator
            output = Disc_net(moving_images).view(-1)
            real_error_disc = BCE_crit(output, label)     
            # Generate batch of latent vectors
            noise = torch.randn(batch_size, INPUT_NOISE_VECTOR, 1, 1, device=DEVICE)

            # # Generate fake image batch with Generator
            # HIDDEN_DIM_GEN = 128
            # img_size = 64
            # Gen_net = Generator(INPUT_NOISE_VECTOR, HIDDEN_DIM_GEN, RGB_CHANNEL, img_size).to(DEVICE)
            # Gen_net.apply(weights_init)
            # generator_path = 'GAN_hidden_layers_128/generator_' + str(epoch) + '.pth'
            # generator_state_dict = torch.load(generator_path)
            # Gen_net.load_state_dict(generator_state_dict)
            # Disc_net = Discriminator(RGB_CHANNEL, HIDDEN_DIM_DISCR, img_size).to(DEVICE)
            # Disc_net.load_state_dict(torch.load('disc_64/discriminator_49.pth'))

            # print("Continuing from epoch " + str(epoch) + "...")
            # # (2) Update the discriminator with fake data
            # # Generate batch of latent vectors
            # latent_vectors = torch.randn(batch_size, INPUT_NOISE_VECTOR, 1, 1, device=DEVICE)
            # # Generate fake image batch with G
            # fake_img = Gen_net(latent_vectors)
            # label.fill_(FAKE_LABEL)
            # # Classify all fake batch with D
            # class_output = Disc_net(fake_img.detach()).view(-1)
            # # Calculate D's loss on the all-fake batch
            # # print(class_output.shape)
            # Disc_loss_fake = BCE_crit(class_output, label)
            # # Calculate the average of the predictions of the discriminator over fake
            # D_G_z1 = class_output.mean().item()
            # # Compute error of D as sum over the fake and the real batches
            # errD = real_error_disc + Disc_loss_fake

            # list_of_disc_errors_1[epoch] += errD.item()
            # print(errD.item())
            
            HIDDEN_DIM_GEN = 64
            img_size = 64
            Gen_net = Generator(INPUT_NOISE_VECTOR, HIDDEN_DIM_GEN, RGB_CHANNEL, img_size).to(DEVICE)
            Gen_net.apply(weights_init)
            generator_path = 'GAN_baseline/generator_' + str(epoch) + '.pth'
            generator_state_dict = torch.load(generator_path)
            Gen_net.load_state_dict(generator_state_dict)
            Disc_net = Discriminator(RGB_CHANNEL, HIDDEN_DIM_DISCR, img_size).to(DEVICE)
            Disc_net.load_state_dict(torch.load('disc_64/discriminator_49.pth'))

            print("Continuing from epoch " + str(epoch) + "...")
            # (2) Update the discriminator with fake data
            # Generate batch of latent vectors
            latent_vectors = torch.randn(batch_size, INPUT_NOISE_VECTOR, 1, 1, device=DEVICE)
            # Generate fake image batch with G
            fake_img = Gen_net(latent_vectors)
            label.fill_(FAKE_LABEL)
            # Classify all fake batch with D
            class_output = Disc_net(fake_img.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            # print(class_output.shape)
            Disc_loss_fake = BCE_crit(class_output, label)
            # Calculate the average of the predictions of the discriminator over fake
            D_G_z1 = class_output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = real_error_disc + Disc_loss_fake

            list_of_disc_errors_baseline[epoch] += errD.item()
            print(errD.item())

            # HIDDEN_DIM_GEN = 32
            # img_size = 64+
            # Gen_net = Generator(INPUT_NOISE_VECTOR, HIDDEN_DIM_GEN, RGB_CHANNEL, img_size).to(DEVICE)
            # Gen_net.apply(weights_init)
            # generator_path = 'GAN_hidden_layers_32/generator_' + str(epoch) + '.pth'
            # generator_state_dict = torch.load(generator_path)
            # Gen_net.load_state_dict(generator_state_dict)
            # Disc_net = Discriminator(RGB_CHANNEL, HIDDEN_DIM_DISCR, img_size).to(DEVICE)
            # Disc_net.load_state_dict(torch.load('disc_64/discriminator_49.pth'))

            # print("Continuing from epoch " + str(epoch) + "...")
            # # (2) Update the discriminator with fake data
            # # Generate batch of latent vectors
            # latent_vectors = torch.randn(batch_size, INPUT_NOISE_VECTOR, 1, 1, device=DEVICE)
            # # Generate fake image batch with G
            # fake_img = Gen_net(latent_vectors)
            # label.fill_(FAKE_LABEL)
            # # Classify all fake batch with D
            # class_output = Disc_net(fake_img.detach()).view(-1)
            # # Calculate D's loss on the all-fake batch
            # # print(class_output.shape)
            # Disc_loss_fake = BCE_crit(class_output, label)
            # # Calculate the average of the predictions of the discriminator over fake
            # D_G_z1 = class_output.mean().item()
            # # Compute error of D as sum over the fake and the real batches
            # errD = real_error_disc + Disc_loss_fake

            # list_of_disc_errors_2[epoch] += errD.item()
            # print(errD.item())
            break
plt1.plot(list_of_epochs,list_of_disc_errors_baseline,label="Discriminator Loss of baseline")
# plt1.plot(list_of_epochs,list_of_disc_errors_2,label="Discriminator Loss of GAN model with hidden dimension = 32")
# plt1.plot(list_of_epochs,list_of_disc_errors_1,label="Discriminator Loss of GAN model with hidden dimension = 128")
plt1.legend()
fig.savefig("Discriminator_vs_GAN_baseline_V2.png",dpi=300)