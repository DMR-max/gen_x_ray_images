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
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, folder, txt_file, trans=None):
        self.folder = folder
        self.trans = trans
        self.image_names = []
        self.target = []

        # Read the txt file containing information about dataset split and possibly labels
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                target, _ = line.strip().rsplit("_",1)  # Assuming each line contains image_name and target label
                image_name = line.strip()
                self.image_names.append(image_name)
                self.target.append(int(target))  # Convert target to int

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.folder, self.image_names[idx])
        image = Image.open(img_name)

        if self.trans:
            image = self.trans(image)

        target = self.target[idx]
        return image, target  
    

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
                # output layer
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
                # # 4th layer

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
CUDA = True
DATA_PATH = './data'
BATCH_SIZE = 128
IMAGE_CHANNEL = 1
Z_DIM = 100
G_HIDDEN = 64
X_DIM = 32
D_HIDDEN = 64
EPOCH_NUM = 50
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
cudnn.benchmark = True

    # Example usage:
# os.chdir('..')
# os.chdir('..')
# root_dir = r'/data/s3287297/NIH_data/images'
# txt_file = r'/data/s3287297/NIH_data/train_val_list.txt'
# txt_file_test = r'/data/s3287297/NIH_data/test_list.txt'
root_dir = r'NIH_data/images'
txt_file = r'NIH_data/train_val_list.txt'
txt_file_test = r'NIH_data/test_list.txt'

# Define your transform
transform = transforms.Compose([
    transforms.Resize((X_DIM)),  # Resize image to (224, 224)
    transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
    transforms.ToTensor(),  # Convert PIL Image to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the tensor with mean and standard deviation
])

# Create custom dataset instance
custom_dataset = CustomImageDataset(folder=root_dir, txt_file=txt_file, trans=transform)
custom_dataset_test = CustomImageDataset(folder=root_dir, txt_file=txt_file_test, trans=transform)


# Define the size of your training and testing sets
train_size = custom_dataset # 80% of the dataset for training
test_size = custom_dataset_test  # 20% of the dataset for testing



# Create DataLoaders for training and testing sets
train_loader = torch.utils.data.DataLoader(train_size, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_size, batch_size=BATCH_SIZE, shuffle=True)


# Create the discriminator
img_size = 64
netD = Discriminator(IMAGE_CHANNEL, D_HIDDEN, img_size).to(DEVICE)
netD.load_state_dict(torch.load('disc_64/discriminator_49.pth'))
            
# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that I will use to visualize the progression of the generator
viz_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=DEVICE)

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0


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




# Create DataLoader
dataloader = torch.utils.data.DataLoader(custom_dataset, batch_size=128, shuffle=True)
print("Starting Training Loop...")
print("Starting Testing Loop...")
list_of_disc_errors_500 = np.zeros(EPOCH_NUM)
list_of_disc_errors_300 = np.zeros(EPOCH_NUM)
list_of_disc_errors_baseline = np.zeros(EPOCH_NUM)
list_of_epochs = np.arange(0, EPOCH_NUM, 1)
fig, plt1 = plt.subplots()
plt1.set_xlabel('Epoch')
plt1.set_ylabel('Discriminator Loss')      
plt1.set_title("Discriminator Loss vs epoch Variational Autoencoder")
for epoch in range(EPOCH_NUM):
    for i, data in enumerate(dataloader, 0):
        img_size = 32
        netD = Discriminator(IMAGE_CHANNEL, D_HIDDEN, img_size).to(DEVICE)
        netD.load_state_dict(torch.load('disc_32/discriminator_49.pth'))  
        real_cpu = data[0].to(DEVICE)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), REAL_LABEL, dtype=torch.float, device=DEVICE)
        print(torch.Tensor.size(real_cpu))
        # if label.size(dim=0) != 64:
        #     top, left, bottom, right = 0, 0, 120, 120
        # else:
        #     top, left, bottom, right = 0, 0, 3, 2  # Example coordinates

        # Extract the region using slicing
        # This will extract the region from each image in the batch
        # real_cpu = real_cpu[top:bottom, left:right]
        # show_and_save_image(real_cpu[0], '/data/s3287297/NIH_VAE/after_slicing.png')
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)     
        # (2) Update the discriminator with fake data
        # Generate batch of latent vectors
        noise = torch.randn(b_size, Z_DIM, 1, 1, device=DEVICE)
        # Generate fake image batch with G
        model.eval()

        with torch.no_grad():
            hidden_dim = 400
            X_DIM = 28
            x_dim = 28*28
            netD = Discriminator(IMAGE_CHANNEL, D_HIDDEN, img_size).to(DEVICE)
            netD.load_state_dict(torch.load('disc_32/discriminator_49.pth'))

            encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
            decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)

            model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)
            model.load_state_dict(torch.load('NIH_VAE_32_imgsize/model_weights_VAE_total_epoch_'+ str(epoch+1) +'_.pth'))
            noise = torch.randn(batch_size, latent_dim).to(DEVICE)
            fake = decoder(noise)
        
        with torch.no_grad():
            # Set the noise
            noise = torch.randn(batch_size, latent_dim).to(DEVICE)
            # Generate images from the decoder by using the generated noise
            generated_images = decoder(noise)
            # Save the images
            save_image(generated_images.view(batch_size, 1, X_DIM, X_DIM), f'NIH_VAE_32_imgsize/BSgenerated_images_epoch_{epoch+1}'+ '_lr='+str(lr)+ '.png')
                    


        # show_image(fake, idx = 0)
        label.fill_(FAKE_LABEL)
        # Classify all fake batch with D
        # Define the coordinates of the region to extract (top, left, bottom, right)
        # if label.size(dim=0) != 64:
        #     top, left, bottom, right = 0, 0, 64, 64
        # else:
        #     top, left, bottom, right = 0, 0, 64, 64  # Example coordinates

        # Extract the region using slicing
        # This will extract the region from each image in the batch
        fake = fake.view(batch_size, 1, X_DIM, X_DIM)
        # fake = fake[top:bottom, left:right]
        # list_of_tensors = torch.stack(torch.split(fake, 0, dim=0), dim=0)


        print(torch.Tensor.size(fake))
        print(torch.Tensor.size(fake))
        print(torch.Tensor.size(label))
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        print(output.size())
        print(label.size())
        errD_fake = criterion(output, label)
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        list_of_disc_errors_baseline[epoch] = errD.item()
        print(epoch)
        print(errD.item())

                        # Open the file in append mode ('a') and write the loss value
        with open('loss_values.txt', 'a') as file:
            file.write(f"{errD.item()}\n")  # Add a newline character to separate entries


        # with torch.no_grad():
        #     X_DIM = 64
        #     x_dim = 64*64
        #     img_size = 64
        #     netD = Discriminator(IMAGE_CHANNEL, D_HIDDEN, img_size).to(DEVICE)
        #     netD.load_state_dict(torch.load('disc_64/discriminator_49.pth'))

        #     encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        #     decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)

        #     model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)
        #     model.load_state_dict(torch.load('VAE_1e-5/model_weights_VAE_total_epoch_'+ str(epoch+1) +'_.pth'))
        #     lr = 1e-5
        #     noise = torch.randn(batch_size, latent_dim).to(DEVICE)
        #     fake = decoder(noise)
        
        # with torch.no_grad():
        #     # Set the noise
        #     noise = torch.randn(batch_size, latent_dim).to(DEVICE)
        #     # Generate images from the decoder by using the generated noise
        #     generated_images = decoder(noise)
        #     # Save the images
        #     save_image(generated_images.view(batch_size, 1, X_DIM, X_DIM), f'UNOgenerated_images_epoch_{epoch+1}'+ '_lr='+str(lr)+ '.png')

        # # show_image(fake, idx = 0)
        # label.fill_(FAKE_LABEL)
        # # Classify all fake batch with D
        # # Define the coordinates of the region to extract (top, left, bottom, right)
        # if label.size(dim=0) != 64:
        #     top, left, bottom, right = 0, 0, 64, 64
        # else:
        #     top, left, bottom, right = 0, 0, 64, 64  # Example coordinates

        # # Extract the region using slicing
        # # This will extract the region from each image in the batch
        # fake = fake.view(batch_size, 1, X_DIM, X_DIM)
        # # fake = fake[top:bottom, left:right]
        # # list_of_tensors = torch.stack(torch.split(fake, 0, dim=0), dim=0)


        # print(torch.Tensor.size(fake))
        # print(torch.Tensor.size(fake))
        # print(torch.Tensor.size(label))
        # output = netD(fake.detach()).view(-1)
        # # Calculate D's loss on the all-fake batch
        # print(output.size())
        # print(label.size())
        # errD_fake = criterion(output, label)
        # D_G_z1 = output.mean().item()
        # # Compute error of D as sum over the fake and the real batches
        # errD = errD_real + errD_fake
        # list_of_disc_errors_300[epoch] = errD.item()
        # print(errD.item())


        # with torch.no_grad():
        #     X_DIM = 28
        #     x_dim = 28*28
        #     img_size = 32
        #     netD = Discriminator(IMAGE_CHANNEL, D_HIDDEN, img_size).to(DEVICE)
        #     netD.load_state_dict(torch.load('disc_32/discriminator_49.pth'))
        #     encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        #     decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)

        #     model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)
        #     model.load_state_dict(torch.load('NIH_VAE_32_imgsize/model_weights_VAE_total_epoch_'+ str(epoch+1) +'_.pth'))
        #     noise = torch.randn(batch_size, latent_dim).to(DEVICE)
        #     fake = decoder(noise)

        
        # with torch.no_grad():
        #     # Set the noise
        #     noise = torch.randn(batch_size, latent_dim).to(DEVICE)
        #     # Generate images from the decoder by using the generated noise
        #     generated_images = decoder(noise)
        #     # Save the images
        #     save_image(generated_images.view(batch_size, 1, X_DIM, X_DIM), f'DOSgenerated_images_epoch_{epoch+1}'+ '_lr='+str(lr)+ '.png')

        # # show_image(fake, idx = 0)
        # label.fill_(FAKE_LABEL)
        # # Classify all fake batch with D
        # # Define the coordinates of the region to extract (top, left, bottom, right)
        # if label.size(dim=0) != 64:
        #     top, left, bottom, right = 0, 0, 64, 64
        # else:
        #     top, left, bottom, right = 0, 0, 64, 64  # Example coordinates

        # # Extract the region using slicing
        # # This will extract the region from each image in the batch
        # fake = fake.view(batch_size, 1, X_DIM, X_DIM)
        # # Upscale the fake images to 32x32
        # fake = F.interpolate(fake, size=(32, 32), mode='bilinear', align_corners=True)
        # # fake = fake[top:bottom, left:right]
        # # list_of_tensors = torch.stack(torch.split(fake, 0, dim=0), dim=0)


        # print(torch.Tensor.size(fake))
        # print(torch.Tensor.size(fake))
        # print(torch.Tensor.size(label))
        # output = netD(fake.detach()).view(-1)
        # # Calculate D's loss on the all-fake batch
        # print(output.size())
        # print(label.size())
        # errD_fake = criterion(output, label)
        # D_G_z1 = output.mean().item()
        # # Compute error of D as sum over the fake and the real batches
        # errD = errD_real + errD_fake
        # list_of_disc_errors_500[epoch] = errD.item()
        # print(errD.item())
        break
# plt1.plot(list_of_epochs,list_of_disc_errors_300,label="Discriminator Loss of VAE with 128 x 128")
# plt1.plot(list_of_epochs,list_of_disc_errors_500,label="Discriminator Loss of VAE with 32 x 32")
plt1.plot(list_of_epochs,list_of_disc_errors_baseline,label="Discriminator Loss of baseline")
plt1.legend()
# fig.savefig("Discriminator_vs_VAE_baseline_extended.png",dpi=300)