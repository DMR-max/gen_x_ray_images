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

def main():
    version = input("Which version of the model do you want to make a graph of? 1 = learning rate, 2 = Image size, 3 = hidden dimension, 4 = Baseline")
    baseline_path = input("what is the checkpoint location of the baseline model?")
    if version == '1' or version == '2' or version == '3':
        input_model_1 = input("what is the checkpoint location of the second model? (for learning rate 1e-3, for image size 32, for hidden dimension 32)")
        input_model_2 = input("what is the checkpoint location of the third model? (for learning rate 1e-5, for image size 128, for hidden dimension 128)")
    location_of_discriminator = input("What is the location of the discriminator?")
    if version == "2":
        location_of_discriminator_32 = input("What is the location of the (32 x 32) discriminator?")
        location_of_discriminator_128 = input("What is the location of the (128 x 128) discriminator?")
    
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
    EPOCH_NUM = 50
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
        lr_list = [1e-4, 1e-3, 1e-5]
    else:
        lr_list = [1e-4]
    for lr in lr_list:
        for image_size in image_size_list:
            for hidden_dim in hidden_dim_list:
                HIDDEN_DIM_DISCR = hidden_dim
                HIDDEN_DIM_GEN = hidden_dim
                if lr_list == 1e-3:
                    HIDDEN_DIM_DISCR = 128
                    HIDDEN_DIM_GEN = 128
                elif lr_list == 1e-4:
                    HIDDEN_DIM_DISCR = 64
                    HIDDEN_DIM_GEN = 64
                elif lr_list == 1e-5:
                    HIDDEN_DIM_DISCR = 64
                    HIDDEN_DIM_GEN = 64

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
                Disc_net = Disc(RGB_CHANNEL, HIDDEN_DIM_DISCR, image_size).to(DEVICE)
                if image_size == 64:
                    Disc_net.load_state_dict(torch.load(location_of_discriminator))
                elif image_size == 32:
                    Disc_net.load_state_dict(torch.load(location_of_discriminator_32))
                elif image_size == 128:
                    Disc_net.load_state_dict(torch.load(location_of_discriminator_128))
                
                # Initialize BCELoss function
                BCE_crit = nn.BCELoss()

                INPUT_NOISE_VECTOR = 100
                HIDDEN_DIM_GEN = 64
                RGB_CHANNEL = 1

                # Create the generator
                Gen_net = Generator(INPUT_NOISE_VECTOR, HIDDEN_DIM_GEN, RGB_CHANNEL, image_size).to(DEVICE)
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
                if version == "1" or version == "2" or version == "3":
                    paths = [baseline_path, input_model_1, input_model_2]
                else:
                    path = [baseline_path]
                for path in paths:
                    for epoch in range(EPOCH_NUM):

                        batch_count = 0
                        for i, data in enumerate(train_loader, 0): 
                            with torch.no_grad():
                                # Create the discriminator made for the 64x64 images
                                Disc_net = Disc(RGB_CHANNEL, HIDDEN_DIM_DISCR, image_size).to(DEVICE)
                                # Load the weights of the discriminator
                                if image_size == 64:
                                    Disc_net.load_state_dict(torch.load(location_of_discriminator))
                                elif image_size == 32:
                                    Disc_net.load_state_dict(torch.load(location_of_discriminator_32))
                                elif image_size == 128:
                                    Disc_net.load_state_dict(torch.load(location_of_discriminator_128))
                                
                                # Load the images and labels 
                                moving_images = data.to(DEVICE)
                                batch_size = moving_images.size(0)
                                label = torch.full((batch_size,), REAL_LABEL, dtype=torch.float, device=DEVICE)
                                
                                # Forward pass real batch through Discriminator
                                output = Disc_net(moving_images).view(-1)
                                real_error_disc = BCE_crit(output, label)     
                                # Generate batch of latent vectors
                                noise = torch.randn(batch_size, INPUT_NOISE_VECTOR, 1, 1, device=DEVICE)
                    
                                Gen_net = Generator(INPUT_NOISE_VECTOR, HIDDEN_DIM_GEN, RGB_CHANNEL, image_size).to(DEVICE)
                                Gen_net.apply(weights_init)
                                generator_path = 'GAN_baseline/generator_' + str(epoch) + '.pth'
                                generator_state_dict = torch.load(generator_path)
                                Gen_net.load_state_dict(generator_state_dict)
                                Disc_net = Disc(RGB_CHANNEL, HIDDEN_DIM_DISCR, image_size).to(DEVICE)
                                if image_size == 64:
                                    Disc_net.load_state_dict(torch.load(location_of_discriminator))
                                elif image_size == 32:
                                    Disc_net.load_state_dict(torch.load(location_of_discriminator_32))
                                elif image_size == 128:
                                    Disc_net.load_state_dict(torch.load(location_of_discriminator_128))
                                
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
                        plt1.plot(list_of_epochs,list_of_disc_errors_1,label="Discriminator Loss of the GAN model with lr = 1e-3")
                    elif path == input_model_1 and version == "2":
                        plt1.plot(list_of_epochs,list_of_disc_errors_1,label="Discriminator Loss of the GAN model with image size = 32")
                    elif path == input_model_1 and version == "3":
                        plt1.plot(list_of_epochs,list_of_disc_errors_1,label="Discriminator Loss of the GAN model with hidden dimension = 300")
                    elif path == input_model_2 and version == "1":
                        plt1.plot(list_of_epochs,list_of_disc_errors_2,label="Discriminator Loss of the GAN model with lr = 1e-5")
                    elif path == input_model_2 and version == "2":
                        plt1.plot(list_of_epochs,list_of_disc_errors_2,label="Discriminator Loss of the GAN model with image size = 128")
                    elif path == input_model_2 and version == "3":
                        plt1.plot(list_of_epochs,list_of_disc_errors_2,label="Discriminator Loss of the GAN model with hidden dimension = 500")
    plt1.legend()
    if version == "1":
        fig.savefig("Discriminator_vs_GAN_lr.png",dpi=300)
    if version == "2":
        fig.savefig("Discriminator_vs_GAN_image_size.png",dpi=300)
    if version == "3":
        fig.savefig("Discriminator_vs_GAN_hidden_dim.png",dpi=300)
    if version == "4":
        fig.savefig("Discriminator_vs_GAN_baseline.png",dpi=300)    