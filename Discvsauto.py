import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
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
class Model(torch.nn.Module):
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

def main():
    os.chdir("..")
    version = input("Which version of the model do you want to make a graph of? 1 = learning rate, 2 = Image size, 3 = Baseline")
    baseline_path = input("what is the checkpoint location of the baseline model?")
    if version == '1' or version == '2':
        input_model_1 = input("what is the checkpoint location of the second model? (for learning rate 5e-4, for image size 32)")
        input_model_2 = input("what is the checkpoint location of the third model? (for learning rate 1e-5, for image size 128)")
    location_of_discriminator = input("What is the location of the discriminator?")
    if version == 2:
        location_of_discriminator_32 = input("What is the location of the (32 x 32) discriminator?")
        location_of_discriminator_128 = input("What is the location of the (128 x 128) discriminator?")
    # Flag to use CUDA
    CUDA = True
    # Batch size for training
    SIZE_OF_BATCH = 128
    # Number of channels in the RGB image (1 channel = grayscale)
    RGB_CHANNEL = 1
    # Size of the input noise vector
    INPUT_NOISE_VECTOR = 100
    # Dimensions of the image
    DIM_IMAGE = 64
    # Number of hidden layers in the discriminator
    HIDDEN_DIM_DISCR = 64
    # Number of epochs to train the GAN
    EPOCH_NUM = 24
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
    location_of_txt_file_test = input("What is the location of the txt file with the test value names?")

    if version == "2":
        image_size_list = [32, 64, 128]
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
            training_set = LoadImages(directory_of_images, location_of_txt_file, transform_params)
            testing_set = LoadImages(directory_of_images, location_of_txt_file_test, transform_params)

            # Create DataLoaders for training and testing sets
            train = torch.utils.data.DataLoader(training_set, batch_size=SIZE_OF_BATCH, shuffle=True)
            test = torch.utils.data.DataLoader(testing_set, batch_size=SIZE_OF_BATCH, shuffle=True)


            # Create the discriminator
            Disc_net = Disc(RGB_CHANNEL, HIDDEN_DIM_DISCR, DIM_IMAGE).to(dev)
            Disc_net.load_state_dict(torch.load(location_of_discriminator))
            # Initialize BCELoss function
            BCE_crit = nn.BCELoss()

            # Create batch of latent vectors
            Create_noise = torch.randn(SIZE_OF_BATCH, INPUT_NOISE_VECTOR, 1, 1, device=dev)

            # Setup Adam optimizers to help with the training of the GAN to stabilize the training
            Adam_optim = torch.optim.Adam(Disc_net.parameters(), lr=lr, betas=(0.4999, 0.999))

            # Lists to keep track of progress
            list_of_images = []
            loss_of_gen = []
            loss_of_disc = []



            print("Starting Training Loop...")
            print("Starting Testing Loop...")
            # list_of_disc_errors_500 = [3.67794132232666, 3.503178119659424, 3.6076016426086426, 3.2208688259124756, 3.860383987426758, 3.8884263038635254, 3.5787596702575684, 4.286342144012451, 3.7707061767578125, 3.720978021621704, 3.254955530166626, 3.0980122089385986, 3.4185142517089844, 3.418964147567749, 3.989417314529419, 3.519407272338867, 3.5756218433380127, 4.06866455078125, 3.508878231048584, 3.6577486991882324, 3.6472702026367188, 4.0691304206848145, 3.7421486377716064, 3.6138362884521484, 4.099040508270264 ]
            # list_of_disc_errors_300 = [69.31978607177734, 69.20040130615234, 69.24744415283203, 69.27139282226562, 69.21062469482422, 69.39337921142578, 69.55066680908203, 69.40667724609375, 69.22758483886719, 69.4830551147461, 69.5771713256836, 69.32005310058594, 69.31959533691406, 69.28053283691406, 69.39614868164062, 69.37401580810547, 69.03463745117188, 69.16797637939453, 69.31694793701172, 68.93363952636719, 69.4273910522461, 69.13243103027344, 69.41190338134766, 69.30957794189453, 69.27418518066406 ]
            # list_of_disc_errors_baseline = [23.622438430786133, 24.197343826293945, 23.40094566345215, 23.98157501220703, 23.79994010925293, 23.728822708129883, 23.974998474121094, 23.890085220336914, 24.14176368713379, 23.74187660217285, 23.650381088256836, 24.00697135925293, 24.07979393005371, 23.579343795776367, 24.249757766723633, 23.928030014038086, 23.831193923950195, 23.691335678100586, 23.59296226501465, 23.849149703979492, 23.411224365234375, 23.728271484375,24.34131622314453, 23.822879791259766, 24.16730499267578 ]
            list_of_epochs = np.arange(0, EPOCH_NUM, 1)
            # list_of_disc_errors_500 = [18.82244873046875,18.918537139892578, 19.124990463256836, 18.5001220703125, 19.009017944335938, 18.741548538208008, 18.939895629882812, 18.613489151000977, 19.12775993347168, 19.06594467163086, 18.558467864990234, 18.073972702026367, 18.947999954223633, 18.942468643188477, 18.720260620117188, 19.045827865600586, 18.850317001342773, 18.98076820373535, 18.432954788208008, 18.444000244140625, 19.335107803344727, 18.65494728088379, 18.569318771362305, 18.9158935546875 ]
            # list_of_disc_errors_300 = [26.757959365844727, 26.80496597290039, 26.54191017150879, 26.69702911376953, 26.790878295898438, 26.674758911132812, 26.609771728515625, 26.883424758911133, 26.8837947845459, 26.916271209716797, 26.763290405273438, 26.727956771850586, 26.631988525390625, 27.034324645996094, 26.825286865234375, 27.003862380981445, 26.634309768676758, 26.861234664916992, 26.896944046020508, 26.71302604675293, 27.313766479492188, 26.183168411254883, 26.706396102905273, 26.690881729125977 ]
            # list_of_disc_errors_baseline = [34.59003829956055, 33.089088439941406, 32.711219787597656, 31.56462287902832, 29.641326904296875, 23.40835952758789, 27.863941192626953, 17.428369522094727, 20.87794303894043, 6.416240215301514, 26.180452346801758, 21.99127197265625, 18.7678165435791, 24.87398910522461, 21.016145706176758, 13.378898620605469, 21.922250747680664, 17.909151077270508, 21.157373428344727, 18.828563690185547, 20.63825798034668, 19.880434036254883, 20.312255859375, 15.060330390930176 ]
            list_of_disc_errors_baseline = np.zeros(EPOCH_NUM)
            list_of_disc_errors_300 = np.zeros(EPOCH_NUM)
            list_of_disc_errors_500 = np.zeros(EPOCH_NUM)
            fig, plt1 = plt.subplots()
            plt1.set_xlabel('Epoch')
            plt1.set_ylabel('Discriminator Loss')      
            plt1.set_title("Discriminator Loss vs epoch Autoregressive model")
            if version == "1" or version == "2":
                paths = [baseline_path, input_model_1, input_model_2]
            else:
                path = [baseline_path]
            for path in paths:
                for epoch in range(EPOCH_NUM):
                    for i, data in enumerate(train, 0):
                        # Create the discriminator made for the 64x64 images
                        Disc_net = Disc(RGB_CHANNEL, HIDDEN_DIM_DISCR, DIM_IMAGE).to(dev)
                        # Load the weights of the discriminator
                        if DIM_IMAGE == 64:
                            Disc_net.load_state_dict(torch.load(location_of_discriminator))
                        elif DIM_IMAGE == 32:
                            Disc_net.load_state_dict(torch.load(location_of_discriminator_32))
                        elif DIM_IMAGE == 128:
                            Disc_net.load_state_dict(torch.load(location_of_discriminator_128))
                        # Load the images and labels 
                        moving_imgs = data.to(dev)
                        b_size = moving_imgs.size(0)
                        label = torch.full((b_size,), R_LABEL, dtype=torch.float, device=dev)
                        
                        # Propagation through the Discriminator for the real images
                        output = Disc_net(moving_imgs).view(-1)
                        real_error_disc = BCE_crit(output, label)     
                        # Generate noise
                        noise = torch.randn(b_size, INPUT_NOISE_VECTOR, 1, 1, device=dev)

                        with torch.no_grad():
                            # load in autoregressive model weights
                            load_path = str(path) + '/Model_Checkpoint_'+ str(epoch) +'.pt'
                            if epoch == 24:
                                load_path = str(path) + '/Model_Checkpoint_Last.pt'
                            assert os.path.exists(load_path), 'Saved Model File Does not exist!'
                            # standard parameters for the autoregressive model
                            no_images = 128
                            images_size = DIM_IMAGE
                            images_channels = 1
                            
                            # load in the PixelCNN
                            if images_size == 32:
                                ch_numb = 200
                            else:
                                ch_numb = 64
                            net = Model(ch_numb = ch_numb).to(dev)

                            # load the state dictionary    
                            curr_state_dict = torch.load(load_path)
                            net.load_state_dict(curr_state_dict)
                            # set evaluation mode, so that the model does not change the weights
                            net.eval()


                            # create a tensor to store the generated images
                            fake = torch.Tensor(no_images, images_channels, images_size, images_size).to(dev)
                            # fill the tensor with zeros
                            fake.fill_(0)

                            #Generating images pixel by pixel
                            for rows in range(images_size):
                                for cols in range(images_size):
                                    out = net(fake)
                                    probability = F.softmax(out[:,:,rows,cols], dim=-1).data
                                    fake[:,:,rows,cols] = torch.multinomial(probability, 1)
                                    fake[:,:,rows,cols] = fake[:,:,rows,cols].float()
                                    fake[:,:,rows,cols] = fake[:,:,rows,cols] / 255.0

                            # Saving images row wise
                            torchvision.utils.save_image(fake, str(path) + '/auto_reg_epoch_'+ str(epoch)+'.png', nrow=12, padding=0)
                        # Generate noise
                        latent_vectors = torch.randn(b_size, INPUT_NOISE_VECTOR, 1, 1, device=dev)
                        label.fill_(F_LABEL)
                        # Classify all generated images
                        class_output = Disc_net(fake.detach()).view(-1)
                        # Calculate the discriminator loss for the generated images
                        Disc_loss_fake = BCE_crit(class_output, label)
                        # Calculate the average of the predictions of the discriminator over fake
                        average_pred_fake = class_output.mean().item()
                        # Calculate the total loss of the discriminator over the real and fake images
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
                    plt1.plot(list_of_epochs,list_of_disc_errors_500,label="Discriminator Loss of the autoreg model with lr = 5e-4")
                elif path == input_model_1 and version == "2":
                    plt1.plot(list_of_epochs,list_of_disc_errors_500,label="Discriminator Loss of the autoreg model with image size = 32")
                elif path == input_model_2 and version == "1":
                    plt1.plot(list_of_epochs,list_of_disc_errors_300,label="Discriminator Loss of the autoreg model with lr = 1e-5")
                elif path == input_model_2 and version == "2":
                    plt1.plot(list_of_epochs,list_of_disc_errors_300,label="Discriminator Loss of the autoreg model with image size = 128")

    plt1.legend()
    if version == "1":
        fig.savefig("Discriminator_vs_autoreg_lr.png",dpi=300)
    if version == "2":
        fig.savefig("Discriminator_vs_autoreg_image_size.png",dpi=300)
    if version == "3":
        fig.savefig("Discriminator_vs_autoreg_baseline.png",dpi=300)

if __name__=="__main__":
	main()
