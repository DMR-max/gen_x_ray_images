# Deep learning algorithms for training data: focused on chest X-ray images
This readme explains how to run the experiments.
Steps to train one algorithm:
1. install python, pytorch, matplotlib, huggingface libraries from the requirements.txt file.
2. launch either gan_rework for the GAN, Stable_diff_rework for the diffusion model. VAE_rework for the VAE, autoregressive_rework for the autoregressive models.
3. BEWARE IF THE OPTIONS IF GIVEN PLEASE CHOOSE A DIFFERENT LOCATION FOR SAVING THE MODELS EACH TIME

Steps to run comparison between one algorithm:
0. Make sure you have ran the GAN model for the image size before proceeding, the discriminator_49.pth version is important.
1. run algorithm with python
2. choose different folder for the different hyperparameters you want to test (important)
3. follow the instructions on screen
(This program ran on a Windows 11 Desktop)

Steps to run the comparison between all algorihms
0. Make sure you have ran the 32 image size versions of the diffuison model and the VAE. Also make sure to have ran the baseline versions of the autoregressive model and the GAN.
1. Run discvsall.py for the discriminator vs all the other models, Run the classifier.py for the classifier.

In case there are bugs in the program please run the files from the in case of bugs folder and manually change the parameters.
