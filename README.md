# Classify-14-different-Simpsons-Characters
This project is writing a Pytorch program that learns to classify 14 different Simpsons Characters using the grey scale images.
![image](https://user-images.githubusercontent.com/85326183/130592134-218cfe22-6556-4bbd-992e-cdaefdc94ddf.png)

# The provided file hw2main.py handles the following:

* loading the images from the data directory
* splitting the data into training and validation sets (in the ratio specified by train_val_split)
* data transformation: images are loaded and converted to tensors; this allows the network to work with the data; you can optionally modify and add your own transformation steps, and you can specify different transformations for the training and testing phase if you wish
* loading the data using DataLoader() provided by pytorch with your specified batch_size in student.py
* You should aim to keep your code backend-agnostic in the sense that it can run on either a CPU or GPU. This is generally achieved using the .to(device) function.

If you do not have access to a GPU, you should at least ensure that your code runs correctly on a CPU.
Please take some time to read through hw2main.py and understand what it does.

# At the top of the code, in a block of comments,there is a brief answer (about 300-500 words) to this Question:
Briefly describe how your program works, and explain any design and training decisions you made along the way.

You should try to cover the following points in your Answer:
* choice of architecture, algorithms and enhancements (if any)
* choice of loss function and optimiser
* choice of image transformations
* tuning of metaparameters
* use of validation set, and any other steps taken to improve generalization and avoid overfitting
