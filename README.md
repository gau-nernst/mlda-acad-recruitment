# MLDA Academics Recruitment

This task serves to give people without prior background in Machine Learning (ML) a taste of what working on a Machine Learning project is like. There is no right or wrong response to this task, as we only seek to understand how you would approach an ML problem and learn new ML concepts. For people with prior experience, this task is optional.

You should have basic programming skills to complete this task.

If you have any questions, don't hesitate to contact me (Thien, MLDA Academics Head) at tran0096@e.ntu.edu.sg

**UPDATE**: suggested solution has been uploaded. Click the button below to open it in Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gau-nernst/mlda-acad-recruitment/blob/master/Animal_crossing_vs_Doom.ipynb)

## Introduction to Image Classification and Transfer Learning

Convolution Neural Networks (CNNs) have revolutionized Computer Vision, enabling machines to see and understand the world just like humans. CNNs not only power [self-driving capabilities](https://www.tesla.com/autopilotAI) of Tesla cars, but they are also a [fundamental part](https://www.linkedin.com/posts/bytedance_bytedance-augmentedreality-technology-activity-6811210320040742912-Ti5y) of your favourite TikTok or Instagram filters.

![Mask detection with CNN](https://www.programmersought.com/images/740/dfe069d8c4453ac17f01f8a6445da9f4.gif)

Mask detection with CNN. Source: [pyimagesearch](https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/)

In this task, you will explore how you can use CNNs to classify images. Simply put, given an image, your model will tell you what the object inside the image is!

Training a model from scratch may require a large amount of data and take a long time to train. In practice, people train a model that has already been trained on a lot of generic images. This technique is known as **Transfer Learning**.

## Main task: Finetune a pretrained model

Your friend is an admin of a meme page. He asks if you can help him build a Machine Learning model to categorize memes from different games. To try out if this idea is feasible, you first want to classify [Doom and Animal Crossing memes](https://www.kaggle.com/andrewmvd/doom-crossing).

You will **finetune a pretrained model for image classification** of Doom and Animal Crossing memes. There are many online resources to help you with this task. Feel free to use any framework that you prefer.

Some links to get you started:

- TensorFlow: [Transfer learning and fine-tuning](https://www.tensorflow.org/tutorials/images/transfer_learning)
- PyTorch: [Transfer learning for Computer vision tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

The dataset for this task is [here](https://www.kaggle.com/andrewmvd/doom-crossing).

You are recommended to use either [Google Colab](https://colab.research.google.com/) or [Kaggle notebook](https://www.kaggle.com/code) to train your model. They should be more than sufficient for this task.

## Optional exploration tasks

The following tasks are optional. You only need to discuss how you would approach these problems. If you have time, you can implement your proposed solutions and share with us your results!

1. Which are the misclassified photos? How do you explain why they are misclassified? How can you help your model classify those photos correctly?

2. How do you know if your model has actually learned to classify the images? Or is it simply very lucky at guessing?

3. What happens if you ask your model to classify a new image which is neither a Doom nor an Animal Crossing meme? You can try to upload a random photo and generate prediction on it to see the results. How would you modify your model to handle this situation?

4. After having successfully trained a model to classify Doom and Animal Cross memes, you want your model to classify other memes as well, such as those from Genshin Impact. How would you avoid retraining the model from scratch?

## Submission

You will submit a Jupyter notebook with your original work in it. You should note down your process in the notebook. Some pointers that you can include:

- Your thought process
- What you have tried; what worked and what didn't
- Your results i.e. accuracy

There is no submission deadline. You only need to submit before your scheduled interview. Please send your submission via email together with your particulars (full name and matriculation number)

- Send to this email: tran0096@e.ntu.edu.sg
- Email subject: [MLDA Recruitment] Academics Optional Task submission

Note:

- We will not assess you based on the final accuracy, but how you approach the problem. You don't need to worry too much if your accuracy is not so good.
- You can use any tools and frameworks.
- Feel free to note down any ideas you have in mind but are not able to implement.
- You can reference and discuss with others, as long as the work submitted is done by you.
