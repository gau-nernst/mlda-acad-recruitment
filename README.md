# MLDA Academics Recruitment

This task is meant for people without prior background in Machine Learning to have a taste of working on a Machine Learning project. There is no right or wrong answer for this task, as we just want to understand how you would approach and learn new Machine Learning concepts. For people with prior experience, this task is optional.

You should have basic programming skills to complete this task.

## Introduction to Image Classification and Transfer Learning

Convolution Neural Networks (CNNs) have revolutionized Computer Vision, enabling machines to see and understand the world just like humans. Your favourite TikTok or Instagram filters are [built with CNNs](https://www.linkedin.com/posts/bytedance_bytedance-augmentedreality-technology-activity-6811210320040742912-Ti5y), Tesla uses [CNNs in their self-driving cars](https://www.tesla.com/autopilotAI).

![Mask detection with CNN](https://www.programmersought.com/images/740/dfe069d8c4453ac17f01f8a6445da9f4.gif)

Mask detection with CNN. Source: [pyimagesearch](https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/)

In this task, you will explore how you can use CNNs to classify images. Simply put, given an image, your model will tell you what is the object inside the image!

Training a model from scratch may require a large amount of data and take a long time to train. In practice, people will train a model that has already been trained on a lot of generic images. This technique is known as **Transfer Learning**.

## Main task: Finetune a pretrained model

Your friend is an admin of a meme page. He asks if you can help him build a Machine Learning model to categorize memes from different games. To try out if this idea is feasible, you first want to classify [Doom and Animal Crossing memes](https://www.kaggle.com/andrewmvd/doom-crossing).

You will **finetune a pretrained model for image classification**. There are many resources online for this. Feel free to use any framework you prefer.

Some links to get you started:

- TensorFlow: [Transfer learning and fine-tuning](https://www.tensorflow.org/tutorials/images/transfer_learning)
- PyTorch: [Transfer learning for Computer vision tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

The dataset for this task is [here](https://www.kaggle.com/andrewmvd/doom-crossing).

You are recommended to use either [Google Colab](https://colab.research.google.com/) or [Kaggle notebook](https://www.kaggle.com/code) to train your model. They should be more than sufficient for this task.

## Optional exploration tasks

The following tasks are optional. You only need to discuss how you would approach these problems. If you have time, you can implement your proposed solutions and share with us your results!

1. What are the misclassified photos? How do you explain why they are misclassified? How can you help your model classify those photos correctly?

2. How do you know if your model has actually learned to classify the images? Or it's just very lucky at guessing.

3. What happen if you ask your model to classify a new image which is neither a Doom nor an Animal Crossing meme? You can try upload a random photo and run prediction on it to see the results. How would you modify your model to handle this situation?

4. After successfully trained a model to classify Doom and Animal Cross memes, you want your model to classify other memes also, such as Genshin Impact. How would you avoid retraining the model from scratch?

## Submission

You will submit your Jupyter notebook with your original work in it. You should note down your process in the notebook. Some pointers that you can include:

- Your thought process
- What you have tried; what worked and what didn't
- Your results i.e. accuracy

Note:

- We will not assess you based on the final accuracy, but how you approach the problem. You don't need to worry too much if your accuracy is not so good.
- You can use any tools or frameworks.
- Feel free to note down any ideas you have in mind but are not able to implement.
- You can reference and discuss with others, as long as the work is done by you.
