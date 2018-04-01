# FaceRecognition
OpenCV 3 &amp; Keras implementation of face recognition for specific people.


## Requirement
- Python 3.6    
- Tensorflow-gpu 1.5.0  
- Keras 2.1.3
- scikit-learn 0.19
- OpenCV 3.4

## Model

**Face recognition Model:** 

We use **MobileNetV2** as a feature extraction model. We input the paired face images and output the Euclidean distance between the two image features. The purpose of the training is to minimize the distance of the same category and maximize the distance of the different categories, so the use of the contrast loss as a loss function.

![face_net](/images/face_net.png)

## Experiment

Due to the limited computational resources, we used Face Recognition Data to train and test the model.
	
	device: Tesla K80
	dataset: Face Recognition Data - grimace (University of Essex, UK)
	optimizer: Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)  
	batch_szie: 40 

run the command below to show the experiment result.
```
python vis.py
```

### Train and val loss

![Loss](/images/loss.png)

### t-SNE for different people

The extracted face data is reduced to 2D features by t-SNE. It can be seen that the same face features are clustered together.

![tsne](/images/tsne.png)


### Features distance for different people

We use person 1 as the person to be identified, then compare the Euclidean distance between person 1 and the features of the other 4 individuals. It can be seen that it is closest to the features of another photograph of itself and far from other people.

![distance](/images/distance.png)

## Application

run the command below to train the model.
```
python train.py
```

The app use haar cascade classifier or SSD to detect face，use MobileNetv2 to recognize face.

run the command below to run the app to recognize a specific people from video camera.
 
Press **esc** to  quit window.
Press **space** 5 times to record different gestures of the face. The face id were saved at `data\key.npy`

```
python face.py
```