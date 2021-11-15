# Self-Driving-Car
My Keras implementation of NVIDIA's 2016 paper-"End to End Learning for Self Driving Cars"

## Demo

https://user-images.githubusercontent.com/57484266/137711451-d60aea39-f177-42c5-983d-6685eae420ce.mp4


https://user-images.githubusercontent.com/57484266/137711465-586dcfc0-956b-4a2e-883a-1e11e994b8bc.mp4



## Description

Link to the paper- https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

The CNN architecture trained-

![architecture](https://user-images.githubusercontent.com/57484266/137711239-fb404788-df06-44cd-b346-90ed2d472c47.PNG)


The model trained takes the image from the front camera as the input and predicts the steering wheel angle. 

I have trained the model on a dataset of images collected from 25 minutes of driving by Sully Chen- https://github.com/SullyChen

If you have high computational model you can train the same model on bigger datasets provided by Comma.ai (https://github.com/commaai/comma2k19) and 
Udacity (https://github.com/udacity/self-driving-car-sim) to get better results.


The training process is detailed in the "final.ipynb" notebook. Run the "self_driving.py" file to check the performance of the saved model in an interactive manner.


The images before and after pre-processing-

![before](https://user-images.githubusercontent.com/57484266/137711445-6ee9bb66-47eb-4421-85a6-66830c000666.PNG)  ![after](https://user-images.githubusercontent.com/57484266/137711248-0a7960f5-eb42-4d2c-85a8-c5e0d316a6d0.PNG)


