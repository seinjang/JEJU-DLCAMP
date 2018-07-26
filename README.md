# JEJU-DLCAMP

This repository is about the project during **DEEP LEARNING CAMP JEJU ( 2018. 7. 1 ~ 30 )**

## Motivation of Project

Jeju Island POI(point-of-interst) [Dataset](https://www.data.go.kr/dataset/15004770/fileData.do)
<p align="center">
    <img src="Figure/Jeju_all_restarant.png" height="350"/>
</p>

Sometime it really hard to find proper place to go.


What if, the Recommendation Model can understand where do i want to go and understand location of the place in the map ?

<br/>
The model will be able to recommend a suitable place according to the **Map** you are currently in and **Keywork** you want to go.



### Model Architecture
<p align="center">
    <img src="Figure/first_model.png" height="350"/>
</p>

We using Relation Networks Model basically contains **CNN** for Image processing, and **RN** for Relational reasoning between two object(Image, Test).<br/><br/><br/>


### Dataset
1) 2 - Class Dataset
<p align="center">
    <img src="Figure/2class_dataset.png" height="350"/>
</p>

2) 10 - Class Dataset
<p align="center">
    <img src="Figure/10class_dataset.png" height="350"/>
</p>
The data contains the following information
* Latitude
* Longitude
* Place type (Restaurant, Cafe, ...)
* Place name
* Place Adress<br/><br/>

At the beginning of the project, I decided to only consider the POI for restaurants in Jeju Island. Each **Image** has a number of 
POIs which marked with red marker. Here are some examples of images.
<img src="/Figure/image_sample.png" alt="drawing"/>

Also, the center(user position) **location** of the image, boundary location information and the location information of all restaurants POI in Jeju Island are required for input.

*{center : latitude, longitude}<br/>
{boundary : latitude(left), latitude(right), longitude(up), longitude(down)}<br/>
{POI 1: latitude, longitude}, {POI2 : latitude, longitude},...*


Among the various **Questions**, I will focus on the following questions first.<br/>
* Where is the beachfront restaurant ?
* Where is the nearest restaurant ?
* Where is the restaurant with a nearby cafe ?
* Where is the beachfront restaurant with a nearby cafe ?

The possible answer is a softmax vector whose with probability for all POIs

*[probability of POI 1, probability of POI 2, probability of POI 3,...,probability of POI n]*
