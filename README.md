# JEJU-DLCAMP

This repository is about the project during **DEEP LEARNING CAMP JEJU ( 2018. 7. 1 ~ 30 )**

## Motivation of Project

What if, the Model can understand the map ?
  * Near / Far
  * Beach / Land
  
  
What if, the Model can understand question ?
  * Where is the nearest beachfront restaurant?
  * Where is the restaurant with a nearby cafe?
  
<br/>
The model will be able to recommend a suitable place according to our questions.

## First Draft Model

### Model Architecture
<p align="center">
    <img src="Figure/first_model.png" height="350"/>
</p>

Our model contains 3 Deep-Learning techniques, which **CNN** for Image processing, **LSTM** for Text(Question) processing and
**RN** for Relational reasoning between two object.<br/><br/><br/>


### Dataset

Jeju Island POI(point-of-interst) [Dataset](https://www.data.go.kr/dataset/15004770/fileData.do)
<p align="center">
    <img src="Figure/Jeju_all_restarant.png" height="350"/>
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
