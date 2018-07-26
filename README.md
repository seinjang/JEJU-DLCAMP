# JEJU-DLCAMP

This repository is about the project during **DEEP LEARNING CAMP JEJU ( 2018. 7. 1 ~ 30 )**

## Motivation of Project

Jeju Island POI(point-of-interst) [Dataset](https://www.data.go.kr/dataset/15004770/fileData.do)
<p align="center">
    <img src="Figure/Jeju_all_restarant.png" height="350"/>
</p>

Sometime it really hard to find proper place to go.


What if, the Model can understand where do i want to go and understand location of the place in the map ?
<br/>
The Model will be able to recommend a suitable place according to the **Location** you are currently in and **Keyword** you want to go.
<br/>


## Model Architecture
<p align="center">
    <img src="Figure/first_model.png" height="350"/>
</p>

We using Relation Networks Model basically contains **CNN** for Image processing, and **RN** for Relational reasoning between two object(Image, Test).<br/><br/><br/>



## Dataset

The data contains the following information

* User Location
* Image Location (Center, Boundary)
* Place Information (type [Restaurant, Cafe, ...], Location [Latitude, Longitude])


At the beginning of the project, I decided to only consider the POI for restaurant and cafe in Jeju Island. Each **Image** has a number of POIs which marked with red(restaurant), green(cafe) marker.

Also, the center(user position) **location** of the image, boundary location information and the location information of restaurant and cafe in Jeju Island are required for input.

Among the various **Keyword**, I will focus on the following Keyword first.<br/>
* Restaurant ?
* Cafe ?
* Beachfront ?
* Near ?
* Cafe nearby restaurant ?
* Restaurant nearby cafe ?

The possible **Answer** is a softmax vector whose with probability for POIs

*[probability of POI 1, probability of POI 2, probability of POI 3,...,probability of POI n]*


## 2 - Class Dataset
<p align="center">
    <img src="Figure/2class_dataset.png" height="350"/>
</p>

## 10 - Class Dataset
<p align="center">
    <img src="Figure/10class_dataset_1.png" height="350"/>
</p>
