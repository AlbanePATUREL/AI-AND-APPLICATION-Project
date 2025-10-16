# Exploring survival during Pompeii eruption (Vesuivius)
<img width="755" height="505" alt="vesuve" src="https://github.com/user-attachments/assets/8e283fc8-0b4c-46d6-a46e-4e1fd37337d4" />


**Group members :**

|        Name       |        Major           |        Email              |         
|-------------------|------------------------|---------------------------|
| Lucas VALERION    | Computer science       |lucas.valerion@gmail.com   |
| Thomas BUCHLER    | Computer science       |thomas.buchler@edu.ece.fr |
| Albane PATUREL    | Computer science       |albane.paturel@edu.ece.fr  |
| Zoé BOUSQUENAUD   | Computer Science       |zoe.bousquenaud@gmail.com  |


## **Proposal :**

&emsp;Pompeii was one of the most beautiful cities of the Roman Empire. With its busy streets, luxurious villas, and colorful frescoes, it offered everything for a prosperous life under the shadow of Mount Vesuvius.

However, in 79 AD, disaster struck when Vesuvius violently erupted, covering the city in ash and pumice. Thousands perished, and Pompeii disappeared for centuries beneath volcanic debris.

Today, archaeologists use advanced technology, including AI-based tools, to study the ruins and predict volcanic activity. Our goal is to build a model that analyzes eruption data to better understand and prevent future catastrophes.

Kaggle datasets we will use : 
- https://www.kaggle.com/datasets/jessemostipak/volcano-eruptions
- https://www.kaggle.com/code/mrisdal/exploring-survival-on-the-titanic/report

&emsp;Link on history we will use : https://en.wikipedia.org/wiki/Pompeii

## **Description of the Dataset :**

&emsp;This dataset includes 11 columns: PassengerId, Survived, DistanceFromV, Name, Sex, Age, WealthIndex, ShelterAccess, HasPet, ReactionTime, and Status. It contains numerical features like age, distance from Vesuvius, and wealth index, as well as boolean variables such as survival and shelter access.

&emsp;We created this dataset to simulate the situation of individuals during the Vesuvius eruption and explore how different factors may have influenced their survival. Some details, like injury level, were intentionally excluded as they were not relevant to our project’s objectives. We will use randomForest to create a model predicting survival during Vesuvius eruption.

