# Exploring survival during Pompeii eruption (Vesuivius)
<img width="500" height="200" alt="vesuve" src="https://github.com/user-attachments/assets/8e283fc8-0b4c-46d6-a46e-4e1fd37337d4" />


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


## **I)Introduction** ##
&emsp; -Motivation: This project's focus is on the tragedy that occured in Pompei, which could raise some questions. Why choose an event so ancient ? Well choosing Pompei as a first subject for this AI actually hold some very important reasons.

&emsp; For centuries, ancient ruins and civilization have been the center of archeology's attention : incas, azetcas, greeks and many others.  
&emsp; Studying Pompeii through artificial intelligence is not only about understanding a single event, but about discovering broader patterns in how human societies face disaster. By analyzing who survived and why, we can uncover factors such as social structure, location, or behavior that influenced survival. These insights could then be compared with other ancient civilizations to reveal how different cultures responded to natural catastrophes.
&emsp; In this way, Pompeii becomes a model, a starting point to explore how humanity has always adapted, resisted, or fallen in the face of nature’s power.

&emsp; By analyzing the survival factors in Pompeii, we may gain valuable insights into:

&emsp; - Social hierarchies and inequality — Did people from higher social classes have a better chance to survive because of their housing, mobility, or access to information?

&emsp; - Urban structure and geography — How did the design of the city influence survival? Were certain neighborhoods more exposed or more protected?

&emsp; - Human behavior during crisis — Can we detect patterns of reaction, cooperation, or panic that mirror what we see in modern disasters?

&emsp; - Cultural and societal resilience — What do these patterns reveal about the adaptability and organization of ancient Roman society?

&emsp; - Ultimately, this project aims to bridge archaeology and artificial intelligence, showing how modern tools can help us reinterpret ancient events and perhaps even compare them with other civilizations such as the Maya, the Egyptians, or the Minoans — all of which faced natural disasters in their history.
&emsp; By doing so, we can begin to see universal human patterns in the way societies face destruction and survival.

## **II)Datasets** ##
&emsp; The Vesuvius Survival dataset contains 500 observations, each representing an individual who lived near Mount Vesuvius during a simulated catastrophic event. It includes 11 variables that describe demographic, social, behavioral, and geographical aspects of these individuals. The primary goal of this dataset appears to be identifying the factors that influenced survival likelihood during the eruption. The dataset is entirely complete, with no missing values, making it clean and ready for analysis.

&emsp; The first column, PassengerId, is a simple numerical identifier ranging from 1 to 500. It serves solely as a unique key to distinguish individuals and carries no analytical significance. The key outcome variable is Survived, which indicates whether a person lived or died during the event. It is binary: 1 represents survival, while 0 indicates death. The mean value of 0.488 suggests that approximately 48.8% of the individuals survived, showing a nearly balanced distribution between survivors and non-survivors.

&emsp; The variable DistanceFromV represents the individual’s distance from Mount Vesuvius, likely measured in kilometers. The values range from 0.07 to 49.99 km, with an average of about 25 km. This variable captures a major geographical factor — the farther a person was from the volcano, the higher their chance of survival. It is likely to be one of the most influential predictors in the dataset.

&emsp; The Name column records each person’s full name, written in classical Roman style (for example, Marcus Domitius or Cassia Cornelius). There are 100 unique names across 500 individuals, suggesting that several people share the same family names, possibly representing related individuals or social clusters. While this variable is textual and not directly numerical, it might carry implicit information about family groupings.

&emsp; The Sex variable denotes the individual’s biological sex, with two categories: Male and Female. The dataset includes 267 males (53.4%) and 233 females (46.6%), indicating a roughly balanced gender distribution. Gender may indirectly influence survival through social roles, responsibilities, or behavioral differences in crisis situations.

&emsp; The Age variable provides the age of each individual, ranging from 5 to 79 years old, with an average of 35.2 and a standard deviation of 14.3 years. The population is therefore mostly adult, but includes both children and the elderly. The distribution is centered around middle-aged adults, though extreme age groups may represent more vulnerable individuals with lower survival probabilities.

&emsp; The WealthIndex variable measures socio-economic standing on a scale from 0 to 100. The mean value of 50 and a relatively high standard deviation of 29.7 indicate a diverse population with varying levels of wealth. This variable likely correlates with social status and access to protective resources, such as shelters or the means to evacuate early.

&emsp; The ShelterAccess column indicates whether the person had access to a shelter during the eruption. It is binary, where 1 denotes shelter access and 0 means none. About 41.2% of individuals had access to a shelter, making this one of the most critical variables related to survival, since physical protection would directly impact one’s chance of living through the disaster.

&emsp; The HasPet variable specifies whether a person owned a pet, also expressed as a binary indicator. Around 27.8% of people reported owning a pet. While seemingly minor, this factor may reveal behavioral tendencies — for example, pet owners might delay evacuation to rescue their animals, potentially affecting survival outcomes.

&emsp; The ReactionTime variable measures how quickly an individual responded to the eruption, expressed in seconds. Reaction times range from 1.11 to 19.98 seconds, with an average of 10.52 seconds. This behavioral variable captures psychological readiness and decision-making speed: shorter reaction times likely correspond to higher survival probabilities, as quicker individuals would have more time to act effectively.

&emsp; Finally, the Status variable categorizes each person’s social class into three groups: Citizen, Slave, and Noble. Citizens make up the majority (58.4%), followed by slaves (24.2%) and nobles (17.4%). This feature reflects the Roman social hierarchy and likely correlates with wealth and shelter access. For instance, nobles may have benefited from better infrastructure or quicker means of evacuation, improving their odds of survival.

&emsp; In summary, the Vesuvius Survival dataset blends geographical (DistanceFromV), socio-economic (WealthIndex, Status, ShelterAccess), demographic (Age, Sex), and behavioral (HasPet, ReactionTime) factors to model human survival during a natural disaster. Its balanced structure and diverse variables make it ideal for multivariate analysis aimed at uncovering how social class, wealth, distance, and human behavior interact to determine survival outcomes in catastrophic scenarios.

## **III)Methodology** ##
&emsp; -Explaining your choice of algorithms (methods, any modelsfrom AIML libraries)
&emsp; -Explaining features or code (if any)

## **IV)Evaluation & Analysis** ##
&emsp; -Graphs, tables, any statistics (if any)

## **V)Related Work (e.g., existing studies)** ##
&emsp; -Tools, libraries, blogs, or any documentation that you have used to do this project.

## **VI)Conclusion: Discussion** ##
