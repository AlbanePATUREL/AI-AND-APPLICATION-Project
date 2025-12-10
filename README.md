# Exploring survival during Pompeii eruption (Vesuivius)
<img width="500" height="200" alt="vesuve" src="https://github.com/user-attachments/assets/8e283fc8-0b4c-46d6-a46e-4e1fd37337d4" />


**Group members :**

|        Name       |        Major           |        Email              |         
|-------------------|------------------------|---------------------------|
| Lucas VALERION    | Computer science       |lucas.valerion@gmail.com   |
| Thomas BUCHLER    | Computer science       |thomas.buchler@edu.ece.fr  |
| Albane PATUREL    | Computer science       |albane.paturel@edu.ece.fr  |
| Zoé BOUSQUENAUD   | Computer Science       |zoe.bousquenaud@gmail.com  |

Video link : 


## **Proposal :**

&emsp;Pompeii was one of the most beautiful cities of the Roman Empire. With its busy streets, luxurious villas, and colorful frescoes, it offered everything for a prosperous life under the shadow of Mount Vesuvius. However, in 79 AD, disaster struck when Vesuvius violently erupted, covering the city in ash and pumice. Thousands perished, and Pompeii disappeared for centuries beneath volcanic debris.

Today, archaeologists use advanced technology, including AI-based tools, to study the ruins and predict volcanic activity. Our goal is to build a model that analyzes eruption data to better understand and prevent future catastrophes.

Kaggle dataset we will use : 
- https://www.kaggle.com/code/mrisdal/exploring-survival-on-the-titanic/report

&emsp;Link on history we will use : https://en.wikipedia.org/wiki/Pompeii


## **I)Introduction** ##
&emsp; This project's focus is on the tragedy that occured in Pompei, which could raise some questions. Why choose an event so ancient ? Well choosing Pompei as a first subject for this AI actually hold some very important reasons.

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

&emsp; The key outcome variable is Survived, which indicates whether a person lived or died during the event. It is binary: 1 represents survival, while 0 indicates death. 

&emsp; The first column, CivilianId, is a simple numerical identifier ranging from 1 to 500. It serves solely as a unique key to distinguish individuals and carries no analytical significance.

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
The dataset used in this study was first imported and examined to understand its structure and content. An initial exploration was conducted by viewing sample rows, inspecting variable types, and generating descriptive statistics. This stage made it possible to identify irrelevant fields such as CivilianId and Name, which were removed because they carried no predictive value. The exploration also provided a preliminary understanding of the distribution of key factors such as age, distance from Vesuvius, wealth index, reaction time, and social status, helping form hypotheses about which variables might influence survival likelihood.

Before training the predictive model, the dataset underwent a series of preprocessing steps. All categorical variables, including sex, social status, and gender, were encoded into numerical values using a label encoder to ensure compatibility with machine learning algorithms. The dataset was then split into features (X) and target (y), with the target variable representing survival. To ensure fair comparison between variables and avoid scale imbalances, all numerical features were standardized with a StandardScaler. The observations were divided into training and testing sets using an 80/20 split, stratified on the target variable to maintain a representative distribution of survivors and non-survivors in both subsets.

A Random Forest Classifier was selected as the model for prediction due to its robustness, ability to handle heterogeneous variables, and capacity to capture complex nonlinear relationships. The model was trained using 200 decision trees with a maximum depth of 10, with additional regularization parameters ensuring that the model did not overfit. Once trained, the model was evaluated using several metrics: accuracy, precision, recall, F1-score, and a confusion matrix. These metrics provided a detailed view of the model’s performance and its ability to correctly classify instances of survival and non-survival.

After validation, the trained model was applied to the entire dataset in order to generate survival predictions for every individual. For each observation, both a binary prediction and a survival probability were produced. These results were then saved into a new CSV file so they could be analyzed further or included in the final report. To deepen the interpretation of the findings, visualizations were generated, including survival probability as a function of distance from Vesuvius, feature importance as determined by the Random Forest, correlation matrices, and comparisons between actual and predicted survival outcomes. Additional graphs illustrating survival patterns across age groups, gender, reaction times, and social classes were created to highlight underlying trends in the data.

Taken together, this methodological approach follows a complete data science pipeline, beginning with exploration, followed by preprocessing, modeling, evaluation, and visualization. This ensures both methodological rigor and interpretability, allowing for a clear understanding of the factors influencing survival and for an assessment of the model’s reliability in predicting outcomes within the simulated dataset.

1. Loading and Exploring the Dataset

The randomforest.py script begins by loading the CSV and printing basic information about the data, such as column types and summary statistics. This helps verify that the dataset is clean and ready for analysis.
```
def load_and_explore_data(csv_file):
    df = pd.read_csv(csv_file)
    print(df.head())
    print(df.info())
    print(df.describe())
    return df
```
2. Preparing the Data for Machine Learning

Next, unnecessary columns like CivilianId and Name are removed. Categorical variables such as Sex, Status, and Gender are encoded into numerical values, and the dataset is split into features (X) and labels (y).
```
def prepare_data(df):
    df = df.drop(columns=['CivilianId', 'Name'], errors='ignore')
    encoder = LabelEncoder()
    for col in ['Sex', 'Status', 'Gender']:
        if col in df.columns:
            df[col] = encoder.fit_transform(df[col].astype(str))
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    return X, y
```
3. Training the Random Forest Model

The data is standardized, split into training and testing sets, and then passed into a Random Forest classifier. Parameters such as depth and minimum samples per split are controlled to avoid overfitting.
```
def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))
    return model, scaler
```
4. Predicting Survival for the Entire Dataset

After training, the model is used to predict survival outcomes for the full dataset.
```
def predict_full_dataset(model, scaler, X):
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    return predictions
```
5. Saving Predictions to a CSV

The predictions are saved to a CSV file so they can be used or visualized later.
```
def save_predictions(df, predictions, filename="predictions.csv"):
    df_out = pd.DataFrame({
        "CivilianId": df.get("CivilianId", range(len(predictions))),
        "PredictedSurvived": predictions
    })
    df_out.to_csv(filename, index=False)
```
6. Creating Machine Learning Visualizations

The script generates several visualizations: a correlation matrix, feature-importance plots, and graphs relating predictions to variables like age and distance.
```
def create_visualizations(df, predictions, out_dir="visualizations"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.savefig(os.path.join(out_dir, "correlation_matrix.png"))
    plt.close()
```
7. The Main Pipeline

Finally, the script ties everything together in one workflow.
```
def main():
    df = load_and_explore_data("vesuvius_survival_dataset.csv")
    X, y = prepare_data(df)
    model, scaler = train_model(X, y)
    predictions = predict_full_dataset(model, scaler, X)
    save_predictions(df, predictions)
    create_visualizations(df, predictions)
```
## **IV)Evaluation & Analysis** ##

&emsp;1)Global Survival rate

&emsp; <img width="500" height="400" alt="grapheSurvivalRate" src="plots\01_overall_survival_rate.png" />
48.8% global survival rate

Almost half the population survived
This bar chart shows the overall survival rate of the population in the dataset.
The global survival rate is 48.8%, meaning that slightly less than half of the individuals survived the event (e.g., the volcanic eruption).
The Y-axis represents the survival ratio from 0 to 1.

Code for Survival rate by Age graph : 
```
def plot_survival_by_age_tens(data, out_dir):
    """Survival rate by age decade"""
    age = pd.to_numeric(data["Age"], errors="coerce")
    survived = pd.to_numeric(data["Survived"], errors="coerce")
    df = pd.DataFrame({"Age": age, "Survived": survived}).dropna()
    
    if df.empty:
        return
    
    df["age_decade"] = (df["Age"] // 10).astype(int) * 10
    grp = df.groupby("age_decade")["Survived"].mean().sort_index()
    labels = [f"{int(d)}-{int(d)+9}" for d in grp.index]
    
    fig = plt.figure(figsize=(10, 6))
    plt.plot(range(len(grp)), grp.values, marker="o", linewidth=2, markersize=8, color='darkblue')
    plt.xticks(range(len(grp)), labels, rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Survival rate (0-1)")
    plt.xlabel("Age (decades)")
    plt.title("Survival Rate by Age", fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    out_file = out_dir / "02_survival_by_age_tens.png"
    fig.savefig(out_file, bbox_inches="tight", dpi=140)
    plt.show()
    print(f"✓ Chart saved: {out_file}")
```

To create this visualization, the code first transforms the continuous Age variable into decadal bins using integer division (floor division by 10, multiplied by 10). This creates distinct cohorts (e.g., 20s, 30s). We then utilize the Pandas groupby() function to aggregate these cohorts and calculate the .mean() of the binary 'Survived' column (0 or 1). This mathematical operation effectively converts the binary data into a survival probability rate for each age group. The trend is then plotted using Matplotlib with distinct markers to highlight the variation between decades.

&emsp; <img width="500" height="400" alt="graphesurvivalAge" src="plots\02_survival_by_age_tens.png" />
This line chart illustrates how survival varies by age decade (0–9, 10–19, etc.)

Children (0–9) and teens (10–19) have survival rates around 45–47%.
Young adults (20–39) show slightly higher survival rates, reaching around 50–53%.
Middle-aged groups (40–59) drop back to around 44–46%.
Older adults (70–79) show the highest survival rate, reaching ~71% in this dataset.

This suggests that survival does not follow a simple decreasing trend with age; instead, it varies depending on specific circumstances of the population.

&emsp; <img width="500" height="400" alt="grapheSurvivalDistance" src="plots\03_survival_by_distance_tens.png" />
This graph shows how survival changes based on the distance from the volcano, grouped in 10-km brackets.
There is a clear positive correlation:
At 0–9 km, survival is very low (~20%).
Survival increases steadily with distance.
At 40–49 km, the survival rate reaches ~75%.

This indicates that distance from the eruption center was one of the strongest predictors of survival:
=> The farther away people were, the more likely they were to survive.

Code for Survival rate by Gender graphe :
```
def plot_survival_by_gender(data, out_dir):
    """Survival rate by gender"""
    gender_col = "Sex" if "Sex" in data.columns else "Gender"
    survived = pd.to_numeric(data["Survived"], errors="coerce")
    df = pd.DataFrame({"Gender": data[gender_col], "Survived": survived}).dropna()
    
    grouped = df.groupby("Gender")["Survived"].mean()
    
    fig = plt.figure(figsize=(8, 6))
    plt.bar(grouped.index, grouped.values, color=['lightcoral', 'lightblue'], alpha=0.7)
    plt.ylim(0, 1)
    plt.ylabel("Survival rate (0-1)")
    plt.xlabel("Gender")
    plt.title("Survival Rate by Gender", fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    out_file = out_dir / "04_survival_by_gender.png"
    fig.savefig(out_file, bbox_inches="tight", dpi=140)
    plt.show()
    print(f"✓ Chart saved: {out_file}")
```
To create this graph, the script first identifies the gender column. It then uses the Pandas groupby() function to split the data into two groups: Male and Female. To find the survival rate, the code simply calculates the average (.mean()) of the 'Survived' column. Since the data uses 0 for deceased and 1 for survivor, the average gives us the exact percentage of survivors. Finally, the results are displayed as a bar chart using specific colors (light coral and light blue) to easily distinguish between the two groups.

&emsp; <img width="500" height="400" alt="grapheSurvivalGender" src="plots\04_survival_by_gender.png" />

This graph shows how survival varies between men and women. The two proportions are almost identical, indicating no meaningful difference in survival rates between genders. Any small variations fall within what could be expected by chance.
This suggests that gender was not a significant predictor of survival during the event: → Men and women survived at roughly the same rate.

&emsp; <img width="500" height="400" alt="grapheSurvivalReacTime" src="plots\05_survival_by_reaction_time.png" />

This graph displays survival outcomes based on reaction time categories. Across all groups—from fast responders to slower ones—the survival percentages remain very similar. There is no clear upward or downward trend.
This indicates that reaction time did not play a major role in determining survival: → People survived at comparable rates regardless of how quickly they reacted.

&emsp; <img width="500" height="400" alt="grapheSurvivalStatus" src="plots\06_survival_by_status.png" />

This graph shows how survival rates change according to social status. A clear negative pattern is visible: individuals with lower social status have noticeably higher survival rates, while those of higher status show reduced survival.
This suggests that social status was a strong predictor of survival: → The lower someone’s social status, the more likely they were to survive.

Code for the graph below :
```
def create_visualizations(data, X, y, rf_model, scaler, X_test, y_test, 
                         y_pred, y_pred_proba, out_dir="plots"):
    """Create all visualizations"""
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True, parents=True)
    
    # Prepare data for charts - encoding for correlation
    data_clean = data.drop(['CivilianId', 'Name'], axis=1, errors='ignore').copy()
    
    # Encode categorical columns for correlation
    for col in data_clean.select_dtypes(include=['object']).columns:
        if col in data_clean.columns:
            le = LabelEncoder()
            data_clean[col] = le.fit_transform(data_clean[col])
    
    correlation_matrix = data_clean.corr(numeric_only=True)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE")
    print("="*70)
    print(feature_importance)
    
    # Figure 1: 4 main model charts
    fig1, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Survival probability vs Distance
    if 'DistanceFromV' in X.columns:
        distance_idx = list(X.columns).index('DistanceFromV')
        distances_test = X_test[:, distance_idx] * scaler.scale_[distance_idx] + scaler.mean_[distance_idx]
        survival_proba_test = y_pred_proba[:, 1]
        
        sort_idx = np.argsort(distances_test)
        distances_sorted = distances_test[sort_idx]
        proba_sorted = survival_proba_test[sort_idx]
        
        axes[0, 0].scatter(distances_test[y_test == 0], survival_proba_test[y_test == 0], 
                           alpha=0.6, c='red', label='Actually Deceased', s=50, edgecolors='darkred')
        axes[0, 0].scatter(distances_test[y_test == 1], survival_proba_test[y_test == 1], 
                           alpha=0.6, c='green', label='Actually Survived', s=50, edgecolors='darkgreen')
        
        window = 10
        if len(distances_sorted) >= window:
            proba_smooth = uniform_filter1d(proba_sorted, size=window)
            axes[0, 0].plot(distances_sorted, proba_smooth, 'b-', linewidth=2.5, 
                            label='Trend', alpha=0.8)
        
        axes[0, 0].axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='50% Threshold')
        axes[0, 0].set_title('Survival Probability vs Distance from Vesuvius', fontweight='bold', fontsize=12)
        axes[0, 0].set_xlabel('Distance from Vesuvius (km)')
        axes[0, 0].set_ylabel('Survival Probability')
        axes[0, 0].legend(loc='best')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(-0.05, 1.05)
    
    # 2. Feature importance
    feature_importance.plot(x='feature', y='importance', kind='barh', ax=axes[0, 1])
    axes[0, 1].set_title('Feature Importance', fontweight='bold', fontsize=12)
    axes[0, 1].set_xlabel('Importance')
    
    # 3. Correlation matrix
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=axes[1, 0], vmin=-1, vmax=1, 
                cbar_kws={'label': 'Correlation coefficient'})
    axes[1, 0].set_title('Correlation Matrix between Variables', fontweight='bold', fontsize=12)
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
    axes[1, 0].set_yticklabels(axes[1, 0].get_yticklabels(), rotation=0)
    
    # 4. Actual vs Predicted comparison
    axes[1, 1].scatter(range(len(y_test)), y_test, alpha=0.5, label='Actual', s=30)
    axes[1, 1].scatter(range(len(y_pred)), y_pred, alpha=0.5, label='Predicted', s=30, marker='x')
    axes[1, 1].set_title('Comparison: Actual vs Predicted Survival', fontweight='bold', fontsize=12)
    axes[1, 1].set_xlabel('Sample index')
    axes[1, 1].set_ylabel('Survival (0=No, 1=Yes)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig1.savefig(out_path / "00_model_analysis.png", dpi=140, bbox_inches='tight')
    plt.show()
    print(f"\n✓ Chart saved: {out_path / '00_model_analysis.png'}")
    
    # Individual charts by factor
    plot_survival_rate(y, out_path)
    plot_survival_by_age_tens(data, out_path)
    plot_survival_by_distance_tens(data, out_path)
    
    if 'Sex' in data.columns or 'Gender' in data.columns:
        plot_survival_by_gender(data, out_path)
    
    if 'ReactionTime' in data.columns:
        plot_survival_by_reaction_time(data, out_path)
    
    if 'Status' in data.columns:
        plot_survival_by_status(data, out_path)
```
  :
To construct this complex visualization, the script initializes a Matplotlib figure with a 2x2 subplot grid, allowing for the simultaneous display of four distinct analytical dimensions. For the survival probability chart, the code segregates test data by outcome to color-code the scatter plot and applies a uniform_filter1d function to compute a rolling average, generating the smooth blue trend line that visualizes probability distribution. The feature importance chart is derived directly from the trained Random Forest attributes, while the correlation matrix utilizes the Pandas corr() method rendered through a Seaborn heatmap with annotated coefficients. The final verification chart is achieved by superimposing two scatter plots sharing the same index, enabling a direct visual comparison between the ground truth and the model's predictions.

&emsp; <img width="500" height="400" alt="graphe4" src="plots\00_model_analysis.png" />

## **V)Related Work (e.g., existing studies)** ##
In this project, the goal was to analyze historical and geological data related to the eruption of Pompeii using modern machine learning tools. To achieve this, several Python libraries were combined to form a complete processing pipeline. Each library plays a specific role, from data collection to visualization and predictive modeling, ensuring that the analysis is both efficient and scientifically reliable.

The project begins with argparse and Path, which make the program flexible and easy to run from the command line. These tools allow the user to specify file locations or configuration options without modifying the code itself. This makes the system more robust and adaptable to different datasets or environments.

For data manipulation, the project relies heavily on pandas and numpy. These two libraries are essential for loading the dataset, cleaning missing values, transforming variables, and performing mathematical operations. With their powerful data structures, they make it possible to prepare the eruption dataset in a way that is both structured and efficient.

To visualize the data, the project uses matplotlib.pyplot and seaborn. These libraries allow the creation of detailed and informative graphs that help reveal trends in volcanic activity, correlations between variables, and patterns hidden inside the dataset. Whether it is heatmaps, histograms, or line plots, the visual component is crucial for understanding the behavior leading up to the eruption.

For the predictive modeling part, the project integrates several tools from scikit-learn. The function train_test_split divides the dataset into training and testing sets to evaluate how well the model generalizes. The RandomForestClassifier is then used to build a machine learning model capable of predicting categories or events based on the available features. To measure performance, metrics such as accuracy_score, classification_report, and confusion_matrix provide a complete evaluation of the model’s strengths and weaknesses.

Additionally, the project uses preprocessing tools such as StandardScaler and LabelEncoder, which help normalize numerical features and convert categorical variables into numerical values. These steps are essential to ensure that the machine learning algorithms interpret the data correctly and perform at their best.

Finally, the SciPy function uniform_filter1d is applied to smooth certain time-series data. This makes long-term volcanic trends more visible by reducing noise or abrupt variations in the dataset. Thanks to this smoothing step, the graphs become cleaner and easier to interpret.

Together, all these libraries form the backbone of the project. They provide a strong foundation to explore, visualize, and model historical eruption data with modern AI techniques. This combination of tools turns raw geological information into meaningful insights that help us better understand the Pompeii eruption from a data-driven perspective.
## **VI)Conclusion: Discussion** ##

This project demonstrates how artificial intelligence can help reinterpret historical events and extract meaningful patterns from complex scenarios such as the eruption of Mount Vesuvius. By constructing a synthetic dataset and applying machine-learning techniques—particularly the Random Forest model—we explored how demographic, social, behavioral, and geographical factors may have influenced survival during the catastrophe.

 One of the main insights from this study is that survival is rarely the result of a single cause. Instead, it reflects an interplay of multiple dimensions: distance from danger, reaction time, social class, wealth, and even access to shelter. These findings echo what archaeologists and historians already suspect about Pompeii—namely, that societal structures, urban layout, and individual behavior greatly shaped the outcome of the disaster. Through our model, we reproduced these dynamics in a simplified but analytically meaningful way.

 More broadly, this work illustrates how AI can enrich our understanding of ancient societies by providing new tools for simulation, prediction, and comparison. Although our dataset is fictional, the methodology is applicable to real archaeological data. Machine learning could help identify hidden patterns, test historical hypotheses, or compare how different civilizations responded to catastrophic events. Such approaches create bridges between technology and the humanities, showing that AI is not only a tool for modern problems but also a powerful means of exploring humanity’s past.

 Finally, this project highlights the importance of interdisciplinary research. Combining archaeology, history, data science, and machine learning opens the door to innovative perspectives on human resilience. While we cannot change what happened in Pompeii, we can learn from it—and these lessons may one day help us better prepare for future natural disasters.
