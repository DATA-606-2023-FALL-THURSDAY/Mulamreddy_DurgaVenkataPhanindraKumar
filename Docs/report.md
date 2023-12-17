# 0. Title and Author

* Fraud Detection in Financial transactions
* Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang
* Durga venkata Phanindra kumar Mulamreddy
* [Github] (https://github.com/Phanid221)
* [Linkedin] (https://www.linkedin.com/in/durga-venkata-phanindra-kumar-mulamreddy-19985114b/)
* [Presentation File] (Will update later)
* [Youtube video] (Will update the link later)
#

# 2 .Background

1. Financial fraud is a pervasive and costly issue in today's digital age. As the world increasingly relies on electronic payment systems and online transactions, the risk of fraudulent activities has grown significantly. Fraudulent activities can take various forms, including credit card fraud, identity theft, money laundering, and insider trading, among others. These activities not only result in significant financial losses for individuals and organizations but also erode trust and confidence in financial systems.
2. To address this critical issue, the field of data science and machine learning has been instrumental in developing sophisticated techniques for fraud detection. These techniques leverage the power of data to identify suspicious patterns and behaviors, helping financial institutions and businesses detect and prevent fraudulent transactions in real-time.
3. The primary objective of this project is to develop a state-of-the-art fraud detection system for financial transactions using data science and machine learning (Anamoly detection) techniques.
# 
4. **Some of the research questions are mentioned below**
 * What are the most common types of fraud in financial transactions, and how have they evolved over time?
 * The key challenges that are going to be faced during the development and implementation of the project.
 * Applying different machine learning techniques to compare and improve the proficiency of the model.
 * How does the insights derived from the past historical data helps the model in detecting a better accurate prediction?
 * Does the size or extent of data will pose any changes in the accuracy of a model for prediction and how can the scalability challenges be addressed?
 # 

# 3. Data

* Data was taken from the kaggle and the link for the data is provided below.
* The data is not a real data but a simulated data which was synthesized by the PaySim Simulator based on the real world mobile transactions.This simulator utilizes aggregated data from private sources to generate a synthetic dataset that closely mimics the typical patterns of real transactions. What sets it apart is its ability to inject instances of malicious behavior, simulating fraudulent activities. This synthetic dataset serves as a valuable resource for evaluating the effectiveness of various fraud detection methods.
* The Data is simulated based on a sample of real transactions extracted from one month of financial logs from a mobile money service implemented in an African country. The original logs were provided by an europen company which has their services ongoing in almost 14 countries throughout the world.
* [Link to the data] (https://www.kaggle.com/datasets/ealaxi/paysim1)
* The Data set is of pretty big size (493.5 MB) and it contains many columns which were all explained below for clear understanding.
* Data shape (6362620 of Rows and 11 Columns)
- Data Dictionary

| Column Name       | Data Type | Definition                                               | Potential Values                  |
|-------------------|-----------|---------------------------------------------------------|-----------------------------------|
| Step              | int       | Maps a unit of time in the real world. Here 1 step is 1 hour for a total of 31 days. | 1 - 743  (24*31=744)                         |
| Type              | object    | Mode of the payment. Cashin, CashOut, Debit, Payment and transfer | CashOut 35%, payment 34%, other 31% |
| Amount            | float     | Amount of the transaction (African Currency)                               | 0 - 92.4m                         |
| NameOrig          | object    | Customer who started the transaction                    | Almost all are unique values      |
| OldbalanceOrg     | float     | Initial balance before the transaction                 | 0 - 59.6m                        |
| NewbalanceOrig    | float     | Customer's balance after the transaction               | 0 - 49.6m                         |
| NameDest          | object    | Recipient ID of the transaction                         | Unique values specified to the transaction area |
| OldbalanceDest    | float     | Initial balance before the transaction                 | 0 - 356m                         |
| NewbalanceDest    | float     | Recipient's balance after the transaction               | 0 - 356m                         |
| IsFraud           | int       | Identifies a fraudulent transaction (1) and non-fraudulent (0) | 0 - 1                             |



* The target variable of the dataset is "isFraud" which will tell us whether the transaction is fraudulent or not depending on the value of 0 and 1.

  
# 4. Exploratory Data Analysis(EDA)

## 4.1 Data Cleaning.

### 4.1.1 Checking and removing Duplicates from the Data Set.
* Checking the duplicates from in the data set and we can see that there are no duplicates present in the data. So the data is free from duplicates.

## 4.2 Analysing the data and Visualizations.
* In this step we checked the columns in the data frame and derived some insights to modify the data.
* First, we will see the what are the types present in the Type(Column).
* This column contains the mode of payment as unique values as Cashin, CashOut, Debit, Payment and transfer
* The below is the distribution of these types throughout the data set.
   <img width="1000" alt="image" src="https://github.com/DATA-606-2023-FALL-THURSDAY/Mulamreddy_DurgaVenkataPhanindraKumar/blob/main/Data/visualize/Types_payment.png">
* The categories were spread all over the df with Cash_out occupying the most and followed by Payment, cash_in, and Transfer with Debit being the least occupied.
* Now we move onto the next column that is our target variable. the distribution and the pie chart was shown below.
   <img width="1000" alt="image" src="https://github.com/DATA-606-2023-FALL-THURSDAY/Mulamreddy_DurgaVenkataPhanindraKumar/blob/main/Data/visualize/image_2.png">
* From the graph, it can be seen that there were 6.3million trasactions are good whereas the 8213 trasactions reported fradulent behavior which has a 0.1 percentage.
* Now, we need to know which type of transactions are showing anamoly and which are not.
* We will see whether the fraud is happening in all categories or only in some particular categories. So, the below displayed plot shows the categories which have fraud and which have no fraud.
   <img width="600" alt="image" src="https://github.com/DATA-606-2023-FALL-THURSDAY/Mulamreddy_DurgaVenkataPhanindraKumar/blob/main/Data/visualize/image_11.png">  
* As the plot labels suggests, any label with zero means it doesn't have any fradulent behavior or transaction occured are secured.
* As we can see it, the catogories Cash_In, Debit and payment didn't have nay fradulent behavior and there is no need for these types in the dataframe. Whereas the Cash_out and Transfer have shown significant fradulent behavior. Then, we will drop the Cash_In, Debit and payment and we have with only cash_out and transfer categories in the type column (Which shows the Mode of Transaction).
* After we have the data set with the almost equal distribution over the type and these only represent the anamoly in the transactions and the rest which doesn't have any relation were dropped.
 
<img width="600" alt="image" src="https://github.com/DATA-606-2023-FALL-THURSDAY/Mulamreddy_DurgaVenkataPhanindraKumar/blob/main/Data/visualize/image_4.png">  
* With the above transformations our data frame was reduced to less that 50% of the original data i.e., from >6mil to 2.7mil.
* Now we will see is there any relation between the oldbalanceorg, newbalanceorig with fraud and compare this with mode of the payment.
<img width="800" alt="image" src="https://github.com/DATA-606-2023-FALL-THURSDAY/Mulamreddy_DurgaVenkataPhanindraKumar/blob/main/Data/visualize/image_8.png"> 

<img width="800" alt="image" src="https://github.com/DATA-606-2023-FALL-THURSDAY/Mulamreddy_DurgaVenkataPhanindraKumar/blob/main/Data/visualize/image_9.png"> 
* The distribution of the overall transaction amount is right-skewed, ranging between 0-10 million.
* The distribution of fraudulent transaction amounts is also right-skewed, ranging between 0-10 million, with a spike at 10 million. This suggests that all fraudulent transactions do not exceed 10 million per transfer. This might be due to reaching the maximum transfer limit of the bank.
<img width="800" alt="image" src="https://github.com/DATA-606-2023-FALL-THURSDAY/Mulamreddy_DurgaVenkataPhanindraKumar/blob/main/Data/visualize/image_!2.png"> 
* There's a notable correlation between 'amount' and 'oldbalanceOrg' that distinguishes between all transactions and fraudulent transactions. This is typical for fraud because criminals are likely intending to empty the target's account regardless of the savings balance.

## 4.3 Data Preprocessing

* Before diving into model training, it's essential to preprocess the data to ensure it's in the right format for our machine learning algorithms.

  ### In this Section
* We drop the unused columns.
* Convert data types for isFraud and isFlaggedFraud to boolean type.
* Encode the categorical columns using OneHotEncoder.
* Further divide the data into training and testing sets. Stratified sampling is used to ensure that both sets have a similar distribution of the target variable.

### OneHotEncoder
* If we Label Encoding categorical variable (in this case 'type') as integer e.g. (0,1,2,..) algorithms might assume an ordinal relationhsip between the categories. For example

* **PAYMENT = 0**
* **TRANSFER = 1**
* **CASH_OUT = 2**
* **DEBIT = 3**
* **CASH_IN = 4**
* This might lead the model to assume that DEBIT(3) somehow greater than TRANSFER(2), which doesn't make sense. OneHotEncoding avoids this problem.
* Instead of hard coding a single column, we will be defining a pipeline which will split the categorical features and numerical features based on their datatypes then OneHotEncode the categorical and scale the numerical data using standardscalar which were defined in a preprocessing step. 
### Pipeline
* A pipeline is a way to streamline a lot of the routine processes, making it easier to combine different steps of a machine learning workflow. A typical machine learning pipeline consists of three main stages:

#### Data Preprocessing:

* Data Cleaning: Handling missing values, outliers, and other inconsistencies in the data.
Feature Scaling: Standardizing or normalizing features to bring them to a similar scale.
Feature Engineering: Creating new features from existing ones or transforming features to improve model performance.
One-Hot Encoding or Label Encoding: Handling categorical variables.
#### Model Training:

* Choosing a Model: Selecting a machine learning algorithm based on the nature of the problem (classification, regression, etc.) and the characteristics of the data.

* Training the Model: Using training data to teach the model the patterns in the data.
#### Model Evaluation:

* Testing and Validation: Assessing the model's performance on data it hasn't seen before.

* Hyperparameter Tuning: Adjusting the hyperparameters of the model to improve its performance.
### Logistic Regression
<img width="600" alt="image" src="https://github.com/DATA-606-2023-FALL-THURSDAY/Mulamreddy_DurgaVenkataPhanindraKumar/blob/main/Data/visualize/image_13.png"> 
* Here are the results from the logistic regression classifier trained on the unbalanced dataset:

* Accuracy: 99.83%
* ROC AUC: 0.98
* Precision (for Fraudulent transactions): 0.92
* Recall (for Fraudulent transactions): 0.48
* F1-score (for Fraudulent transactions): 0.63
- Logistic Regression achieves high accuracy and ROC AUC, which means the model's overall reliability in classifying transactions is good. However, in fraud detection, where fraudulent transactions are intermingled with non-fraudulent ones, the balance between precision and recall often carries more weight than mere accuracy.

- The recall is 0.48, indicating that the model can detect 48% of fraudulent activities. This is not ideal. In fraud detection, a high recall is critical to ensure that fewer fraudulent transactions go undetected.

- The precision is 0.92, which means that for every 100 transactions the model predicts as fraudulent, 92 of them are genuinely fraudulent. A lower precision implies that a higher number of legitimate transactions are incorrectly flagged. We have a good value of precision but the recall is lower so need to have a balance between these two to get a good performance in the model. 

- The F1-score is reflecting the trade-off in fraud detection between capturing real fraudulent activities and minimizing false alarms. Since this is baseline model, a score of 0.6 indicates that there is room for improvement.
### Random Forest Classifier
<img width="600" alt="image" src="https://github.com/DATA-606-2023-FALL-THURSDAY/Mulamreddy_DurgaVenkataPhanindraKumar/blob/main/Data/visualize/image_14.png">
* Here are the results from the Random Forest classifier trained on the unbalanced dataset:

* Accuracy: 89.83%
* ROC AUC: 0.89
* Precision (for Fraudulent transactions): 0.98
* Recall (for Fraudulent transactions): 0.78
* F1-score (for Fraudulent transactions): 0.86
- Random Forest achieves good accuracy and ROC AUC, which means the model's overall reliability in classifying transactions is good. However, in fraud detection, where fraudulent transactions are intermingled with non-fraudulent ones, the balance between precision and recall often carries more weight than mere accuracy.

- The recall is 0.78, indicating that the model can detect 78% of fraudulent activities. This is good. In fraud detection, a high recall is critical to ensure that fewer fraudulent transactions go undetected.

- The precision is 0.98, which means that for every 100 transactions the model predicts as fraudulent, 98 of them are genuinely fraudulent. A lower precision implies that a higher number of legitimate transactions are incorrectly flagged. We have a good value of precision but the recall is lower so need to have a balance between these two to get a good performance in the model. 

- The F1-score is reflecting the trade-off in fraud detection between capturing real fraudulent activities and minimizing false alarms. Since this is baseline model, a score of 0.8 indicates that the model is performing good..
# Over Sampling and Under Sampling
### Class imbalance in Classification
* There are two ways to balance classes - by increasing the smaller class with random duplication, or by decreasing the larger class by randomly removing observations.

#### SMOTE | Overcoming Class Imbalance Problem Using SMOTE

* SMOTE is an oversampling technique where the synthetic samples are generated for the minority class. This algorithm helps to overcome the overfitting problem posed by random oversampling.

* Using SMOTE has made the Random Forest classifier more sensitive to the minority class (in this case 'Fraud' = 1), leading to a significant increase in recall for fraudulent transactions. This model is now better at catching most of the fraudulent activities. However, this comes at the cost of reduced precision. The trade-off between Precision and Recall has now become clearer.
#### K Nearest neighbour
<img width="600" alt="image" src="https://github.com/DATA-606-2023-FALL-THURSDAY/Mulamreddy_DurgaVenkataPhanindraKumar/blob/main/Data/visualize/image_15.png">

### Hyper Parameter Tuning
* Hyperparameter tuning is the process of finding the best set of hyperparameters for a machine learning model. Hyperparameters are configuration settings that are not learned from the data but are set prior to the training process. They can significantly impact the performance of a machine learning model.

* The process of hyperparameter tuning involves searching across a predefined hyperparameter space, training and evaluating the model with different combinations of hyperparameters, and selecting the set of hyperparameters that results in the best model performance.

* There are different techniques and we will be using GridSearch for both the models
### Grid Search:
* This involves exhaustively searching through a manually specified subset of the hyperparameter space. It evaluates the model performance for all possible combinations of hyperparameters within the specified range.
<img width="600" alt="image" src="https://github.com/DATA-606-2023-FALL-THURSDAY/Mulamreddy_DurgaVenkataPhanindraKumar/blob/main/Data/visualize/image_16.png">

# Results and Comparision

<img width="800" alt="image" src="https://github.com/DATA-606-2023-FALL-THURSDAY/Mulamreddy_DurgaVenkataPhanindraKumar/blob/main/Data/visualize/image_17.png">
<img width="800" alt="image" src="https://github.com/DATA-606-2023-FALL-THURSDAY/Mulamreddy_DurgaVenkataPhanindraKumar/blob/main/Data/visualize/image_18.png">
