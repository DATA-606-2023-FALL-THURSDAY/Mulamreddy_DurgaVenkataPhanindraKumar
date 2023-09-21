# 0. Title and Author

* Fraud Detection in Financial transactions
* Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang
* Durga venkata Phanindra kumar Mulamreddy
* [Github] (https://github.com/Phanid221)
* [Linkedin] (https://www.linkedin.com/in/durga-venkata-phanindra-kumar-mulamreddy-19985114b/)
* [Presentation File] (Will update later)
* [Youtube video] (Will update the link later)


# 1 .Background

1. A typical and useful use of data science in the fields of finance and cybersecurity is the "Fraud Detection in Financial Transactions" project. Build a machine learning model for this project with the goal of accurately identifying fraudulent transactions in a collection of financial transaction records.

2. There are many things in which the credability of transaction is quite questionable. For protecting financial institutions from fradualent transactions/ unauthorized transactions, identity theft which will lead to a substantial loss for the banks. It helps in preventing criminal activity. This fraud detection will boost the consumer confidence in utilizing online merchant platforms for their daily use and many more.

3. **Some of the research questions are mentioned below**
 * What are the most common types of fraud in financial transactions, and how have they evolved over time?
 * The key challenges that are going to be faced during the development and implementation of the project.
 * Applying different machine learning techniques to compare and improve the proficiency of the model.
 * How does the insights derived from the past historical data helps the model in detecting a better accurate prediction?
 * Does the size or extent of data will pose any changes in the accuracy of a model for prediction and how can the scalability challenges be addressed?
 

# 2. Data

* Data was taken from the kaggle and the link for the data is provided below.
* [Link to the data] (https://www.kaggle.com/datasets/ealaxi/paysim1)
* Data Size (493.5 MB)
* Data shape (6362620 of Rows and 11 Columns)
- Data Dictionary

| Column Name       | Data Type | Definition                                               | Potential Values                  |
|-------------------|-----------|---------------------------------------------------------|-----------------------------------|
| Step              | int       | Maps a unit of time in the real world. Here 1 step is 1 hour | 1 - 743                           |
| Type              | object    | Mode of the payment. Cashin, CashOut, Debit, Payment and transfer | CashOut 35%, payment 34%, other 31% |
| Amount            | float     | Amount of the transaction                               | 0 - 92.4m                         |
| NameOrig          | object    | Customer who started the transaction                    | Almost all are unique values      |
| OldbalanceOrg     | float     | Initial balance before the transaction                 | 0 - 59.6m                        |
| NewbalanceOrig    | float     | Customer's balance after the transaction               | 0 - 49.6                         |
| NameDest          | object    | Recipient ID of the transaction                         | Unique values specified to the transaction area |
| OldbalanceDest    | float     | Initial balance before the transaction                 | 0 - 356m                         |
| NewbalanceDest    | float     | Recipient's balance after the transaction               | 0 - 356m                         |
| IsFraud           | int       | Identifies a fraudulent transaction (1) and non-fraudulent (0) | 0 - 1                             |



* The target variable of the dataset is "isFraud" which will tell us whether the transaction is fraudulent or not depending on the value of 0 and 1.
* As of now, I am in a thought of considering all the columns but will make changes in the future depending on the EDA and further analysis. 
