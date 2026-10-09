# NovaPay Fraud Detection

This project cleans and explores NovaPay cross-border transaction data to prepare it for a supervised fraud detection model. The dataset contains 11,400 transactions across 26 columns, and the target column is `is_fraud`, where 0 means legitimate and 1 means fraud. The work was done in Python using pandas, NumPy, SciPy, Matplotlib and seaborn in a Jupyter notebook.

## Setup

To run the project, create a virtual environment with `python -m venv fraud-venv`, activate it with `fraud-venv\Scripts\activate`, and install the dependencies with `pip install -r requirements.txt`. Place `nova_pay_combined.csv` in the `data/` folder, open `notebook/EDA.ipynb`, select the `fraud-venv` kernel and click Run All.

## 1. Data Cleaning

The `timestamp` column was stored as text, so it was converted to a proper datetime. During this conversion, 32 values turned out to be impossible dates, such as month 13, hour 25 or a `0000-00-00` placeholder, so they were set to missing. The `amount_src` column was also stored as text because some values contained comma separators, such as "9,998.85". The commas were removed and the column was converted to numbers.

Next, 200 fully duplicated rows were found and removed, which reduced the dataset from 11,400 to 11,200 rows. The category columns were then checked for inconsistencies. They contained extra spaces such as " US", mixed casing such as "STANDARD" and "WEB", misspellings such as "standrd", "enhancd", "mobille" and "weeb", and placeholder text such as "unknown" and "nan" in place of real blank values. All values were standardised so each category has one consistent spelling, and the placeholder text was converted into proper missing values.

Finally, the numeric columns were checked for values that cannot exist in real transactions. These included negative amounts as low as -9,997.16, negative fees of -1.0, a placeholder fee of 9,999.99, IP risk scores above 1, negative device trust scores and negative transaction counts. These were treated as data errors and set to missing. To confirm the 9,999.99 fee was a placeholder, the 103 transactions with this value were investigated. They were spread evenly across all channels and currencies, and only 3.9% were fraud, which is below the overall rate of 8.9%. This confirmed it was a system placeholder rather than a real fee or a fraud signal.

## 2. Missing Values

After cleaning, every column had less than 5% missing data, so no columns were dropped. Each column was handled according to its meaning. The 60 rows with no timestamp were dropped, because a date cannot be reliably estimated.

For `amount_usd`, the missing values were recovered rather than guessed. The conversion rate from each source currency to USD was calculated from the rows that had both amounts, giving rates of 1.00 for USD, 1.25 for GBP and 0.74 for CAD, which match real exchange rates. Missing `amount_usd` values were then filled by multiplying `amount_src` by its currency's rate, and the reverse calculation was used to fill missing `amount_src` values.

Because fees differ by channel, missing `fee` values were filled with the median fee of each channel. Missing `kyc_tier` values were filled with the most common tier in each home country, since KYC rules vary by country, and any remaining blanks were labelled "unknown". The `channel` and `home_country` columns were filled with their most common value, and the remaining numeric columns were filled with the median, because the median is not affected by outliers. The `ip_address` column was left as it was, because it is only an identifier and will not be used in modelling.

## 3. Outliers

Outliers were detected using the IQR method, which flags values more than 1.5 times the interquartile range below the first quartile or above the third quartile. About 9.5% of the values in amount, fee and IP risk score were flagged. When these outliers were compared with normal rows, they showed a much higher fraud rate. Outliers in amount and fee were about 44% to 45% fraud compared with about 5% in normal rows, and outliers in IP risk score were about 70% fraud compared with about 3%. Because they carry such a strong fraud signal, the outliers were kept. Removing them would have discarded almost half of all fraud cases.

Every outlier in `risk_score_internal` was fraud, which suggests this score may have been calculated after the fraud outcome was already known. This is a sign of data leakage, so the column will be excluded from modelling. A 10% trimmed mean also showed that a typical transaction is about $198, while the raw mean of $443 is inflated by extreme values.

## 4. Exploratory Data Analysis

Only 8.9% of transactions are fraudulent, so the dataset is imbalanced. A model that predicted "not fraud" for every transaction would be 91% accurate but would not catch any fraud, so recall, precision and F1 will be used to evaluate models instead of accuracy.

The analysis showed clear differences between fraud and legitimate transactions. Transaction velocity had the strongest link to fraud, as fraudsters tend to make many transactions in a short time. Customers with a low KYC tier had a fraud rate of about 52%. Transactions from new devices were about 38% fraud compared with about 5% from known devices, and transactions with a location mismatch were about 36% fraud compared with about 3.5%. Web transactions were about 16% fraud compared with about 5% on mobile. Transfers to NGN and MXN had the highest fraud rate at about 19%, compared with 3% to 5% for USD, CAD and GBP. Fraud also peaked between 03:00 and 08:00 UTC, which is late at night in the US. Fraudulent accounts were usually less than 50 days old, and fraudulent transactions were much larger, with a median of about $600 compared with about $160 for legitimate ones.

Overall, a typical fraudulent transaction comes from a new account with a low KYC tier, uses a new or untrusted device from a risky and mismatched IP address, involves rapid repeated transactions through the web channel late at night in the US, and sends a large amount to NGN or MXN.

## Next Steps

The next stage is to build a supervised classification model. This will use a stratified train and test split to keep the fraud rate the same in both sets, address the class imbalance with class weights or resampling, and evaluate the model using recall, precision, F1 and PR-AUC. The `risk_score_internal` column will be excluded to prevent data leakage.

## Changelog

On 7 October 2026, the missing value handling was improved. The simple median fill for `amount_usd` was replaced with a calculation based on currency conversion rates, the median fill for `fee` was replaced with the median per channel, and the "unknown" fill for `kyc_tier` was replaced with the most common tier per home country, with "unknown" kept as a fallback. After re-running the analysis, the findings stayed consistent, because only 2% to 4% of rows were affected. This shows the results are robust.



## 8 October 2026: Model Optimisation

For this task, I built a baseline model and then applied feature engineering, hyperparameter tuning and threshold tuning to see how much each one improved fraud detection. I used a Random Forest classifier with a stratified 80/20 train and test split, and I removed ID columns and risk_score_internal to avoid data leakage. Because the data is imbalanced, I measured precision, recall, F1 and PR-AUC instead of accuracy.

The baseline model reached a precision of 1.0, a recall of 0.819, an F1 score of 0.901 and a PR-AUC of 0.869. This means every transaction it flagged as fraud was correct, but it missed about 18% of fraud cases.

For feature engineering, I created new features based on my EDA findings. These included a night-time flag for transactions between 3am and 8am UTC, a new account flag for accounts under 50 days old, a risky corridor flag for transfers to NGN and MXN, the fee as a percentage of the amount, a velocity ratio comparing the last hour with the last day, how a transaction's amount compares with the customer's usual amount, and how many customers or transactions share the same device or IP address. This improved PR-AUC slightly from 0.869 to 0.873, while the other scores stayed the same. The gain was small because Random Forest can already learn most of these patterns from the raw columns by itself.

For hyperparameter tuning, I used RandomizedSearchCV with 5-fold stratified cross-validation, testing 20 combinations of settings and choosing the best by PR-AUC. The best settings were 300 trees, a maximum depth of 10, a minimum of 10 samples to split, a minimum of 2 samples per leaf, log2 max features and balanced class weights. This gave the best PR-AUC of 0.880, meaning it was the best at ranking fraud above legitimate transactions. Precision dropped slightly to 0.948 because the balanced class weights made the model bolder, and the shallower trees helped reduce overfitting.

I also tested threshold tuning by lowering the decision threshold from 0.5 down to 0.1. Lowering it only increased recall slightly, from 0.819 to 0.894, while precision dropped sharply from 0.948 to 0.276. This showed that 0.5 gives the best balance.

The main finding is that recall stayed at about 82% across every strategy. This suggests that about 18% of fraud cases look almost the same as normal transactions with the features currently available. Further tuning is unlikely to fix this, so adding new data sources, such as device fingerprints or longer customer history, would likely improve fraud detection more than further model optimisation.