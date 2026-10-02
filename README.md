# novapay-fraud-detection

In this project, I cleaned the NovaPay transaction data to get it ready for fraud detection. The data started with 11,400 rows and 26 columns. First, I checked the datatypes and found that timestamp and amount_src were saved as text. I changed timestamp to a proper date, and 32 dates were impossible (like month 13 or hour 25), so I set them as missing. amount_src had commas in some numbers like "9,998.85", so I removed the commas and turned it into numbers. Next, I found 200 duplicate rows and removed them, which left 11,200 rows. Then I checked the text columns for mistakes and found extra spaces, mixed capital letters, spelling errors like "standrd", "enhancd", "mobille" and "weeb", and fake empty values written as "unknown" or "nan". I fixed all of them so each category is written one way. Lastly, I checked for outliers and found values that can't be real, like negative amounts, negative fees, a fake fee of 9999.99, risk scores above 1, and negative transaction counts. I set these as missing because they were errors. I kept the very large amounts because in fraud data, extreme values are often the fraud itself. I also noticed only 8.9% of transactions are fraud, so the data is imbalanced, and I'll need to handle that when building the model. My next step is to deal with the missing values.




# New Task

Using IQR, about 9.5% of values in amount, fee and ip_risk_score were outliers. Their fraud rate was 44–70%, compared with 3–5% in normal rows, so I kept them because they carry the fraud signal. risk_score_internal outliers were 100% fraud, which suggests possible data leakage. A trimmed mean (10%) showed the typical transaction is about $198, versus a raw mean of $443 inflated by outliers.

# NovaPay Fraud Detection

In this project I explored and cleaned NovaPay transaction data to find patterns that separate fraud from normal transactions.

Missing values:
After cleaning, every column had less than 5% missing data, so I didn't drop any whole columns. I dropped the 60 rows with no timestamp because you can't make up a date. For number columns like fee, amount and risk scores, I filled the blanks with the median, because the median isn't pulled by outliers. For channel and home country, I used the most common value. For KYC tier and IP country, I filled the blanks with "unknown" so I wouldn't hide them. I left IP address as it was, because it's just a label and won't be used in a model.

Outliers:
Using the IQR method, I found that about 9.5% of the values in amount, fee and IP risk score were outliers. When I checked them, 44% to 70% of these outliers were fraud, compared with only 3% to 5% in normal rows. So I kept them, because they carry the fraud signal. All the outliers in risk_score_internal were fraud, which makes me think that column may have been made after the fraud was known (data leakage). I also used a trimmed mean, which showed a typical transaction is about $198, while the normal mean of $443 was pushed up by the outliers.

EDA findings:
Only 8.9% of transactions are fraud, so the data is imbalanced. Fraud transactions had bigger amounts, higher IP risk scores, lower device trust scores and much newer accounts. Low KYC customers had about 52% fraud. Web transactions, new devices and location mismatches all had much higher fraud rates. Sending money to NGN and MXN had the most fraud (about 19%). Fraud also jumped between 3am and 8am UTC, which is late night in the US. The strongest link to fraud was transaction velocity: fraudsters make many transactions in a short time.