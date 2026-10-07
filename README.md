# novapay-fraud-detection

In this project, I cleaned the NovaPay transaction data to get it ready for fraud detection. The data started with 11,400 rows and 26 columns. First, I checked the datatypes and found that timestamp and amount_src were saved as text. I changed timestamp to a proper date, and 32 dates were impossible (like month 13 or hour 25), so I set them as missing. amount_src had commas in some numbers like "9,998.85", so I removed the commas and turned it into numbers. Next, I found 200 duplicate rows and removed them, which left 11,200 rows. Then I checked the text columns for mistakes and found extra spaces, mixed capital letters, spelling errors like "standrd", "enhancd", "mobille" and "weeb", and fake empty values written as "unknown" or "nan". I fixed all of them so each category is written one way. Lastly, I checked for outliers and found values that can't be real, like negative amounts, negative fees, a fake fee of 9999.99, risk scores above 1, and negative transaction counts. I set these as missing because they were errors. I kept the very large amounts because in fraud data, extreme values are often the fraud itself. I also noticed only 8.9% of transactions are fraud, so the data is imbalanced, and I'll need to handle that when building the model. My next step is to deal with the missing values.




# NEW TASK

In this project I explored and cleaned NovaPay transaction data to find patterns that separate fraud from normal transactions.

Missing values:
After cleaning, every column had less than 5% missing data, so I didn't drop any whole columns. I dropped the 60 rows with no timestamp because you can't make up a date. For number columns like fee, amount and risk scores, I filled the blanks with the median, because the median isn't pulled by outliers. For channel and home country, I used the most common value. For KYC tier and IP country, I filled the blanks with "unknown" so I wouldn't hide them. I left IP address as it was, because it's just a label and won't be used in a model.

Outliers:
Using the IQR method, I found that about 9.5% of the values in amount, fee and IP risk score were outliers. When I checked them, 44% to 70% of these outliers were fraud, compared with only 3% to 5% in normal rows. So I kept them, because they carry the fraud signal. All the outliers in risk_score_internal were fraud, which makes me think that column may have been made after the fraud was known (data leakage). I also used a trimmed mean, which showed a typical transaction is about $198, while the normal mean of $443 was pushed up by the outliers.

EDA findings:
Only 8.9% of transactions are fraud, so the data is imbalanced. Fraud transactions had bigger amounts, higher IP risk scores, lower device trust scores and much newer accounts. Low KYC customers had about 52% fraud. Web transactions, new devices and location mismatches all had much higher fraud rates. Sending money to NGN and MXN had the most fraud (about 19%). Fraud also jumped between 3am and 8am UTC, which is late night in the US. The strongest link to fraud was transaction velocity: fraudsters make many transactions in a short time.



# 7 October 2026 – Smarter Missing Value Filling

In this task, I improved how I filled the missing values for amount_usd, fee and kyc_tier. Before, I used one simple median or "unknown" for everything, but now I used methods that make more sense for each column.

amount_usd:
Instead of guessing, I worked out each currency's conversion rate to USD using the rows that had both amount_src and amount_usd. I divided amount_usd by amount_src and took the median rate for each source currency. The rates came out as CAD = 0.74, GBP = 1.25 and USD = 1.00, which match real exchange rates. Then I filled the missing amount_usd values by multiplying amount_src by its currency's rate. I also did the reverse to fill missing amount_src values.

fee:
Fees depend on the channel, because ATM, web and mobile charge differently. So I grouped the data by channel and filled each missing fee with the median fee of its own channel.

kyc_tier:
I grouped the data by home country and filled each missing KYC tier with the most common tier in that country, because KYC rules can be different in each country. Anything still missing after that was filled with "unknown".

Result:
After re-running everything, the EDA charts and fraud patterns stayed almost the same. This is because only 2–4% of rows were filled differently, and it shows my findings are strong and don't depend on how the blanks were filled.