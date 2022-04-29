# Work Logs

 
## Work Log 1 (27/04/2022)

- Loaded the excel files
- Merged them into a single DF
- Split the information for every cryptocurrency into it's own dataframe (10 in total, df_ADA, df_ATOM...) 
- Each DF contains the specific market_metrics for each crypto ('ADJCLOSE', 'CLOSE', 'HIGH', 'LOW', 'OPEN', 'VOLUME')

## Work Log 2 (28/04/2022)
- Decision of extract the DF for each currency and then we will do the Data Preparation and Modelling steps for each one. With this decision, we will work on 10 notebooks.
- We deleted the rows with Nan values. After that we check duplicates, we changed date to datime.
- We create new features, pctdif_closeopen and pctdif_highlow to see the differences between close and open values and high and low values.

## Work Log 3 (29/04/2022)
- Regarding the Notebook of the AVAX currency, we droppped the 2 first rows of the Dataset, because those 2 rows correspond to the dates 2020-07-13 and 2020-07-13 and in the third row it had a "jump" to the date 2020-09-2020. So we will only maintain the dates that start in 22-09-2020.
- When we were analyzing the graph about the Volume of the LINK currency, we checked if the high value that appeared in the graph was true or not because we thought it could be an outlier. So searched on the site https://nomics.com/assets/link-chainlink and we checked that it was a true value. But we decided to drop the row with the index of 1205 and volume of 1.705493e11 since it may affect the performance of the models.


## Contributors:

- Rodrigo Guedes
- Diogo Pires 
- Catarina Garcez
- Beatriz Selidónio

