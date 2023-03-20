# Smoke-Detection

Smoke detection prediction with dataset from https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset with classification model in R Programming language & Used for Balikpapan Hackathon 2023.

# Dataset consists of variables:
1. UTC: Time when experiment was performed
2. Temperature[C]: Temperature of surroundings, measured in celcius
3. Humidity[%]: Air humidity during the experiment
4. TVOC[ppb]: Total Volatile Organic Compounds, measured in ppb (parts per billion)
5. eCO2[ppm]: CO2 equivalent concentration, measured in ppm (parts per million)
6. Raw H2: The amount of Raw Hydrogen [Raw Molecular Hydrogen; not compensated (Bias, Temperature etc.)] present in surroundings
7. Raw Ethanol: The amount of Raw Ethanol present in surroundings
8. Pressure[hPa]: Air pressure, Measured in hPa
9. PM1.0: Paticulate matter of diameter less than 1.0 micrometer
10. PM2.5: Paticulate matter of diameter less than 2.5 micrometer
11. NC0.5: Concentration of particulate matter of diameter less than 0.5 micrometer
12. NC1.0: Concentration of particulate matter of diameter less than 1.0 micrometer
13. NC2.5: Concentration of particulate matter of diameter less than 2.5 micrometer
14. CNT: Sample Count. Fire Alarm(Reality) If fire was present then value is 1 else it is 0
15. Fire Alarm: 1 means Positive and 0 means Not Positive

![image](https://user-images.githubusercontent.com/31152913/226340039-07eb38eb-8d45-4c44-b3f4-835e63d8121c.png)

**Insight**
1. There is not any high correlation between target feature and other features. Small positive correlation between target **feature** and **Humidity**, **Pressure**. Small negative correlation between target feature and *TVOC*, *Raw Ethanol*.
2. High positive correlation between eCO2 and TVOC, PM1.0. Pressure and Humidity. Raw H2 and Raw Ethanol. PM1.0 and eCO2, PM2.5.


![image](https://user-images.githubusercontent.com/31152913/226340162-b34cd8d0-1c4b-45dd-8aba-033f863dadb7.png)

based on running Random Forest is highest accuracy (99.99%), following by BAG (Bagged CART) (99.98%) and C5.0 (99.93%)
