
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
print("hello world")


heart_disease_df = pd.read_csv("heart_disease.csv")

#--------------------------------------------------------------------------DATA CLEANING-----------------------------------------------------------------------------
#Sample test
#1. Check for NaN values for value counts
#2. Replace NaN with average data (Mean, Median, or Mode depending on the circumstances)
#3. Check to see if Value count lines up

print(heart_disease_df["Alcohol Consumption"].value_counts(dropna=False))
#print(heart_disease_df["Alcohol Consumption"].unique())
print(heart_disease_df["Alcohol Consumption"].mode()[0])

alcohol_mode = heart_disease_df["Alcohol Consumption"].mode()[0]

heart_disease_df["Alcohol Consumption"] = heart_disease_df["Alcohol Consumption"].fillna("Medium")

print(heart_disease_df["Alcohol Consumption"].value_counts(dropna=False))
heart_disease_df.info()

#Check overall Table now for missing values


# GENDER - categorical → mode

heart_disease_df["Gender"] = heart_disease_df["Gender"].fillna(
    heart_disease_df["Gender"].mode()[0]
)


# AGE - numerical -> median

print(heart_disease_df["Age"].value_counts(dropna=False))

heart_disease_df["Age"] = heart_disease_df["Age"].fillna(
    heart_disease_df["Age"].median()
)

print(heart_disease_df["Age"].unique())
print(heart_disease_df["Age"].value_counts(dropna=False))
print(heart_disease_df["Age"].unique())


# GENDER - categorical → mode

print(heart_disease_df["Gender"].value_counts(dropna=False))

heart_disease_df["Gender"] = heart_disease_df["Gender"].fillna(
    heart_disease_df["Gender"].mode()[0]
)

print(heart_disease_df["Gender"].value_counts(dropna=False))
print(heart_disease_df["Gender"].unique())


# BLOOD PRESSURE - numerical → median

print(heart_disease_df["Blood Pressure"].value_counts(dropna=False))

print("Median Blood Pressure:", heart_disease_df["Blood Pressure"].median())

heart_disease_df["Blood Pressure"] = heart_disease_df["Blood Pressure"].fillna(
    heart_disease_df["Blood Pressure"].median()
)

print(heart_disease_df["Blood Pressure"].value_counts(dropna=False))


# Cholesterol level - numerical → median

print(heart_disease_df["Cholesterol Level"].value_counts(dropna=False))

print("Median Cholesterol Level:", heart_disease_df["Cholesterol Level"].median())

heart_disease_df["Cholesterol Level"] = heart_disease_df["Cholesterol Level"].fillna(
    heart_disease_df["Cholesterol Level"].median()
)

print(heart_disease_df["Cholesterol Level"].value_counts(dropna=False))


# Exercise Habits - Categorical → Mode

print(heart_disease_df["Exercise Habits"].value_counts(dropna=False))

print("Mode Exercise Habits:", heart_disease_df["Exercise Habits"].mode()[0])

heart_disease_df["Exercise Habits"] = heart_disease_df["Exercise Habits"].fillna(
    heart_disease_df["Exercise Habits"].mode()[0]
)

print(heart_disease_df["Exercise Habits"].value_counts(dropna=False))


# Smoking - Categorical → Mode

print(heart_disease_df["Smoking"].value_counts(dropna=False))

print("Mode Smoking:", heart_disease_df["Smoking"].mode()[0])

heart_disease_df["Smoking"] = heart_disease_df["Smoking"].fillna(
    heart_disease_df["Smoking"].mode()[0]
)

print(heart_disease_df["Smoking"].value_counts(dropna=False))


# Family Heart Disease - Categorical → Mode

print(heart_disease_df["Family Heart Disease"].value_counts(dropna=False))

print("Mode Family Heart Disease:", heart_disease_df["Family Heart Disease"].mode()[0])

heart_disease_df["Family Heart Disease"] = heart_disease_df["Family Heart Disease"].fillna(
    heart_disease_df["Family Heart Disease"].mode()[0]
)

print(heart_disease_df["Family Heart Disease"].value_counts(dropna=False))


# Diabetes - Categorical → Mode

print(heart_disease_df["Diabetes"].value_counts(dropna=False))

print("Mode Diabetes:", heart_disease_df["Diabetes"].mode()[0])

heart_disease_df["Diabetes"] = heart_disease_df["Diabetes"].fillna(
    heart_disease_df["Diabetes"].mode()[0]
)

print(heart_disease_df["Diabetes"].value_counts(dropna=False))


# BMI - Numeric → Median

print(heart_disease_df["BMI"].value_counts(dropna=False))

print("Median BMI:", heart_disease_df["BMI"].median())

heart_disease_df["BMI"] = heart_disease_df["BMI"].fillna(
    heart_disease_df["BMI"].median()
)

print(heart_disease_df["BMI"].value_counts(dropna=False))


# High Blood Pressure - Categorical → Mode

print(heart_disease_df["High Blood Pressure"].value_counts(dropna=False))

print("Mode High Blood Pressure:", heart_disease_df["High Blood Pressure"].mode()[0])

heart_disease_df["High Blood Pressure"] = heart_disease_df["High Blood Pressure"].fillna(
    heart_disease_df["High Blood Pressure"].mode()[0]
)

print(heart_disease_df["High Blood Pressure"].value_counts(dropna=False))


# Low HDL Cholesterol - Categorical

print(heart_disease_df["Low HDL Cholesterol"].value_counts(dropna=False))

heart_disease_df["Low HDL Cholesterol"] = heart_disease_df["Low HDL Cholesterol"].fillna("Yes")

print(heart_disease_df["Low HDL Cholesterol"].value_counts(dropna=False))


# High LDL Cholesterol - Categorical

print(heart_disease_df["High LDL Cholesterol"].value_counts(dropna=False))

heart_disease_df["High LDL Cholesterol"] = heart_disease_df["High LDL Cholesterol"].fillna("No")

print(heart_disease_df["High LDL Cholesterol"].value_counts(dropna=False))


#Note -- The cholesterol data sets may introduce possible sets of bias
#True for any type of Yes and No data


# Stress Level

print(heart_disease_df["Stress Level"].value_counts(dropna=False))

heart_disease_df["Stress Level"] = heart_disease_df["Stress Level"].fillna("Medium")

print(heart_disease_df["Stress Level"].value_counts(dropna=False))


# Sleep Hours *

print(heart_disease_df["Sleep Hours"].value_counts(dropna=False))

heart_disease_df = heart_disease_df.dropna(subset=["Sleep Hours"])

#print(heart_disease_df["Sleep Hours"].unique())

print(heart_disease_df.info())


# Sugar Consumption

print(heart_disease_df["Sugar Consumption"].value_counts(dropna=False))

heart_disease_df["Sugar Consumption"] = heart_disease_df["Sugar Consumption"].fillna("Low")

print(heart_disease_df.info())


# Tri Level

print(heart_disease_df["Triglyceride Level"].value_counts(dropna=False))

heart_disease_df["Triglyceride Level"] = heart_disease_df["Triglyceride Level"].fillna(
    heart_disease_df["Triglyceride Level"].mean()
)

print(heart_disease_df["Triglyceride Level"].value_counts(dropna=False))

mean_data = heart_disease_df["Triglyceride Level"].mean()

print(f"TRI MEAN------{mean_data}")

print(heart_disease_df.info())


# Fasting Blood Sugar

print(heart_disease_df["Fasting Blood Sugar"].value_counts(dropna=False))

heart_disease_df["Fasting Blood Sugar"] = heart_disease_df["Fasting Blood Sugar"].fillna(
    heart_disease_df["Fasting Blood Sugar"].median()
)

print(heart_disease_df.info())


# CRP Level *

print(heart_disease_df["CRP Level"].value_counts(dropna=False))

heart_disease_df = heart_disease_df.dropna(subset=["CRP Level"])

print(heart_disease_df["CRP Level"].unique)
print("---------------------------------------------------------CRP")
print(heart_disease_df.info())

heart_disease_df["Homocysteine Level"] = heart_disease_df["Homocysteine Level"].fillna(
    heart_disease_df["Homocysteine Level"].mean()
)

# Save cleaned dataset as CSV

heart_disease_df.to_csv("heart_disease_cleaned.csv", index=False)

print("Data cleaning completed!")
print("File: heart_disease_cleaned.csv")

#-------------------------------Data Modelling/splitting-------------------------
x=heart_disease_df.drop("Heart Disease Status", axis=1)
#Feature related data
y=heart_disease_df["Heart Disease Status"]
#Label related data
#First stage in the supervised learning aspect is splitting data
print(x.head())
print(y.value_counts())
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y, 
    test_size=0.2)
x_train.shape, x_test.shape, y_train.shape, y_test.shape