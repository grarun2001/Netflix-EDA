# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. LOAD DATASET

df = pd.read_csv("netflix.csv")

print("=" * 50)
print("FIRST 5 ROWS")
print("=" * 50)
print(df.head())

print("\nLAST 5 ROWS")
print(df.tail())

print("\nDataset Shape:")
print(df.shape)

# 3. CLEAN THE DATASET

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove Duplicate Rows
df.drop_duplicates(inplace=True)

# Fill Missing Values
df["director"] = df["director"].fillna("Unknown")
df["cast"] = df["cast"].fillna("Unknown")
df["country"] = df["country"].fillna("Unknown")
df["rating"] = df["rating"].fillna(df["rating"].mode()[0])
df["duration"] = df["duration"].fillna("Unknown")

# Convert date_added to datetime
df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")

print("\nData Cleaning Completed")

# 4. EXPLORE THE DATASET

print("\nDataset Information")
print("-" * 30)
df.info()

print("\nData Types")
print(df.dtypes)

print("\nColumn Names")
print(df.columns)

print("\nMovies and TV Shows")
print(df["type"].value_counts())

print("\nRatings")
print(df["rating"].value_counts())

print("\nTop 10 Countries")
print(df["country"].value_counts().head(10))

print("\nTop 10 Directors")
print(df["director"].value_counts().head(10))

print("\nTop 10 Genres")
print(df["listed_in"].value_counts().head(10))

print("\nMovies Released After 2020")
print(df[df["release_year"] > 2020][["title", "release_year"]].head())

print("\nGroup By Type")
print(df.groupby("type").size())

print("\nLatest Releases")
print(df.sort_values(by="release_year", ascending=False).head())

# 5. SUMMARIZE THE DATASET

print("\nNumerical Summary")
print(df.describe())

print("\nCategorical Summary")
print(df.describe(include="object"))

years = df["release_year"].to_numpy()

print("\nNumPy Summary")

print("Mean :", np.mean(years))
print("Median :", np.median(years))
print("Minimum :", np.min(years))
print("Maximum :", np.max(years))
print("Standard Deviation :", np.std(years))
print("Variance :", np.var(years))

# 6. EDA VISUALIZATIONS

# Count Plot
plt.figure(figsize=(6,5))
sns.countplot(data=df, x="type")
plt.title("Movies vs TV Shows")
plt.xlabel("Type")
plt.ylabel("Count")
plt.show()

# Histogram
plt.figure(figsize=(8,5))
plt.hist(df["release_year"], bins=25)
plt.title("Release Year Distribution")
plt.xlabel("Release Year")
plt.ylabel("Frequency")
plt.show()

#Bar Chart
plt.figure(figsize=(10,5))
df["country"].value_counts().head(10).plot(kind="bar")
plt.title("Top 10 Countries")
plt.xlabel("Country")
plt.ylabel("Number of Shows")
plt.xticks(rotation=45)
plt.show()

# ===============DIRECTOR & DURATION ANALYSIS==================

print("\n" + "="*60)
print("DIRECTOR & DURATION ANALYSIS")
print("="*60)

# Create a copy of movie data
movies = df[df["type"] == "Movie"].copy()

# Convert duration from text to numeric
movies["duration"] = movies["duration"].str.replace(" min", "", regex=False)
movies["duration"] = pd.to_numeric(movies["duration"], errors="coerce")

# 1. Top Directors by Number of Movies

print("\nTop 10 Directors by Number of Movies")
top_directors = movies["director"].value_counts().head(10)
print(top_directors)

plt.figure(figsize=(10,5))
top_directors.plot(kind="bar")
plt.title("Top 10 Directors by Number of Movies")
plt.xlabel("Director")
plt.ylabel("Number of Movies")
plt.xticks(rotation=45)
plt.show()

# 2. Directors with Highest Average Movie Duration

print("\nTop Directors by Average Movie Duration")

director_avg = (
    movies.groupby("director")["duration"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

print(director_avg)

plt.figure(figsize=(10,5))
director_avg.plot(kind="bar")
plt.title("Top Directors by Average Movie Duration")
plt.xlabel("Director")
plt.ylabel("Average Duration (Minutes)")
plt.xticks(rotation=45)
plt.show()

# 3. Directors with Highest Maximum Movie Duration

print("\nDirectors with Highest Maximum Movie Duration")

director_max = (
    movies.groupby("director")["duration"]
    .max()
    .sort_values(ascending=False)
    .head(10)
)

print(director_max)

plt.figure(figsize=(10,5))
director_max.plot(kind="bar")
plt.title("Directors with Highest Maximum Movie Duration")
plt.xlabel("Director")
plt.ylabel("Maximum Duration (Minutes)")
plt.xticks(rotation=45)
plt.show()

# 4. Top 10 Longest Movies

print("\nTop 10 Longest Movies")

longest_movies = movies.sort_values(
    by="duration",
    ascending=False
)[["title", "director", "duration"]].head(10)

print(longest_movies)

# 5. Directors with Most Movies Longer Than 120 Minutes

print("\nDirectors with Most Movies Longer Than 120 Minutes")

long_movies = movies[movies["duration"] > 120]

director_long = long_movies["director"].value_counts().head(10)

print(director_long)

plt.figure(figsize=(10,5))
director_long.plot(kind="bar")
plt.title("Directors with Most Movies Longer Than 120 Minutes")
plt.xlabel("Director")
plt.ylabel("Number of Movies")
plt.xticks(rotation=45)
plt.show()



#==================TOP 5 COMPARISON ANALYSIS=====================


print("\n" + "=" * 60)
print("TOP 5 COMPARISON ANALYSIS")
print("=" * 60)

# 1. Top 10 Actors by Number of Titles

print("\n1. Top 10 Actors by Number of Titles")
top_actors = (
    df["cast"]
    .dropna()
    .str.split(", ")
    .explode()
    .value_counts()
    .head(10)
)

print(top_actors)

# 2. Rating vs Content Type

print("\n2. Rating vs Content Type")
rating_vs_type = pd.crosstab(df["rating"], df["type"])

print(rating_vs_type)

# 3. Number of Titles Added Each Year

print("\n3. Number of Titles Added Each Year")

df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")

titles_added = (
    df["date_added"]
    .dt.year
    .value_counts()
    .sort_index()
)

print(titles_added)

# 4. Average Movie Duration by Rating

print("\n4. Average Movie Duration by Rating")

movies = df[df["type"] == "Movie"].copy()

movies["duration"] = movies["duration"].str.replace(" min", "", regex=False)
movies["duration"] = pd.to_numeric(movies["duration"], errors="coerce")

avg_duration = (
    movies.groupby("rating")["duration"]
    .mean()
    .sort_values(ascending=False)
)

print(avg_duration)

# 5. Country vs Number of Movies

print("\n5. Top 10 Countries by Number of Movies")

country_movies = (
    movies["country"]
    .value_counts()
    .head(10)
)

print(country_movies)

print("\nAdditional Comparison Analysis Completed Successfully!")