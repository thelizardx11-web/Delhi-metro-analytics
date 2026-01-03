import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data load
df = pd.read_csv("delhi_metro.csv")

# Preview
print(df.shape)
print(df.head())
print(df.dtypes)

# Missing values check
print(df.isna().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

# Convert date/time columns
df["tap_in_time"] = pd.to_datetime(df["tap_in_time"])
df["tap_out_time"] = pd.to_datetime(df["tap_out_time"])

# Create helper columns
df["hour"] = df["tap_in_time"].dt.hour
df["day_type"] = df["tap_in_time"].dt.dayofweek.apply(lambda x: "Weekend" if x>=5 else "Weekday")

print(df.columns)

# EDA 
# 1. Trips by hour 
trips_by_hour = df.groupby("hour").size()
sns.lineplot(x=trips_by_hour.index, y=trips_by_hour.values)
plt.title("Trips by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Trips")
plt.tight_layout()
plt.show()

# 2.weekday vs weekend trips 
trips_day = df.groupby("day_type").size()
sns.barplot(x=trips_day.index, y=trips_day.values)
plt.title("Trips: Weekday vs Weekend")
plt.xlabel("Day Type")
plt.ylabel("Number of Trips")
plt.tight_layout()
plt.show()

# 3. Top 10 Origin stations
top_stations = df["origin_station"].value_counts().head(10)
sns.barplot(x=top_stations.index, y=top_stations.values)
plt.xticks(rotation=45)
plt.title("Top 10 Origin Stations")
plt.xlabel("Station")
plt.ylabel("Trip started")
plt.tight_layout()
plt.show()

# 4. Revenue by line 
revenue_line = df.groupby("origin_line")["fare"].sum()
sns.barplot(x=revenue_line.index, y=revenue_line.values)
plt.xticks(rotation=45)
plt.title("Revenue by line")
plt.xlabel("Metro line")
plt.ylabel("Total Fare Collected")
plt.tight_layout()
plt.show()

# ------------------ Insights ------------------

# Peak hours
peak_hours = trips_by_hour.sort_values(ascending=False).head(3)
print("Peak Hours (Top 3):")
print(peak_hours)

# Top revenue lines
top_revenue_lines = revenue_line.sort_values(ascending=False).head(3)
print("\nTop Revenue Lines:")
print(top_revenue_lines)

# Average trip duration (minutes)
df["trip_duration_min"] = (df["tap_out_time"] - df["tap_in_time"]).dt.total_seconds()/60
print("\nAverage Trip Duration (minutes):", df["trip_duration_min"].mean())
