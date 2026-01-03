# Delhi-metro-analytics
## Problem
- **Goal:** Delhi Metro trips ka demand, crowding, aur revenue understand karna.
- **Focus:** Peak hours, busy stations, line-wise revenue, average durations.

## Dataset
- **Source:** Internal/Practice dataset (Delhi Metro trips).
- **Key columns:** trip_id, origin_station, destination_station, origin_line, trip_date, tap_in_time, tap_out_time, fare, hour, day_type.

## Steps
1. **Data loading:** CSV read + preview (shape, dtypes).
2. **Cleaning:** Missing values check, duplicates remove, datetime convert, helper columns (hour, day_type).
3. **EDA:** Trips by hour, Weekday vs Weekend, Top 10 origin stations, Revenue by line.
4. **Insights:** Peak hours, crowded stations, lines with longer durations, revenue concentration.
5. **Conclusion:** Actionable business points.

## Key insights
- **Peak windows:** 8–10 AM, 5–8 PM me highest traffic.
- **Crowded stations:** Top 3 origin stations me crowd management ki zarurat.
- **Duration:** Kuch lines me average trip duration zyada — schedule optimize karein.
- **Revenue:** Revenue kuch selected lines se major share — waha fare strategy/focus beneficial.

## How to run
- **Install:** `pip install -r requirements.txt`
- **Run notebook:** Open `notebooks/delhi_metro_analysis.ipynb` and run all cells.
- **Outputs:** Charts `images/` folder me (hourly, weekday/weekend, top stations, revenue).

## Folder structure
