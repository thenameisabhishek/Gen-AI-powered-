import pandas as pd
import matplotlib.pyplot as plt
import sys # <-- ADD THIS LINE

# --- 1. LOAD DATA ---
try:
    df = pd.read_csv("sample_data.csv")
    print("INFO: Sample data loaded successfully!")
except FileNotFoundError:
    print("ERROR: sample_data.csv not found. Make sure it's in the same folder.")
    sys.exit() # This will now work correctly

# --- 2. CREATE PIE CHART for Overall Mood Distribution ---
print("INFO: Creating mood distribution pie chart...")
mood_counts = df['mood_bucket'].value_counts()
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#c2c2f0']

plt.figure(figsize=(8, 8))
plt.pie(mood_counts, labels=mood_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('Overall Mood Distribution', fontsize=16)
plt.axis('equal')

pie_chart_filename = 'mood_pie_chart.png'
plt.savefig(pie_chart_filename, dpi=300)
print(f"SUCCESS: Pie chart saved as '{pie_chart_filename}'")
plt.close()


# --- 3. CREATE BAR CHART for Daily Mood Trends ---
print("INFO: Creating daily mood trend bar chart...")
df['date'] = pd.to_datetime(df['date'])
df['day'] = df['date'].dt.strftime('%b %d')

daily_counts = df.groupby(['day', 'mood_bucket']).size().unstack(fill_value=0)

daily_counts.plot(kind='bar', stacked=True, figsize=(10, 6), color=colors)
plt.title('Daily Mood Trends', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Number of Check-ins')
plt.xticks(rotation=45)
plt.legend(title='Mood')
plt.tight_layout()

bar_chart_filename = 'mood_trend_chart.png'
plt.savefig(bar_chart_filename, dpi=300)
print(f"SUCCESS: Bar chart saved as '{bar_chart_filename}'")
plt.close()