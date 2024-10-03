import streamlit as st
import pandas as pd
import time 
from datetime import datetime
import os

# Set the title of the app
st.title("Attendance Tracker")

ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")

from streamlit_autorefresh import st_autorefresh
# Check if the CSV file exists before trying to read it
csv_file_path = f"Attendance/Attendance_{date}.csv"
if os.path.isfile(csv_file_path):
    df = pd.read_csv(csv_file_path)
    st.dataframe(df.style.highlight_max(axis=0))
else:
    st.write(f"No attendance data available for {date}.")