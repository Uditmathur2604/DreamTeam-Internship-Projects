🌡️ Temperature Prediction Using Historical Weather Data

> Developed during my internship at **Dreamteam Technologies Pvt. Ltd.**

🚀 Project Overview

This project uses 10 years of hourly weather data (2014–2024, 87,000+ records) to build a linear regression model that forecasts short-term temperature trends based on environmental factors.

---

🧰 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook

---

🔍 Key Steps

1. **Load & Clean the Data**
   Converted numeric types, handled missing values, ensured consistency.

2. **Filter for September 10 AM**  
   Ensured model training on uniform conditions.

3. **Feature Engineering**  
   Created `TEMP_CHANGE` to represent the temperature shift each hour.

4. **Train the Model**  
   Applied Linear Regression using humidity, wind speed, and cloud cover.

5. **Make Predictions**  
   Predicted the next hour's temperature change.

6. **Forecast 12 Hours Ahead**  
   Simulated temperature for 12 future hourly steps under constant conditions.

7. **Visualize Results**  
   Displayed predicted trend via a line chart.

---

📁 Files

- `/data/_temperature_data.csv` – Raw weather dataset
- `/notebooks/Task-1_Temperature.ipynb` – Full code and output

---

📈 Output

A minimal, fast temperature forecasting model that works well for short-term predictions.

---

🔧 How to Run

```bash
git clone https://github.com/your-username/temperature-prediction.git
cd temperature-prediction
pip install -r requirements.txt
jupyter notebook notebooks/Task-1_Temperature.ipynb
