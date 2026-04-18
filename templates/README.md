# ⚡ Power Estimation System (AI-Based)

An intelligent web-based energy forecasting system using a **CNN-BiLSTM deep learning model** to predict electricity load and provide real-time insights like cost, CO₂ emissions, and alerts.

---

## 🚀 Features

* 🔮 AI-based power load prediction
* 📊 Real-time dashboard interface
* 🌡️ Weather-based inputs (temperature, humidity, rainfall)
* ⚡ Load forecasting with alerts (Normal / Medium / High)
* 💰 Cost estimation (₹)
* 🌍 CO₂ emission estimation
* 🔐 Login system with session management

---

## 🧠 Model Details

* Architecture: **CNN + BiLSTM**
* Framework: PyTorch
* Input Features:

  * Time-based: hour, day, month, weekday, season
  * Historical: lag features (1h, 24h, 168h)
  * Rolling averages
  * Weather data

---

## 🛠️ Tech Stack

* Backend: Flask (Python)
* Frontend: HTML, Tailwind CSS, JavaScript
* ML: PyTorch, NumPy, Pandas, Joblib

---

## 📁 Project Structure

```
Power/
│
├── app.py
├── Results/
│   ├── sota_model_final.pth
│   └── sota_scaler.pkl
│
├── templates/
│   ├── login.html
│   ├── home.html
│   ├── index.html
│   ├── AboutUs.html
│   └── contactUs.html
│
├── static/ (optional)
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```
git clone https://github.com/your-username/power-estimation.git
cd power-estimation
```

---

### 2️⃣ Create virtual environment

```
python -m venv venv
```

Activate:

* Windows:

```
venv\Scripts\activate
```

* Mac/Linux:

```
source venv/bin/activate
```

---

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

### 4️⃣ Run the app

```
python app.py
```

Open browser:

```
http://127.0.0.1:5000
```

---

## 🔐 Login Credentials

```
Username: admin
Password: power123
```

---

## 📡 API Endpoint

### POST `/api/predict`

#### Request:

```json
{
  "temperature": 30,
  "humidity": 60,
  "rain": 0,
  "current_load": 120,
  "day": 15,
  "month": 6,
  "weekday": 4
}
```

#### Response:

```json
{
  "success": true,
  "forecast_kwh": 150.5,
  "co2_kg": 120.4,
  "cost_inr": 1053,
  "alert": "🟡 MEDIUM",
  "confidence": 97.3
}
```

---

## ⚠️ Notes

* Model files are excluded using `.gitignore`
* Ensure `Results/` folder contains model + scaler before running
* Runs on CPU (no GPU required)

---

## 📌 Future Improvements

* 📈 Live graph visualization
* ☁️ Deployment (Render / AWS / Azure)
* 📊 Historical analytics dashboard
* 🔔 Smart alert notifications

---

## 👨‍💻 Author

Final Year Project – Power Estimation using AI

---
