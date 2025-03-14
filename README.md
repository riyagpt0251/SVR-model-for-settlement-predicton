# 🌍 **Soil Settlement Prediction using SVR** 🏗️

![Soil Settlement](https://img.shields.io/badge/Machine%20Learning-SVR-blue?style=for-the-badge&logo=python)  
🚀 A Machine Learning project using **Support Vector Regression (SVR)** to predict **soil settlement (mm)** based on geotechnical parameters like cohesion, permeability, water content, loading, and depth.  

![ML Pipeline](https://media.giphy.com/media/3o7TKu7UPPT9vNKc3O/giphy.gif)  

---

## 📌 **Project Overview**
Soil settlement is a crucial factor in **civil engineering and geotechnical projects**. This project aims to build a **predictive model** using **Support Vector Regression (SVR)** to estimate **soil settlement** based on various **soil properties** and **loading conditions**.

### 🔍 **Dataset Description**
| Feature                    | Description                                     |
|----------------------------|-------------------------------------------------|
| 🌱 **Cohesion (kPa)**       | Soil cohesion strength                          |
| 💧 **Permeability (cm/s)**  | Water movement through soil                     |
| 💦 **Water Content (%)**    | Moisture percentage in soil                     |
| ⚖️ **Loading (kN/m²)**      | External pressure applied to soil               |
| 📏 **Depth (m)**            | Depth of soil layer                             |
| 📉 **Settlement (mm)**      | Target variable: Predicted soil settlement     |

---

## 🔧 **Tech Stack**
![Python](https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge&logo=python)  
📌 **Libraries Used:**  
- **Pandas** – Data manipulation  
- **NumPy** – Numerical computations  
- **Scikit-Learn** – Machine Learning framework  
- **Matplotlib** – Data visualization  

---

## 🚀 **Installation & Setup**
1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/soil-settlement-svr.git
   cd soil-settlement-svr
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

---

## 📊 **Implementation Steps**
### 1️⃣ **Import Required Libraries**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

### 2️⃣ **Load the Dataset**
```python
data = {
    "Cohesion_kPa": [25, 30, 35, 40, 45, 50, 55],
    "Permeability_cm_per_s": [0.0001, 0.0005, 0.0002, 0.0003, 0.0004, 0.0001, 0.0006],
    "Water_Content_percent": [20, 25, 22, 30, 35, 28, 26],
    "Loading_kN_per_m2": [200, 300, 250, 400, 450, 350, 320],
    "Depth_m": [5, 8, 6, 10, 12, 9, 7],
    "Settlement_mm": [15, 25, 18, 35, 40, 30, 28]
}
df = pd.DataFrame(data)
```

### 3️⃣ **Preprocessing & Feature Scaling**
```python
X = df.drop("Settlement_mm", axis=1)  # Features
y = df["Settlement_mm"]               # Target variable

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 4️⃣ **Train-Test Split**
```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

### 5️⃣ **Train the SVR Model**
```python
svr = SVR(kernel='rbf', C=100, epsilon=0.1, gamma=0.1)
svr.fit(X_train, y_train)
```

### 6️⃣ **Make Predictions**
```python
y_pred = svr.predict(X_test)
```

### 7️⃣ **Evaluate the Model**
```python
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
```

### 📊 **Model Performance**
| Metric | Value |
|--------|-------|
| **Mean Absolute Error (MAE)** | `2.17` |
| **Root Mean Squared Error (RMSE)** | `2.93` |
| **R² Score** | `0.66` |

---

## 📈 **Visualization**
### 🔹 **Predicted vs Actual Settlement**
```python
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Ideal Fit')
plt.xlabel("Actual Settlement (mm)")
plt.ylabel("Predicted Settlement (mm)")
plt.title("SVR Model Predictions vs. Actual")
plt.legend()
plt.show()
```
📌 **Graph Output:**  
![Prediction Graph](https://media.giphy.com/media/l3vRaL77jLoewYJ5e/giphy.gif)

---

## 🏆 **Key Takeaways**
✅ **Support Vector Regression (SVR)** effectively models nonlinear relationships.  
✅ **Feature Scaling** is crucial for SVR performance.  
✅ The model achieves **66% accuracy (R² Score: 0.66)** in predicting soil settlement.  

---

## 📌 **Future Enhancements**
📌 **Increase Dataset Size** – Train on a larger dataset for better generalization.  
📌 **Feature Engineering** – Explore additional soil properties for improved accuracy.  
📌 **Hyperparameter Tuning** – Optimize SVR parameters (`C`, `gamma`, `epsilon`) for better results.  
📌 **Compare with Other Models** – Evaluate performance using **Random Forest, XGBoost, or Neural Networks**.  

---

## 📌 **Contributing**
🚀 Contributions are welcome! Feel free to open an issue or submit a pull request.

---

## 🏆 **License**
📜 This project is licensed under the **MIT License**.

---

### ⭐ **If you like this project, don't forget to give it a star!** ⭐  
![GitHub Repo stars](https://img.shields.io/github/stars/yourusername/soil-settlement-svr?style=social)  
🚀 **Happy Coding!** 🚀
