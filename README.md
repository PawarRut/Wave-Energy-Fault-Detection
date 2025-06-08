# 🌊 Wave Energy Fault Detection

A machine learning-powered web application to detect faults in wave energy converter systems based on real-time sensor data. This app is built using **Streamlit** and leverages a **Random Forest** model for prediction.

---

## 📌 Features

- Predicts whether a fault is likely to occur in a wave energy converter.
- Interactive sidebar to input sensor readings.
- Displays prediction result with confidence score.
- Shows model accuracy on test data.
- Background image for enhanced UI.
- Dataset preview included. |

---

## 🚀 How to Run

1. **Clone the repository** or download the project files.

2. Ensure the following files are in the **same directory**:
   - `app.py`
   - `wave_energy_converter_dataset.csv`
   - `background.png`

3. **Install required packages**:

```bash
pip install streamlit pandas scikit-learn pillow
