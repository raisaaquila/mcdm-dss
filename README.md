
# 🧮 MCDM Decision Support App

Made for Decision Support System coursework by Raisa Aquila Zahra Kholiq and Kireina Kalila Putri.

This is a Streamlit-based web application for performing decision analysis using various **Multi-Criteria Decision Making (MCDM)** methods:

- **SAW** (Simple Additive Weighting)
- **WP** (Weighted Product)
- **TOPSIS** (Technique for Order Preference by Similarity to Ideal Solution)
- **AHP** (Analytic Hierarchy Process)

---

## 🚀 Features

- Compare alternatives using four MCDM techniques.
- Dynamic number of criteria and alternatives.
- Manual input or CSV upload for matrices.
- Visual ranking
- Consistency ratio reports (AHP).
- Easy-to-use interface with sidebar navigation.

---

## 🛠 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/MCDM-App.git
cd MCDM-App
````

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is not available, install dependencies manually:

```bash
pip install streamlit numpy pandas
```

---

## ▶️ Run the App

To start the app, run:

```bash
streamlit run Home.py
```

This will launch the app in your default browser. Use the sidebar to switch between SAW, WP, TOPSIS, and AHP modules.

---

## 📝 Notes

* The app supports both manual matrix input and CSV uploads for flexibility.
* AHP includes automatic consistency ratio calculation.
* Suitable for educational or operational decision-making.

```
