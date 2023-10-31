# Medical Health Buddy

## 📍 Introduction

Medical Health Buddy offers multiple features, including:

* **Disease prediction**
* **Lab report analysis**

## 📦 Features

**Disease prediction**

User enters the symptoms they are experiencing. It will then use its TensorFlow Keras model to predict the top 4 diseases that the user might be suffering from.

The TensorFlow Keras model was trained on a dataset of over 5000 patient records. The model has an accuracy of over 85% on most instances.

**Lab report analysis**

User enters the important information from their lab reports. It will then use a set of 3 unique models to predict whether the user is suffering from a liver disease, heart disease, or diabetes.

Each of these models were trained on a dataset of about 1000 patient records. The models have an accuracy of over 90%.

## 📂 Repository Structure

```sh
└── Medical-Health-Buddy/
    ├── .streamlit/
    │   └── config.toml
    ├── Backend/
    │   ├── diabetes_model.py
    │   ├── heart_disease_model.py
    │   ├── liver_model.py
    │   └── model.py
    ├── Dataset/
    │   ├── diseases.txt
    │   ├── mental_health_diseases.txt
    │   ├── mental_health_symptoms.txt
    │   └── symptoms.txt
    ├── Frontend/
    │   ├── app.html
    │   ├── app.py
    │   └── pages/
    │       ├── diabetes.py
    │       ├── disease_diagnosis.py
    │       ├── heart_disease.py
    │       ├── liver.py
    │       └── mental_health.py
    ├── Models/
    │   ├── model_dl.keras
    ├── Resources/
    └── requirements.txt

```

---

## Future plans

We plan to add functionality to automatically parse lab reports, rather than requiring users to manually enter the details in correct parameters. This will make it even easier to use and make it more accessible to users who are not familiar with medical terminology.

## 🚀 Getting Started

### 🔧 Installation

1. Clone the Medical-Health-Buddy repository:
```
git clone https://github.com/arnav003/Medical-Health-Buddy
```

2. Change to the project directory:
```
cd Medical-Health-Buddy
```

3. Install the dependencies:
```
pip install -r requirements.txt
```

### 🤖 Running

```
python main.py
```

---

## Contact

If you have any questions or feedback, please contact us at [lalaarnav003@gmail.com](mailto:lalaarnav003@gmail.com)