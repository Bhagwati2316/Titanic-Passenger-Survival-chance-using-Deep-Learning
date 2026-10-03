# PROJECT TITLE :

Titanic-Passenger-Survival-chance-using-Deep-Learning

# DESCRIPTION:

The Application predicts that whether if a passenger is going to survive the titanic journey or not. using deep learning ANN


Titanic Passenger Survival Prediction (ANN + Streamlit)

A deep learning project that predicts whether a Titanic passenger would have survived, based on ticket class, gender, family size, fare and port of embarkation. The model is a feed-forward Artificial Neural Network built with **TensorFlow/Keras**, trained on a **~1 million row** Titanic-style dataset, and served through an interactive **Streamlit** web app.

---

## 📌 Features

- End-to-end ML pipeline: data cleaning → encoding → scaling → model training → deployment
- ANN binary classifier (sigmoid output = survival probability)
- Saved preprocessing objects (`LabelEncoder`, `OneHotEncoder`, `StandardScaler`) so inference uses the exact same transformations as training
- Interactive Streamlit UI to enter passenger details and get a survival probability
- Standalone inference notebook (`Predict.ipynb`) for testing a single passenger

---

## 🗂️ PROJECT STRUCTURE

```
├── Project.ipynb        # Data preprocessing, training and model saving
├── Predict.ipynb        # Single-passenger inference walkthrough
├── app.py               # Streamlit web app
├── model.h5             # Trained Keras model
├── scaler.pkl           # StandardScaler (Pclass, SibSp, Parch, Fare)
├── label_encoder.pkl    # LabelEncoder (Sex)
├── onehot_encoder.pkl   # OneHotEncoder (Embarked)
├── requirements.txt     # Python dependencies
└── README.md
```

> The training dataset (`huge_1M_titanic.csv`) is not included in this repository because of its size.

---

# PIPLINE
...
Training 

<img width="702" height="214" alt="Image" src="https://github.com/user-attachments/assets/1f0a1ebd-6beb-41ea-a67e-0f4b16fae3e7" />

After training Streamlit app (app.py) 

<img width="702" height="214" alt="Image" src="https://github.com/user-attachments/assets/bdd8553f-eee2-4674-a283-b3652592bc3f" />
...

...
# DEMO 

<img width="1906" height="1020" alt="Image" src="https://github.com/user-attachments/assets/57eb67bc-02e5-40fb-864a-fbaacf6240e0" />

<img width="778" height="777" alt="Image" src="https://github.com/user-attachments/assets/5f7eb5b0-d3dd-48bc-aff1-cda4aa52dd49" />
...

## 📊 DATASET

- **Rows:** 1,000,000 | **Original columns:** 12
- **Target:** `Survived` (0 = did not survive, 1 = survived)
- **Input features used (6):**

| Feature    | Description                                  |
|------------|----------------------------------------------|
| `Pclass`   | Passenger class (1, 2, 3)                    |
| `Sex`      | male / female                                |
| `SibSp`    | Number of siblings / spouses aboard          |
| `Parch`    | Number of parents / children aboard          |
| `Fare`     | Ticket fare                                  |
| `Embarked` | Port: Southampton, Cherbourg, Queenstown     |

- **Dropped columns:** `PassengerId`, `Name`, `Age` (about 20% missing), `Ticket`, `Cabin` (about 77% missing)

---

## ⚙️ PREPROCESSING

1. Dropped irrelevant / heavily-missing columns
2. Removed rows with a missing `Embarked` value
3. Renamed port codes (`S`, `C`, `Q`) to full names
4. Converted `Fare` to integer
5. **Encoding**
   - `Sex` → `LabelEncoder` (female = 0, male = 1)
   - `Embarked` → `OneHotEncoder` (3 binary columns)
6. **Scaling:** `StandardScaler` on `Pclass`, `SibSp`, `Parch`, `Fare`
7. Train / validation split of 80% / 20% (plus a small hold-out test set)

---

## 🧠 MODEL ARCHITECTURE

| Layer    | Units | Activation |
|----------|-------|------------|
| Input    | 8     | n/a        |
| Dense    | 128   | ReLU       |
| Dense    | 64    | ReLU       |
| Dense    | 32    | ReLU       |
| Output   | 1     | Sigmoid    |

- **Total parameters:** 11,521
- **Optimizer:** Adam (learning rate = 0.01)
- **Loss:** Binary Crossentropy
- **Metric:** Accuracy
- **Callback:** `EarlyStopping` (monitor = `val_loss`, patience = 5, `restore_best_weights=True`)
- **Epochs:** up to 10

### RESULT

| Metric              | Value   |
|---------------------|---------|
| Validation accuracy | ~85%    |
| Validation loss     | ~0.33   |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 🖥️ How to Use the App

1. Select the passenger class, gender, number of siblings/spouses and parents/children
2. Enter the fare and choose the embarkation port
3. Click **Predict Survival Chance**
4. The app shows the survival probability and a final verdict (probability > 0.5 means likely to survive)

### Example (from `Predict.ipynb`)

```python
{'Pclass': 1, 'Sex': 'female', 'SibSp': 2, 'Parch': 1, 'Fare': 76, 'Embarked': 'Chebourg'}
```

```
Probability : 0.9999
Result      : The passenger likely survive the titanic journey
```

---

## 🛠️ TECH STACK

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas, NumPy
- Matplotlib, TensorBoard
- Streamlit

---

## 🔮 FUTURE IMPROVEMENTS

- Include `Age` using proper imputation
- Save the model in the native `.keras` format instead of legacy `.h5`
- Add test-set evaluation (confusion matrix, precision, recall, F1)
- Deploy the app on Streamlit Community Cloud
- Add app screenshots / demo GIF



⭐ If you found this project useful, consider giving it a star!
