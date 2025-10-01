❤️ Heart Disease Prediction — Codebase

This repository contains the source code for predicting heart disease using:

Classical machine learning models (Logistic Regression, Random Forest, SVM) on tabular clinical data.

Convolutional Neural Network (ResNet-18) on chest X-ray imaging data.

📂 Repository Structure
heart-disease-ml/
│── data/                 
│   ├── uci/heart_cleveland.csv   # clinical dataset
│   ├── cxr/                      # chest X-ray dataset
│
│── src/
│   ├── tabular/                  
│   │   ├── dataset.py            # load & preprocess tabular data
│   │   ├── train.py              # train ML models (LogReg, RF, SVM)
│   │   ├── infer.py              # inference on new tabular input
│   │
│   ├── imaging/
│   │   ├── dataset.py            # load & preprocess X-ray images
│   │   ├── model.py              # ResNet-18 CNN definition
│   │   ├── train.py              # train CNN model
│   │   ├── infer.py              # inference on new X-ray
│   │
│── models/                       # saved trained models
│   ├── tabular/                  # joblib files for ML models
│   ├── imaging/                  # PyTorch .pt CNN model
│
│── results/                      # evaluation outputs & charts
│── streamlit_app.py              # Streamlit UI (clinical + imaging demo)
│── requirements.txt              # dependencies
│── README.md                     # project documentation

⚙️ Installation

Clone the repo:

git clone https://github.com/your-username/heart-disease-ml.git
cd heart-disease-ml


Create virtual environment & install dependencies:

python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.\.venv\Scripts\activate    # Windows
pip install -r requirements.txt

▶️ How to Run
1. Train Tabular Models
python -m src.tabular.train --csv data/uci/heart_cleveland.csv


📌 Models trained: Logistic Regression, Random Forest, SVM
📌 Output: models/tabular/best_model.joblib + metrics JSON

2. Train Imaging Model (CNN)
python -m src.imaging.train --data_dir data/cxr --epochs 10 --batch_size 32


📌 Model: ResNet-18
📌 Output: models/imaging/best_cnn.pt + metrics JSON

3. Inference on New Data

Tabular example:

python -m src.tabular.infer --example


Imaging example:

python -m src.imaging.infer --image data/cxr/Cardiomegaly/sample.png

4. Run Streamlit UI
streamlit run streamlit_app.py


📌 Features:

Enter clinical attributes → predict heart disease.

Upload chest X-ray → CNN predicts cardiomegaly.

📊 Outputs

Tabular: Best model saved as .joblib, ROC-AUC around 0.92 (Logistic Regression).

Imaging: CNN saved as .pt, ROC-AUC around 0.70.

Results: Charts (ROC, accuracy, etc.) stored in results/.

🧩 Dependencies

Python 3.9+

scikit-learn

pandas, numpy

matplotlib, seaborn

PyTorch + torchvision

Streamlit

Install with:

pip install -r requirements.txt

🙌 Acknowledgments

UCI / Kaggle Heart Disease Dataset (Cleveland subset)

Kaggle Chest X-ray Cardiomegaly Dataset

PyTorch, scikit-learn, Streamlit❤️ Heart Disease Prediction — Codebase

This repository contains the source code for predicting heart disease using:

Classical machine learning models (Logistic Regression, Random Forest, SVM) on tabular clinical data.

Convolutional Neural Network (ResNet-18) on chest X-ray imaging data.

Hybrid fusion models that combine predictions from both data types.

📂 Repository Structure
heart-disease-ml/
│── data/                 
│   ├── uci/heart_cleveland.csv     # clinical dataset
│   ├── cxr/                        # chest X-ray dataset
│   ├── pairs/id_map.csv            # mapping for hybrid fusion
│
│── src/
│   ├── tabular/                    
│   │   ├── dataset.py              # load & preprocess tabular data
│   │   ├── train.py                # train ML models (LogReg, RF, SVM)
│   │   ├── infer.py                # inference on tabular input
│   │
│   ├── imaging/
│   │   ├── dataset.py              # load & preprocess X-rays
│   │   ├── model.py                # ResNet-18 CNN
│   │   ├── train.py                # train CNN
│   │   ├── infer.py                # inference on image
│   │
│   ├── hybrid/
│   │   ├── make_toy_pairs.py       # create mapping between data
│   │   ├── late_fusion.py          # late fusion (probability blending)
│   │   ├── mid_fusion_train.py     # mid-level fusion (concatenate embeddings)
│   │
│── models/                         # saved trained models
│── results/                        # charts & metrics
│── streamlit_app.py                # Streamlit UI (tabular + imaging + hybrid)
│── requirements.txt                # dependencies
│── README.md                       # documentation

⚙️ Installation
git clone https://github.com/your-username/heart-disease-ml.git
cd heart-disease-ml
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.\.venv\Scripts\activate    # Windows
pip install -r requirements.txt

▶️ How to Run
1. Train Clinical Models
python -m src.tabular.train --csv data/uci/heart_cleveland.csv


Trains LogReg, RF, SVM.

Saves best model to: models/tabular/best_model.joblib.

2. Train Imaging Model
python -m src.imaging.train --data_dir data/cxr --epochs 10 --batch_size 32


Trains ResNet-18 CNN.

Saves model to: models/imaging/best_cnn.pt.

3. Prepare Hybrid Data

Since UCI and X-ray datasets are separate, we create a toy mapping between them:

python -m src.hybrid.make_toy_pairs --uci_csv data/uci/heart_cleveland.csv --image_root data/cxr --out data/pairs/id_map.csv --n 500

4. Run Hybrid Late Fusion
python -m src.hybrid.late_fusion --mapping data/pairs/id_map.csv --uci_csv data/uci/heart_cleveland.csv --image_root data/cxr


Combines tabular + imaging probabilities.

Saves weights (alpha.json) and metrics to models/hybrid/.

5. (Optional) Run Hybrid Mid-level Fusion
python -m src.hybrid.mid_fusion_train --mapping data/pairs/id_map.csv --uci_csv data/uci/heart_cleveland.csv --image_root data/cxr --epochs 5


Concatenates tabular features + CNN embeddings.

Saves model to: models/hybrid/mid_fusion.pt.

6. Inference

Tabular Example:

python -m src.tabular.infer --example


Imaging Example:

python -m src.imaging.infer --image data/cxr/Cardiomegaly/sample.png


Hybrid Example:

python -m src.hybrid.late_fusion --mapping data/pairs/id_map.csv --uci_csv data/uci/heart_cleveland.csv --image_root data/cxr

7. Run Streamlit UI
streamlit run streamlit_app.py


📌 Features:

Clinical input → ML prediction.

X-ray upload → CNN prediction.

Hybrid fusion → combined prediction.

📊 Results

Clinical (Tabular): Logistic Regression best with ROC-AUC ≈ 0.92.

Imaging (CNN): ResNet-18 achieved ROC-AUC ≈ 0.70.

Hybrid Fusion: Produced blended predictions; performance depended on dataset alignment.

📌 Limitations

Clinical and imaging datasets are not from the same patients → limits hybrid accuracy.

Small dataset size increases variance.

Imaging labels (Cardiomegaly) ≠ direct heart disease labels.

🔮 Future Work

Use true multimodal datasets (paired clinical + imaging per patient).

Improve preprocessing (advanced augmentation, normalization).

Explore attention-based fusion instead of simple late fusion.

Add explainability tools (SHAP for tabular, Grad-CAM for imaging).

🙌 Acknowledgments

UCI / Kaggle Heart Disease Dataset

Kaggle Chest X-ray Dataset

PyTorch, scikit-learn, Streamlit
