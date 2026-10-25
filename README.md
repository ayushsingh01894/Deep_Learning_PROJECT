# Smart Hospitality Management System

**A Unified Deep Learning Platform for the Hospitality Industry — Final Year Project**

> Status: Architecture & dataset finalization complete. Model training has not yet been run.
> **Results will be updated after model training.**

---

## 1. Abstract

The **Smart Hospitality Management System (SHMS)** is a single, integrated Deep Learning platform designed to solve four real, interconnected problems faced by hotels: predicting booking cancellations, understanding guest sentiment from reviews, forecasting room demand, and automatically classifying food images for menu/inventory digitization. Rather than being treated as four disconnected mini-projects, all four capabilities are built as **modules of one system**, sharing a common project structure, a common data/model management convention, and a **single unified dashboard** through which a hotel manager (or an examiner, during viva) can interact with every module from one place.

The system uses distinct Deep Learning architectures suited to each task — a feed-forward **ANN/DNN** for tabular booking-cancellation prediction, an **LSTM** for time-series room-demand forecasting, an **LSTM/embedding-based sequence model** (with an optional transformer/BERT-style variant) for review sentiment analysis, and a **CNN with transfer learning** for food image classification. All models are trained on real, publicly available datasets — no data or metrics in this project are fabricated.

## 2. Problem Statement

Hotels generate large volumes of operational data every day — reservations, cancellations, guest reviews, seasonal occupancy patterns, and food/menu images — but most small and mid-sized hotel management systems treat this data as static records rather than a predictive asset. This leads to:

- Revenue loss from **unanticipated booking cancellations**.
- Missed insight into **guest satisfaction trends** buried in free-text reviews.
- Poor **staffing and inventory planning** due to lack of demand forecasting.
- Manual, error-prone **cataloguing of food images** for digital menus and inventory systems.

There is no single, affordable, integrated system that applies Deep Learning to all four of these problems together for a hospitality business. SHMS addresses this gap.

## 3. Objectives

1. Design and implement **one cohesive Deep Learning system** (not four disjoint projects) covering booking prediction, sentiment analysis, demand forecasting, and food image classification.
2. Source **real, verifiable public datasets** for every module — no synthetic or invented data.
3. Apply the **most suitable DL architecture** to each sub-problem (ANN, LSTM, CNN/Transfer Learning).
4. Build a proper **ML pipeline** for each module: preprocessing → EDA → model training → evaluation → model persistence.
5. Integrate all four trained models behind **one unified Streamlit dashboard**.
6. Produce professional final-year-project documentation: report outline, viva Q&A, evaluation metrics, and visualizations — with all reported results coming only from actual training runs.

## 4. The Four Modules

| # | Module | Task Type | DL Architecture | Output |
|---|--------|-----------|------------------|--------|
| 1 | **Hotel Booking Prediction** | Binary classification (tabular) | ANN / DNN (Keras `Dense` layers) | Probability that a booking will be **cancelled** |
| 2 | **Hotel Review Sentiment Analysis** | Text classification (NLP) | Embedding + **LSTM/BiLSTM** (baseline), optional DistilBERT fine-tune (stretch goal) | Positive / Negative (/ Neutral) sentiment of a review |
| 3 | **Room Demand Forecasting** | Time-series regression/forecasting | **LSTM** (many-to-one sequence model) | Predicted number of bookings/arrivals for upcoming days |
| 4 | **Food Image Classification** | Image classification (Computer Vision) | **CNN with Transfer Learning** (MobileNetV2 / EfficientNetB0, ImageNet-pretrained) | Predicted food category from an uploaded image |

All four modules are exposed as **tabs/sections inside one Streamlit application**, sharing common utility code (config, model-loading, preprocessing helpers) under a single repository.

## 5. Dataset Sources (Verified, Public, Real)

No dataset in this project is invented. All datasets below are real, publicly downloadable, and widely used in academic/Kaggle work. Exact row/image counts will be confirmed and logged in the EDA notebooks once each dataset is downloaded (counts below are as published by the source, not measured by us yet).

### 5.1 Hotel Booking Prediction — *and* — Room Demand Forecasting (Modules 1 & 3)
Both modules reuse the **same base dataset**, since booking-level cancellation records aggregate naturally into a daily demand time series — this is intentional and keeps the system unified rather than fragmented.

- **Dataset:** Hotel Booking Demand
- **Source:** Kaggle — `jessemostipak/hotel-booking-demand`
  (https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- **Original publication:** Antonio, N., de Almeida, A., & Nunes, L. (2019). *Hotel Booking Demand Datasets.* Data in Brief, 22, 41–49.
- **Description:** Real (anonymized) booking records for one city hotel and one resort hotel in Portugal, including arrival date, length of stay, number of guests, market segment, deposit type, and — critically — an `is_canceled` label.
- **Use in Module 1 (Booking Prediction):** `is_canceled` as the binary target; an ANN is trained on engineered tabular features (lead time, deposit type, customer type, market segment, previous cancellations, etc.).
- **Use in Module 3 (Demand Forecasting):** Records are aggregated by `arrival_date` into a **daily bookings/occupancy time series**, which is then windowed and fed into an LSTM to forecast near-future demand.

### 5.2 Hotel Review Sentiment Analysis (Module 2)

- **Dataset:** 515K Hotel Reviews Data in Europe
- **Source:** Kaggle — `jiashenliu/515k-hotel-reviews-data-in-europe`
  (https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe)
- **Description:** ~515,000 real guest reviews of 1,493 luxury hotels across Europe, scraped from Booking.com, with separate `Positive_Review` and `Negative_Review` text fields and a numeric `Reviewer_Score`.
- **Use:** Positive/negative review text is combined into a single review-text-plus-label dataset (label derived from `Reviewer_Score`, e.g. score ≥ 6 → positive), tokenized, and used to train an LSTM/BiLSTM sentiment classifier. A subset will be used if the full 515K rows are too heavy for the training environment.

### 5.3 Food Image Classification (Module 4)

- **Dataset:** Food-101
- **Source:** Kaggle — `kmader/food41` (mirrors the official ETH Zürich Food-101 set), also available via Hugging Face `ethz/food101` and TensorFlow Datasets (`food101`)
  (https://www.kaggle.com/datasets/kmader/food41)
- **Original publication:** Bossard, L., Guillaumin, M., & Van Gool, L. (2014). *Food-101 – Mining Discriminative Components with Random Forests.* ECCV.
- **Description:** 101,000 real photographs across 101 food categories (750 training + 250 test images per class).
- **Use:** Due to training-time constraints typical of a final-year project, we will **fine-tune on a representative subset of classes** (e.g., 10–20 categories relevant to a hotel menu/breakfast buffet — to be finalized before training) using transfer learning on a pretrained CNN backbone. The subset selection and final class list will be documented in the module's own notebook/README before training begins.

> **Note on academic integrity:** Since none of the four models have been trained yet as of this document, no accuracy, F1-score, RMSE, confusion matrix, or other performance numbers are reported anywhere in this README or the project report outline. Every such value will say **"Results will be updated after model training."** until a real training run produces it.

## 6. Technologies

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | TensorFlow / Keras |
| NLP preprocessing | NLTK / re / Keras Tokenizer (optionally Hugging Face `transformers` for the BERT-style stretch goal) |
| Classical/tabular preprocessing | pandas, scikit-learn (`StandardScaler`, `OneHotEncoder`, train/test split) |
| Computer Vision | TensorFlow `ImageDataGenerator` / `tf.data`, transfer learning via `tf.keras.applications` (MobileNetV2 / EfficientNetB0) |
| Visualization / EDA | matplotlib, seaborn, plotly (optional) |
| Application/Dashboard | Streamlit |
| Model persistence | `.h5` / `SavedModel` for Keras models, `joblib`/`pickle` for scalers, encoders, tokenizers |
| Experiment tracking (optional) | simple CSV/JSON logs of metrics per run |

## 7. System Architecture

```
                      ┌───────────────────────────────────────────┐
                      │        Streamlit Unified Dashboard         │
                      │  (Smart Hospitality Management System)     │
                      ├───────────────┬───────────────┬────────────┤
                      │  Booking      │  Sentiment    │  Demand    │  Food Image
                      │  Prediction   │  Analysis     │  Forecast  │  Classification
                      │  tab          │  tab          │  tab       │  tab
                      └──────┬────────┴──────┬────────┴─────┬──────┴──────┬─────┘
                             │                │              │             │
                     ┌───────▼──────┐ ┌───────▼──────┐ ┌─────▼──────┐ ┌────▼───────┐
                     │  ANN model   │ │ LSTM/BiLSTM  │ │ LSTM       │ │ CNN         │
                     │  (.h5)       │ │ sentiment    │ │ forecast   │ │ (Transfer   │
                     │  + scaler/   │ │ model (.h5)  │ │ model      │ │ Learning)   │
                     │  encoders    │ │ + tokenizer  │ │ (.h5) +    │ │ (.h5)       │
                     │              │ │              │ │ scaler     │ │             │
                     └──────┬───────┘ └──────┬───────┘ └─────┬──────┘ └─────┬──────┘
                            │                │               │              │
                     ┌──────▼───────┐ ┌──────▼───────┐ ┌─────▼──────┐ ┌─────▼──────┐
                     │ Hotel Booking│ │ 515K Hotel   │ │ Hotel      │ │ Food-101    │
                     │ Demand       │ │ Reviews (EU) │ │ Booking    │ │ dataset     │
                     │ dataset      │ │ dataset      │ │ Demand     │ │             │
                     │              │ │              │ │ (aggregated)│ │            │
                     └──────────────┘ └──────────────┘ └────────────┘ └────────────┘
```

Each module is trained **independently** (its own preprocessing pipeline, its own script, its own saved model artifact), but all four are loaded together at runtime by one Streamlit app, giving the appearance and behavior of a single integrated hospitality platform.

## 8. Methodology (per module)

For every module the same disciplined pipeline is followed:

1. **Data acquisition** — download the verified public dataset (instructions in §12).
2. **Exploratory Data Analysis (EDA)** — distributions, class balance, missing values, correlation/word-frequency analysis, sample images, etc. Saved as plots in `reports/figures/`.
3. **Preprocessing** — cleaning, encoding, scaling, tokenization/padding, image resizing/augmentation, train/val/test splitting.
4. **Model design & training** — architecture defined in `src/<module>/model.py`, trained via `src/<module>/train.py`, with checkpoints and early stopping.
5. **Evaluation** — accuracy/precision/recall/F1 (classification), RMSE/MAE (forecasting), confusion matrix, training curves — all computed from the actual trained model, saved under `reports/`.
6. **Model persistence** — final model + any preprocessing objects (scaler, tokenizer, label encoder) saved under `models/<module>/`.
7. **Integration** — a thin inference wrapper in `app/` loads the saved artifacts and exposes them to the Streamlit dashboard.

## 9. Model Architectures (planned)

- **Module 1 — Booking Prediction (ANN):** Input (engineered tabular features) → Dense(128, ReLU) → Dropout → Dense(64, ReLU) → Dropout → Dense(1, Sigmoid). Loss: binary cross-entropy. Optimizer: Adam.
- **Module 2 — Sentiment Analysis (LSTM):** Embedding layer → BiLSTM(128) → Dropout → Dense(64, ReLU) → Dense(1 or 3, Sigmoid/Softmax). Loss: binary/categorical cross-entropy. *(Optional stretch: fine-tuned DistilBERT via Hugging Face for comparison.)*
- **Module 3 — Demand Forecasting (LSTM):** Sliding-window sequences of daily booking counts → LSTM(64) → LSTM(32) → Dense(1). Loss: MSE. Evaluated with RMSE/MAE.
- **Module 4 — Food Classification (CNN / Transfer Learning):** Pretrained MobileNetV2 (ImageNet weights, frozen base) → GlobalAveragePooling2D → Dense(128, ReLU) → Dropout → Dense(N classes, Softmax). Fine-tuning of top layers after initial training.

Exact hyperparameters (layer sizes, learning rate, epochs, batch size) will be finalized and documented in each module's training script and its own README once implementation begins.

## 10. Project Structure

```
smart-hospitality-management-system/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── booking_demand/
│   │   ├── hotel_reviews/
│   │   └── food101/
│   └── processed/
│       ├── booking_demand/
│       ├── hotel_reviews/
│       ├── room_demand/
│       └── food101/
├── notebooks/
│   ├── 01_booking_prediction_eda.ipynb
│   ├── 02_sentiment_analysis_eda.ipynb
│   ├── 03_demand_forecasting_eda.ipynb
│   └── 04_food_classification_eda.ipynb
├── src/
│   ├── common/
│   │   ├── config.py
│   │   └── utils.py
│   ├── booking_prediction/
│   │   ├── preprocess.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── sentiment_analysis/
│   │   ├── preprocess.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── demand_forecasting/
│   │   ├── preprocess.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── evaluate.py
│   └── food_classification/
│       ├── preprocess.py
│       ├── model.py
│       ├── train.py
│       └── evaluate.py
├── models/
│   ├── booking_prediction/
│   ├── sentiment_analysis/
│   ├── demand_forecasting/
│   └── food_classification/
├── reports/
│   ├── figures/
│   └── metrics/
├── tests/
│   ├── test_booking_prediction.py
│   ├── test_sentiment_analysis.py
│   ├── test_demand_forecasting.py
│   └── test_food_classification.py
├── app/
│   ├── Home.py                  # Streamlit entry point (unified dashboard)
│   └── pages/
│       ├── 1_Hotel_Booking_Prediction.py
│       ├── 2_Hotel_Review_Sentiment_Analysis.py
│       ├── 3_Room_Demand_Forecasting.py
│       └── 4_Food_Image_Classification.py
├── docs/
│   ├── project_report_outline.md
│   └── viva_questions_and_answers.md
└── .gitignore
```

- **Booking Prediction:** Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrix.
- **Sentiment Analysis:** Accuracy, Precision/Recall/F1 (per class), confusion matrix.
- **Demand Forecasting:** RMSE, MAE, MAPE, actual-vs-predicted plot.
- **Food Classification:** Top-1 accuracy, per-class precision/recall, confusion matrix, sample misclassifications (error analysis).


- 🏨 **Hotel Booking Prediction** — enter booking details → get cancellation probability.
- 💬 **Hotel Review Sentiment Analysis** — paste/enter a review → get sentiment label + confidence.
- 📈 **Room Demand Forecasting** — view historical demand + forecasted demand for upcoming days.
- 🍽️ **Food Image Classification** — upload a food photo → get predicted category + confidence.

## 11. Future Scope

- Fine-tune a transformer-based model (BERT/DistilBERT) for sentiment analysis and compare against the LSTM baseline.
- Extend demand forecasting to multivariate forecasting (weather, local events, seasonality features).
- Expand food classification to the full 101-class Food-101 set with a larger compute budget.
- Add a recommendation module (e.g., personalized room/offer recommendations) as a fifth module.
- Deploy the system with a proper backend (FastAPI) and database instead of only a Streamlit demo.

## 12. References

1. Antonio, N., de Almeida, A., & Nunes, L. (2019). *Hotel Booking Demand Datasets.* Data in Brief, 22, 41–49.
2. Bossard, L., Guillaumin, M., & Van Gool, L. (2014). *Food-101 – Mining Discriminative Components with Random Forests.* ECCV 2014.
3. Kaggle Dataset: Hotel Booking Demand — https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
4. Kaggle Dataset: 515K Hotel Reviews Data in Europe — https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe
5. Kaggle Dataset: Food Images (Food-101) — https://www.kaggle.com/datasets/kmader/food41
6. TensorFlow Datasets — Food101 — https://www.tensorflow.org/datasets/catalog/food101
7. Hugging Face Datasets — ethz/food101 — https://huggingface.co/datasets/ethz/food101
8. Claude AI And some chatgpt 

- Viva questions & answers
 Done ✅ 
  Ayush Singh 
