# Eksperimen SML - Credit Risk Prediction

Repository ini dibuat khusus untuk Kriteria 1 Submission Dicoding MSML: eksperimen dataset dan preprocessing otomatis.

## Dataset

Dataset yang digunakan adalah Credit Risk Prediction dengan target `loan_status`.
Dataset mentah disimpan pada:

```text
dataset_raw/credit_risk_dataset.csv.zip
```

## Struktur Repository

```text
Eksperimen_SML_Vicky-Ferdinand-Samuel
├── dataset_raw
├── preprocessing
│   ├── Eksperimen_Vicky-Ferdinand-Samuel.ipynb
│   ├── automate_Vicky-Ferdinand-Samuel.py
│   └── dataset_preprocessing
├── .github
│   └── workflows
│       └── preprocessing.yml
├── requirements.txt
└── README.md
```

## Workflow Preprocessing

Preprocessing yang dilakukan:

1. Load dataset mentah.
2. Analisis missing value dan duplikasi.
3. Hapus duplikasi.
4. Hapus outlier logis berdasarkan umur dan lama bekerja.
5. Stratified train-test split.
6. Median imputation berdasarkan train set.
7. Winsorization 1st/99th percentile berdasarkan train set.
8. Encoding fitur kategorikal.
9. Standard scaling fitur numerik.
10. Export dataset preprocessing.

Output preprocessing:

```text
preprocessing/dataset_preprocessing/credit_risk_train.csv
preprocessing/dataset_preprocessing/credit_risk_test.csv
preprocessing/dataset_preprocessing/credit_risk_processed.csv
```

## Menjalankan Automate Script

```bash
pip install -r requirements.txt
cd preprocessing
python automate_Vicky-Ferdinand-Samuel.py --input ../dataset_raw/credit_risk_dataset.csv.zip --output dataset_preprocessing
```

## GitHub Actions

Workflow `.github/workflows/preprocessing.yml` berjalan pada `push` ke branch `main` dan manual trigger `workflow_dispatch`.

Workflow akan:

1. Checkout repository.
2. Setup Python 3.12.7.
3. Install dependencies.
4. Menjalankan `automate_Vicky-Ferdinand-Samuel.py`.
5. Memvalidasi output dataset.
6. Upload dataset preprocessing sebagai artifact workflow.
