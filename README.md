# 🐧 Penguin Species Classifier

A beginner machine learning project that predicts a penguin's species (**Adelie**, **Chinstrap**, or **Gentoo**) from its physical measurements, comparing a **Decision Tree** and **K-Nearest Neighbors (KNN)**.

## Project Overview

Researchers studying penguins often need to identify species quickly in the field. This project trains two simple models to classify species from four measurements:

- Bill length (mm)
- Bill depth (mm)
- Flipper length (mm)
- Body mass (g)

**Steps in the notebook:**
1. Load the dataset and check for missing values
2. Clean the data (drop rows with missing measurements) and split into train/test sets
3. Train a Decision Tree and a KNN model (with feature scaling)
4. Evaluate with accuracy, classification reports, and confusion matrices
5. Compare the models (including 5-fold cross-validation) and state conclusions

## Dataset

**Palmer Penguins** by Dr. Kristen Gorman and the Palmer Station LTER.

- Loaded in code with `seaborn.load_dataset("penguins")`
- Source: https://github.com/allisonhorst/palmerpenguins
- Also available on Kaggle: https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data

344 rows originally; 342 remain after removing rows with missing measurements.

## Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/swathi25o3/Penguin_species_classifier.git
cd Penguin_species_classifier

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook penguin_species_classifier.ipynb
```

An internet connection is needed the first time to download the dataset through seaborn.

## Results

| Model | Test Accuracy | 5-Fold CV Accuracy |
|-------|---------------|--------------------|
| Decision Tree (max_depth=4) | 100% | 96.48% |
| KNN (k=5, scaled) | 100% | 98.83% |

## Conclusion
1. Which model performed better?
On the test set, both models scored 100% (69 of 69 correct), so the single split could not tell them apart.
 In 5-fold cross-validation, KNN averaged 98.83% (+/- 0.58%) and the Decision Tree averaged 96.48% (+/- 1.51%). So KNN performed slightly better and was also more stable across folds, though both models work very well for this problem.
2. Where did the errors happen?
 There were no errors on the test set: every confusion matrix entry sits on the diagonal. The small gap in cross-validation means a few penguins were misclassified in other splits, most likely Adelie and Chinstrap, which overlap in the pairplot. Gentoo is clearly separated by flipper length and body mass
3. What I learnt:
  -> Real datasets have missing values, and checking for them is a necessary first step.
  -> KNN needs feature scaling because it relies on distances; Decision Trees do not.
  -> Decision Trees are easy to interpret (we can see which features matter), while KNN is simple but slower on large data.
  -> A single train/test split on a small dataset can be misleading, so cross-validation gives a fairer comparison.
  -> Both models perform well because the species are quite separable by physical measurements.

## Tools Used

Python, pandas, NumPy, seaborn, matplotlib, scikit-learn, Jupyter
