# %% [markdown]
# # Car Price Prediction with Machine Learning
# **CodeAlpha Data Science Internship — Task 3**
#
# **Goal:** Predict the selling price of a used car (or two-wheeler) from its features:
# brand, age, showroom price, kilometres driven, fuel type, seller type, transmission and
# number of previous owners.
#
# **Data:** [Car Price Prediction — Used Cars (Kaggle)](https://www.kaggle.com/datasets/vijayaadithyanvg/car-price-predictionused-cars),
# listings from the Indian used-vehicle market. Prices are in **lakhs of rupees**
# (1 lakh = ₹1,00,000).
#
# **Steps:** load & clean → EDA → feature engineering → train & compare regression models →
# evaluate → interpret → predict a new car.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sns.set_theme(style="whitegrid")
RANDOM_STATE = 42

# %% [markdown]
# ## 1. Load and inspect the data

# %%
df = pd.read_csv("data/car data.csv")
# Some copies of this dataset use slightly different column names, so we standardise them.
df = df.rename(columns={"Kms_Driven": "Driven_kms", "Seller_Type": "Selling_type"})
print("Shape:", df.shape)
df.head()

# %%
df.info()

# %%
print("Missing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
for col in ["Fuel_Type", "Selling_type", "Transmission", "Owner"]:
    print(f"\n{col}:\n{df[col].value_counts().to_string()}")

# %%
df = df.drop_duplicates().reset_index(drop=True)
df.describe().round(2)

# %% [markdown]
# **Notes**
# - No missing values. Duplicate rows (if any) are removed above.
# - `Selling_Price` and `Present_Price` (today's showroom price of the same model) are in lakhs.
# - The data mixes **cars and two-wheelers** (e.g. *Royal Enfield*, *Bajaj Pulsar*).

# %% [markdown]
# ## 2. Feature engineering
# - **Car_Age**: how old the vehicle is. Calculated from the most recent year in the data,
#   which is when the listings were collected.
# - **Brand**: taken from `Car_Name`. Many names are just the model (e.g. *city*, *swift*),
#   so we map models to their maker. Brand is a stand-in for **brand goodwill**.
# - **Vehicle_Type**: car or two-wheeler.
# - **Depreciation** (for EDA only): how much of the showroom price has been lost.

# %%
MODEL_TO_BRAND = {
    # Maruti Suzuki
    "ritz": "Maruti", "sx4": "Maruti", "ciaz": "Maruti", "wagon r": "Maruti", "swift": "Maruti",
    "vitara brezza": "Maruti", "s cross": "Maruti", "alto 800": "Maruti", "ertiga": "Maruti",
    "dzire": "Maruti", "alto k10": "Maruti", "ignis": "Maruti", "800": "Maruti", "baleno": "Maruti",
    "omni": "Maruti", "s-cross": "Maruti",
    # Toyota
    "fortuner": "Toyota", "corolla altis": "Toyota", "corolla": "Toyota", "innova": "Toyota",
    "etios cross": "Toyota", "etios g": "Toyota", "etios liva": "Toyota", "etios gd": "Toyota",
    "camry": "Toyota", "land cruiser": "Toyota",
    # Honda (cars)
    "city": "Honda", "brio": "Honda", "amaze": "Honda", "jazz": "Honda",
    # Hyundai
    "i20": "Hyundai", "grand i10": "Hyundai", "i10": "Hyundai", "eon": "Hyundai",
    "xcent": "Hyundai", "elantra": "Hyundai", "creta": "Hyundai", "verna": "Hyundai",
}
TWO_WHEELER_BRANDS = ["royal enfield", "bajaj", "hero", "honda", "yamaha", "tvs", "ktm",
                      "suzuki", "um", "hyosung", "mahindra mojo", "activa", "harley", "kawasaki"]

def get_brand(name):
    n = str(name).strip().lower()
    if n in MODEL_TO_BRAND:
        return MODEL_TO_BRAND[n], "Car"
    for b in TWO_WHEELER_BRANDS:
        if n.startswith(b):
            brand = "Honda" if b == "activa" else ("Mahindra" if b == "mahindra mojo" else b.title())
            return brand, "Two-wheeler"
    return n.split()[0].title(), "Car"

df[["Brand", "Vehicle_Type"]] = df["Car_Name"].apply(lambda n: pd.Series(get_brand(n)))
REF_YEAR = df["Year"].max() + 1
df["Car_Age"] = REF_YEAR - df["Year"]
df["Depreciation_%"] = (1 - df["Selling_Price"] / df["Present_Price"]) * 100
# Prices and kilometres are very skewed, so we also keep their logs.
# A log-log linear model says "price falls by a percentage", which is how depreciation works.
df["Log_Present_Price"] = np.log(df["Present_Price"])
df["Log_Driven_kms"] = np.log(df["Driven_kms"])

print("Reference year:", REF_YEAR)
print(df["Vehicle_Type"].value_counts().to_string())
print("\nBrands:\n", df["Brand"].value_counts().to_string())

# %% [markdown]
# ## 3. Exploratory data analysis

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sns.histplot(df["Selling_Price"], bins=40, kde=True, ax=axes[0], color="#3182CE")
axes[0].set_title("Selling price distribution (lakhs)")
sns.histplot(np.log1p(df["Selling_Price"]), bins=40, kde=True, ax=axes[1], color="#38A169")
axes[1].set_title("log(1 + Selling price) — much more symmetric")
plt.tight_layout()
fig.savefig("images/price_distribution.png", dpi=120, bbox_inches="tight")

# %% [markdown]
# The price is heavily **right-skewed**: most vehicles are cheap and a few are very expensive.
# Linear models work better on the log of the price, so we will try that.

# %%
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
sns.scatterplot(data=df, x="Present_Price", y="Selling_Price", hue="Vehicle_Type", ax=axes[0, 0], alpha=0.7)
axes[0, 0].set_title("Selling price vs showroom price")
sns.scatterplot(data=df, x="Car_Age", y="Selling_Price", hue="Vehicle_Type", ax=axes[0, 1], alpha=0.7)
axes[0, 1].set_title("Selling price vs age")
sns.scatterplot(data=df, x="Driven_kms", y="Selling_Price", hue="Vehicle_Type", ax=axes[1, 0], alpha=0.7)
axes[1, 0].set_title("Selling price vs kilometres driven")
axes[1, 0].set_xscale("log")
sns.boxplot(data=df[df["Vehicle_Type"] == "Car"], x="Fuel_Type", y="Selling_Price", hue="Transmission", ax=axes[1, 1])
axes[1, 1].set_title("Cars: price by fuel type and transmission")
plt.tight_layout()
fig.savefig("images/eda_relationships.png", dpi=120, bbox_inches="tight")

# %%
cars = df[df["Vehicle_Type"] == "Car"]
brand_stats = cars.groupby("Brand").agg(
    Listings=("Selling_Price", "size"),
    Avg_Selling_Price=("Selling_Price", "mean"),
    Avg_Depreciation_pct=("Depreciation_%", "mean"),
).sort_values("Avg_Selling_Price", ascending=False)
brand_stats.round(2)

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sns.barplot(data=cars, x="Brand", y="Selling_Price", order=brand_stats.index, ax=axes[0], color="#3182CE", errorbar=None)
axes[0].set_title("Average selling price by car brand")
axes[0].set_ylabel("Lakhs")
age_dep = df.groupby(["Car_Age", "Vehicle_Type"])["Depreciation_%"].median().reset_index()
sns.lineplot(data=age_dep, x="Car_Age", y="Depreciation_%", hue="Vehicle_Type", marker="o", ax=axes[1])
axes[1].set_title("Median depreciation by age")
axes[1].set_ylabel("% of showroom price lost")
plt.tight_layout()
fig.savefig("images/brand_depreciation.png", dpi=120, bbox_inches="tight")

# %%
num_cols = ["Selling_Price", "Present_Price", "Driven_kms", "Car_Age", "Owner"]
plt.figure(figsize=(6.5, 5))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation between numeric features")
plt.savefig("images/correlation.png", dpi=120, bbox_inches="tight")

# %% [markdown]
# ## 4. Prepare data for modelling
# - **Features:** log(Present_Price), log(Driven_kms), Car_Age, Owner, Fuel_Type, Selling_type,
#   Transmission, Vehicle_Type, Brand. Using logs stops a few very expensive or very high-mileage
#   vehicles (e.g. a ₹92-lakh Land Cruiser, a 5,00,000 km car) from dominating the linear models.
# - `Car_Name` is dropped: it has too many unique values, and the useful part of it is
#   already in `Brand` and `Vehicle_Type`.
# - Categorical columns are one-hot encoded. Numeric columns are scaled (needed by the linear models).
# - 80% train / 20% test split.

# %%
numeric = ["Log_Present_Price", "Log_Driven_kms", "Car_Age", "Owner"]
categorical = ["Fuel_Type", "Selling_type", "Transmission", "Vehicle_Type", "Brand"]

X = df[numeric + categorical]
y = df["Selling_Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
print("Train:", X_train.shape, "| Test:", X_test.shape)

def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=3), categorical),
    ])

# %% [markdown]
# ## 5. Train and compare models
# Linear models predict **log(price)** (then convert back), because the price is skewed.
# Tree models handle skew on their own.

# %%
def log_target(model):
    return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)

models = {
    "Linear Regression": log_target(LinearRegression()),
    "Ridge Regression": log_target(Ridge(alpha=1.0)),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                   max_depth=3, random_state=RANDOM_STATE),
}

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
rows, fitted = [], {}
for name, model in models.items():
    pipe = Pipeline([("prep", make_preprocessor()), ("model", model)])
    cv_r2 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2")
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    fitted[name] = pipe
    rows.append({
        "Model": name,
        "CV R² (train)": cv_r2.mean(),
        "Test R²": r2_score(y_test, pred),
        "Test MAE (lakhs)": mean_absolute_error(y_test, pred),
        "Test RMSE (lakhs)": np.sqrt(mean_squared_error(y_test, pred)),
    })

results = pd.DataFrame(rows).sort_values("CV R² (train)", ascending=False).reset_index(drop=True)
results.round(3)

# %% [markdown]
# We pick the best model by **cross-validated R²** on the training set, not by test score.
# This keeps the test set as a fair, unseen check.

# %%
best_name = results.loc[0, "Model"]
best = fitted[best_name]
y_pred = best.predict(X_test)
print(f"Best model: {best_name}")
print(f"Test R²:   {r2_score(y_test, y_pred):.3f}  (share of price variation explained)")
print(f"Test MAE:  {mean_absolute_error(y_test, y_pred):.2f} lakhs  (average error)")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f} lakhs")

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(y_test, y_pred, alpha=0.7, color="#3182CE")
lim = [0, max(y_test.max(), y_pred.max()) * 1.05]
axes[0].plot(lim, lim, "r--", label="Perfect prediction")
axes[0].set_xlabel("Actual price (lakhs)")
axes[0].set_ylabel("Predicted price (lakhs)")
axes[0].set_title(f"Actual vs predicted — {best_name}")
axes[0].legend()
residuals = y_test - y_pred
sns.histplot(residuals, bins=30, kde=True, ax=axes[1], color="#DD6B20")
axes[1].axvline(0, color="black", ls="--")
axes[1].set_title("Prediction errors (actual − predicted)")
axes[1].set_xlabel("Lakhs")
plt.tight_layout()
fig.savefig("images/actual_vs_predicted.png", dpi=120, bbox_inches="tight")

# %% [markdown]
# ## 6. What drives the price?

# %%
rf = fitted["Random Forest"]
feat_names = rf.named_steps["prep"].get_feature_names_out()
feat_names = [f.split("__", 1)[1].replace("infrequent_sklearn", "Other (rare)") for f in feat_names]
importance = pd.Series(rf.named_steps["model"].feature_importances_, index=feat_names).sort_values(ascending=False)

importance.head(12).sort_values().plot(kind="barh", figsize=(8, 5), color="#38A169")
plt.title("Top feature importances (Random Forest)")
plt.xlabel("Importance")
plt.savefig("images/feature_importance.png", dpi=120, bbox_inches="tight")
importance.head(8).round(3)

# %%
lin = fitted["Linear Regression"]
coef_names = [f.split("__", 1)[1].replace("infrequent_sklearn", "Other (rare)") for f in lin.named_steps["prep"].get_feature_names_out()]
coefs = pd.Series(lin.named_steps["model"].regressor_.coef_, index=coef_names)
effect = ((np.exp(coefs) - 1) * 100).rename("Approx. % change in price")
print("Linear model on log(price): effect of a one-standard-deviation increase (numeric)")
print("or of having that category (categorical), all else equal.\n")
effect.reindex(effect.abs().sort_values(ascending=False).index).head(10).round(1)

# %% [markdown]
# ## 7. Predict the price of a new car

# %%
new_cars = pd.DataFrame([
    {"Present_Price": 9.85, "Driven_kms": 30000, "Car_Age": 3, "Owner": 0, "Fuel_Type": "Petrol",
     "Selling_type": "Dealer", "Transmission": "Manual", "Vehicle_Type": "Car", "Brand": "Maruti"},
    {"Present_Price": 30.6, "Driven_kms": 45000, "Car_Age": 5, "Owner": 0, "Fuel_Type": "Diesel",
     "Selling_type": "Dealer", "Transmission": "Automatic", "Vehicle_Type": "Car", "Brand": "Toyota"},
    {"Present_Price": 1.47, "Driven_kms": 15000, "Car_Age": 4, "Owner": 0, "Fuel_Type": "Petrol",
     "Selling_type": "Individual", "Transmission": "Manual", "Vehicle_Type": "Two-wheeler", "Brand": "Royal Enfield"},
])
new_cars["Log_Present_Price"] = np.log(new_cars["Present_Price"])
new_cars["Log_Driven_kms"] = np.log(new_cars["Driven_kms"])
new_cars["Predicted_Selling_Price (lakhs)"] = best.predict(new_cars).round(2)
new_cars.drop(columns=["Log_Present_Price", "Log_Driven_kms"])

# %% [markdown]
# ## 8. Conclusion
# - **Best model: Linear Regression on log(price) with log features.** It has a cross-validated
#   R² of about **0.94**, a test R² of about **0.88**, and an average error of about **₹1 lakh**.
#   It beat Random Forest and Gradient Boosting here because the dataset is small (≈300 rows).
#   Tree models also cannot predict prices above the highest price they saw in training.
#   This hurts them on the few expensive cars in the test set, such as a ₹33-lakh Fortuner.
# - **Showroom price (Present_Price) is by far the strongest driver.** It accounts for about
#   90% of Random Forest importance. A vehicle's original value sets its resale value.
# - **Age is the second driver.** Each extra year lowers the price by a fairly steady
#   percentage. Cars and two-wheelers follow a similar curve: about half the value is gone after 7–8 years.
# - **Diesel and automatic** vehicles sell for more, all else being equal. Kilometres driven,
#   number of owners and seller type matter much less once age and showroom price are known.
# - **Brand goodwill shows up in resale value.** Toyota cars sell for the most on average.
#   Hyundai and Honda cars keep their value better (≈30–34% average depreciation) than
#   Toyota (≈48%), whose listings are older on average.
#
# **Real-world use:** a model like this can suggest a fair listing price for sellers, flag
# overpriced listings for buyers, and help dealers value trade-ins quickly.
#
# *Limitations:* small dataset, a few rare categories (CNG, 3 owners), and no data on
# condition, location or variant. More data would improve accuracy.
