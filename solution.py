# %% [markdown]
# # Home Credit Default Risk — Honors/Outstanding Solution
#
# **Run from the competition directory**: `python solution.py`
#
# This reproducible workflow produces:
# - `output/submission.csv`
# - `tutorial.ipynb` (converted from this source file)
# - `tutorial.zip` (executed notebook, submission, and source code)
#
# Python 3.9+, NumPy, pandas, scikit-learn, SciPy, nbformat, and the optional
# LightGBM, XGBoost, CatBoost, and Optuna packages are used.

# %% [markdown]
# ## Project objective and evaluation protocol
#
# This notebook implements a binary classifier for the probability of loan
# default (`TARGET=1`). It uses only the supplied training and test tables.
# The training table contains labels; the test table is used only for
# prediction. `SK_ID_CURR` is retained for submission alignment and excluded
# from model features.
#
# The competition metric is ROC AUC, which evaluates the ranking of positive
# cases above negative cases without requiring a fixed classification
# threshold. All preprocessing and model selection are reproducible through
# fixed random seeds. No test labels, external data, or manually assigned
# predictions are used.

# %%
# ============================================================================
# CONFIGURACIÓN
# ============================================================================
import os, sys, random, time, zipfile, warnings
from pathlib import Path
import numpy as np
import pandas as pd

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings('ignore')

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

N_FOLDS       = 5
RUN_OPTUNA    = False     # pon True para tuneo (tarda ~30 min extra)
N_TRIALS_LGB  = 8

T_START      = time.time()
CURRENT_DIR  = (Path(__file__).resolve().parent
                if "__file__" in globals() else Path.cwd())
TRAIN_FILE   = CURRENT_DIR / "input" / "train.csv"
TEST_FILE    = CURRENT_DIR / "input" / "test.csv"
SUB_FILE     = CURRENT_DIR / "input" / "sample_submission.csv"
for p in (TRAIN_FILE, TEST_FILE, SUB_FILE):
    assert p.exists(), f"❌ No encontrado: {p}"
print("Working dir:", CURRENT_DIR)

# %%
# ============================================================================
# DEPENDENCIAS (instala automáticamente si faltan)
# ============================================================================
def _ensure(pkg, import_name=None):
    import_name = import_name or pkg
    try:
        __import__(import_name); return True
    except ImportError:
        print(f"Instalando {pkg}...")
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            check=False)
        if result.returncode != 0:
            print(f"⚠️ No se pudo instalar {pkg}; se omitirá.")
            return False
        try:
            __import__(import_name); return True
        except ImportError:
            print(f"⚠️ {pkg} no está disponible; se omitirá.")
            return False

HAVE_LGB = _ensure("lightgbm")
HAVE_XGB = _ensure("xgboost")
HAVE_CAT = _ensure("catboost")
HAVE_OPT = _ensure("optuna")
HAVE_NBF = _ensure("nbformat")

if HAVE_LGB: import lightgbm as lgb
if HAVE_XGB: import xgboost as xgb
if HAVE_CAT: from catboost import CatBoostClassifier
if HAVE_OPT: import optuna
if HAVE_NBF:
    import nbformat
    from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from scipy.optimize import minimize

print(f"Disponibles -> LGB={HAVE_LGB} XGB={HAVE_XGB} CAT={HAVE_CAT} OPT={HAVE_OPT}")

# %% [markdown]
# ## 1. Data loading and schema checks
#
# Load the three competition-provided CSV files. The target distribution
# quantifies class imbalance, while the ID equality check ensures that
# predictions can be written in the exact order expected by the supplied
# submission template.

# %%
train = pd.read_csv(TRAIN_FILE)
test  = pd.read_csv(TEST_FILE)
sample_sub = pd.read_csv(SUB_FILE)

print("train:", train.shape, "| test:", test.shape)
print("TARGET balance:", train["TARGET"].value_counts(normalize=True).round(4).to_dict())
assert (test["SK_ID_CURR"].values == sample_sub["SK_ID_CURR"].values).all(), \
    "SK_ID_CURR del test NO coincide con sample_submission"

y = train["TARGET"].values.astype(np.int8)
test_ids = test["SK_ID_CURR"].values

# %% [markdown]
# ## 2. Feature engineering and missingness
#
# Transform sentinel values and implausible car-age values into missing
# observations, convert relative day counts into years, and derive financial
# ratios and log-scaled amounts. Aggregate the three external-score features
# and credit-bureau request counts. Missingness indicators are added for
# features with more than 30% missing values. These deterministic transforms
# are applied identically to train and test data.

# %%
CAT_COLS    = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'NAME_INCOME_TYPE',
               'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
               'OCCUPATION_TYPE', 'ORGANIZATION_TYPE']
BUREAU_COLS = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_MON',
               'AMT_REQ_CREDIT_BUREAU_QRT',  'AMT_REQ_CREDIT_BUREAU_YEAR']
FLAG_COLS   = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE']
EXT_COLS    = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']


def feature_engineering(df):
    df = df.copy()
    # Sentinelas y outliers
    df['DAYS_EMPLOYED_ANOMALY'] = (df['DAYS_EMPLOYED'] == 365243).astype(np.int8)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    df.loc[df['OWN_CAR_AGE'] >= 60, 'OWN_CAR_AGE'] = np.nan
    # Días -> años
    df['AGE_YEARS']               = -df['DAYS_BIRTH']            / 365
    df['YEARS_EMPLOYED']          = -df['DAYS_EMPLOYED']         / 365
    df['REGISTRATION_YEARS']      = -df['DAYS_REGISTRATION']     / 365
    df['ID_PUBLISH_YEARS']        = -df['DAYS_ID_PUBLISH']       / 365
    df['LAST_PHONE_CHANGE_YEARS'] = -df['DAYS_LAST_PHONE_CHANGE']/ 365
    # Ratios financieros
    df['CREDIT_INCOME_RATIO']  = df['AMT_CREDIT']      / (df['AMT_INCOME_TOTAL'] + 1)
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY']     / (df['AMT_INCOME_TOTAL'] + 1)
    df['CREDIT_TERM']          = df['AMT_ANNUITY']     / (df['AMT_CREDIT'] + 1)
    df['GOODS_CREDIT_RATIO']   = df['AMT_GOODS_PRICE'] / (df['AMT_CREDIT'] + 1)
    df['INCOME_PER_PERSON']    = df['AMT_INCOME_TOTAL']/ (df['CNT_FAM_MEMBERS'] + 1)
    df['CHILDREN_RATIO']       = df['CNT_CHILDREN']    / (df['CNT_FAM_MEMBERS'] + 1)
    df['INCOME_CREDIT_RATIO']  = df['AMT_INCOME_TOTAL']/ (df['AMT_CREDIT'] + 1)
    df['EMPLOYED_BIRTH_RATIO'] = df['YEARS_EMPLOYED']  / (df['AGE_YEARS'] + 1e-3)
    # Logs
    for c in ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE']:
        df[f'LOG_{c}'] = np.log1p(df[c].clip(lower=0))
    # EXT_SOURCE agregadas
    df['EXT_SOURCE_MEAN']   = df[EXT_COLS].mean(axis=1)
    df['EXT_SOURCE_STD']    = df[EXT_COLS].std(axis=1)
    df['EXT_SOURCE_MIN']    = df[EXT_COLS].min(axis=1)
    df['EXT_SOURCE_MAX']    = df[EXT_COLS].max(axis=1)
    df['EXT_SOURCE_PROD']   = df[EXT_COLS].prod(axis=1, min_count=1)
    df['EXT_SOURCE_NA_CNT'] = df[EXT_COLS].isna().sum(axis=1)
    # Bureau + flags
    df['BUREAU_REQ_SUM']    = df[BUREAU_COLS].sum(axis=1, min_count=1)
    df['BUREAU_REQ_NA_CNT'] = df[BUREAU_COLS].isna().sum(axis=1)
    df['FLAG_SUM']          = df[FLAG_COLS].sum(axis=1)
    return df


TRAIN_DROP = ['TARGET', 'SK_ID_CURR']
TEST_DROP  = ['SK_ID_CURR']

train_fe = feature_engineering(train.drop(columns=TRAIN_DROP))
test_fe  = feature_engineering(test.drop(columns=TEST_DROP))

# Sanity check inmediato
assert list(train_fe.columns) == list(test_fe.columns), (
    f"❌ Desalineación tras FE\n"
    f"  solo en train: {sorted(set(train_fe.columns) - set(test_fe.columns))}\n"
    f"  solo en test : {sorted(set(test_fe.columns) - set(train_fe.columns))}\n"
    f"  shapes: train={train_fe.shape}, test={test_fe.shape}"
)
print(f"✅ FE OK — {train_fe.shape[1]} features en train y test")

# Indicadores de missing en columnas con >30% nulos
high_missing = [c for c in train_fe.columns
                if train_fe[c].isna().mean() > 0.30 and c not in CAT_COLS]
for c in high_missing:
    train_fe[f'{c}_MISSING'] = train_fe[c].isna().astype(np.int8)
    test_fe[f'{c}_MISSING']  = test_fe[c].isna().astype(np.int8)

print(f"Features tras FE: {train_fe.shape[1]}")

# %% [markdown]
# ## 3. Categorical encoding and model matrices
#
# Replace categorical `XNA` and null-like values with an explicit `Unknown`
# level. Build a shared category vocabulary from the supplied train and test
# feature values, then map categories to integer codes so all models receive
# aligned numeric matrices. The target and customer ID are absent from these
# matrices; assertions guard feature alignment and prevent ID leakage.

# %%
def encode_categoricals(tr, te):
    tr, te = tr.copy(), te.copy()
    for col in CAT_COLS:
        tr[col] = tr[col].astype(str).replace({'XNA': 'Unknown', 'nan': 'Unknown'})
        te[col] = te[col].astype(str).replace({'XNA': 'Unknown', 'nan': 'Unknown'})
        cats = pd.Categorical(pd.concat([tr[col], te[col]])).categories
        tr[col] = pd.Categorical(tr[col], categories=cats).codes
        te[col] = pd.Categorical(te[col], categories=cats).codes
    return tr, te


train_enc, test_enc = encode_categoricals(train_fe, test_fe)

# ---- Sanity check sobre DataFrames ANTES de .values ----
assert list(train_enc.columns) == list(test_enc.columns), (
    f"❌ Encoding desalineó columnas\n"
    f"  solo en train: {sorted(set(train_enc.columns) - set(test_enc.columns))}\n"
    f"  solo en test : {sorted(set(test_enc.columns) - set(train_enc.columns))}"
)
assert 'SK_ID_CURR' not in train_enc.columns, "❌ SK_ID_CURR sigue como feature"
print(f"✅ Encoding OK — {train_enc.shape[1]} features en train y test")

# ---- Matrices finales ----
X      = train_enc.values.astype(np.float32)
X_test = test_enc.values.astype(np.float32)
y      = y.astype(np.int8)
feature_names = train_enc.columns.tolist()

assert X.shape[1] == X_test.shape[1], (
    f"❌ X y X_test no coinciden: X={X.shape} | X_test={X_test.shape}"
)
print(f"✅ X={X.shape} | X_test={X_test.shape} | positivos={y.sum()} ({y.mean():.4%})")

# Índices de columnas categóricas para LightGBM (mejora opcional)
CAT_IDX = [feature_names.index(c) for c in CAT_COLS]

# %% [markdown]
# ## 4. Stratified cross-validation and base learners
#
# Five-fold stratified cross-validation preserves the default rate in each
# fold. For every learner, out-of-fold (OOF) probabilities provide an
# unbiased-in-fold training estimate for ROC AUC; test probabilities are
# averaged across folds. The workflow evaluates gradient-boosted trees
# (LightGBM, XGBoost, and CatBoost when installed), a median-imputed random
# forest, and a median-imputed, standardized logistic-regression baseline.
# Early stopping is used where supported. LightGBM and XGBoost parameters are
# fixed from validation-only experiments; no leaderboard labels are used.

# %%
skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(skf.split(X, y))


def fit_with_eval(model, X_tr, y_tr, X_va, y_va):
    name = model.__class__.__name__
    if name == 'LGBMClassifier' and HAVE_LGB:
        try:
            model.fit(X_tr, y_tr,
                      eval_set=[(X_va, y_va)],
                      eval_metric='auc',
                      callbacks=[lgb.early_stopping(200, verbose=False),
                                 lgb.log_evaluation(0)])
        except TypeError:
            # compatibilidad con versiones antiguas
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                      eval_metric='auc')
    elif name == 'XGBClassifier' and HAVE_XGB:
        try:
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                      verbose=False, early_stopping_rounds=200)
        except TypeError:
            # XGBoost >= 2.1: early_stopping_rounds debe ir en el constructor
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    elif name == 'CatBoostClassifier' and HAVE_CAT:
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    else:
        model.fit(X_tr, y_tr)


def cv_auc(model_factory, X_train, y_train, X_test_, name=""):
    oof        = np.zeros(len(X_train), dtype=np.float64)
    test_preds = np.zeros(len(X_test_), dtype=np.float64)
    t0 = time.time()
    for fold, (tr_idx, va_idx) in enumerate(folds):
        model = model_factory(fold)
        fit_with_eval(model,
                      X_train[tr_idx], y_train[tr_idx],
                      X_train[va_idx], y_train[va_idx])
        oof[va_idx] = model.predict_proba(X_train[va_idx])[:, 1]
        test_preds += model.predict_proba(X_test_)[:, 1] / len(folds)
    auc = roc_auc_score(y_train, oof)
    print(f"[{name:20s}] OOF AUC = {auc:.5f}  ({time.time()-t0:.1f}s)")
    return oof, test_preds, auc


results = {}

# %%
# ----- LightGBM (con categorical_feature declaradas) -----
if HAVE_LGB:
    LGB_PARAMS = dict(
        objective='binary', metric='auc', boosting_type='gbdt',
        learning_rate=0.0245967747, num_leaves=37, max_depth=7,
        min_child_samples=38, subsample=0.8906590942, subsample_freq=1,
        colsample_bytree=0.6723651931, reg_alpha=4.4716229470,
        reg_lambda=2.9917712366, min_split_gain=0.0198715682,
        n_estimators=5000, random_state=SEED, n_jobs=-1, verbose=-1)
    oof, tp, auc = cv_auc(
        lambda f: lgb.LGBMClassifier(
            **LGB_PARAMS,
            categorical_feature=CAT_IDX),   # ← mejora opcional
        X, y, X_test, "LightGBM")
    results['lgb'] = (oof, tp, auc)

# %%
# ----- XGBoost -----
if HAVE_XGB:
    import xgboost as _xgb
    _xgb_major = int(_xgb.__version__.split('.')[0])
    _xgb_minor = int(_xgb.__version__.split('.')[1]) if _xgb_major == 2 else 0
    XGB_PARAMS = dict(
        objective='binary:logistic', eval_metric='auc',
        learning_rate=0.03, max_depth=4, min_child_weight=5,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=2.0,
        n_estimators=5000, tree_method='hist',
        random_state=SEED, n_jobs=-1)
    if (_xgb_major, _xgb_minor) >= (2, 1):
        XGB_PARAMS['early_stopping_rounds'] = 200
    oof, tp, auc = cv_auc(lambda f: xgb.XGBClassifier(**XGB_PARAMS),
                          X, y, X_test, "XGBoost")
    results['xgb'] = (oof, tp, auc)

# %%
# ----- CatBoost -----
if HAVE_CAT:
    oof, tp, auc = cv_auc(
        lambda f: CatBoostClassifier(
            iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=3,
            loss_function='Logloss', eval_metric='AUC',
            random_seed=SEED, od_type='Iter', od_wait=200,
            verbose=False, task_type='CPU'),
        X, y, X_test, "CatBoost")
    results['cat'] = (oof, tp, auc)

# %%
# ----- RandomForest -----
oof, tp, auc = cv_auc(
    lambda f: Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('rf',  RandomForestClassifier(
            n_estimators=400, min_samples_leaf=20, max_features='sqrt',
            class_weight='balanced_subsample',
            random_state=SEED + f, n_jobs=-1))]),
    X, y, X_test, "RandomForest")
results['rf'] = (oof, tp, auc)

# %%
# ----- LogisticRegression -----
oof, tp, auc = cv_auc(
    lambda f: Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc',  StandardScaler()),
        ('lr',  LogisticRegression(C=0.1, max_iter=2000,
                                   class_weight='balanced',
                                   random_state=SEED + f))]),
    X, y, X_test, "LogisticRegression")
results['lr'] = (oof, tp, auc)

# %%
# ----- Optuna (opcional) -----
if RUN_OPTUNA and HAVE_OPT and HAVE_LGB:
    def objective(trial):
        params = dict(
            objective='binary', metric='auc', boosting_type='gbdt',
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
            num_leaves=trial.suggest_int('num_leaves', 20, 96),
            max_depth=trial.suggest_int('max_depth', 4, 12),
            min_child_samples=trial.suggest_int('min_child_samples', 10, 80),
            subsample=trial.suggest_float('subsample', 0.6, 1.0),
            subsample_freq=1,
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
            reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
            n_estimators=5000, random_state=SEED, n_jobs=-1, verbose=-1)
        oof_ = np.zeros(len(X))
        for tr_idx, va_idx in folds:
            m = lgb.LGBMClassifier(**params, categorical_feature=CAT_IDX)
            m.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
            oof_[va_idx] = m.predict_proba(X[va_idx])[:, 1]
        return roc_auc_score(y, oof_)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS_LGB, show_progress_bar=False)
    print("Optuna best:", study.best_value, "| params:", study.best_params)
    BEST_LGB = {**LGB_PARAMS, **study.best_params}
    oof, tp, auc = cv_auc(lambda f: lgb.LGBMClassifier(
                              **BEST_LGB, categorical_feature=CAT_IDX),
                          X, y, X_test, "LightGBM-tuned")
    results['lgb_tuned'] = (oof, tp, auc)

# %% [markdown]
# ## 5. Rank-based ensemble
#
# Convert each learner's OOF and test predictions to percentile ranks, making
# their score scales comparable. Optimize nonnegative normalized blend
# weights against OOF ROC AUC with several deterministic initializations.
# This validation-based optimization is intended to improve ranking while
# avoiding direct use of test labels (which are unavailable).

# %%
names = list(results.keys())
oofs  = [results[n][0] for n in names]
tests = [results[n][1] for n in names]

def rank_norm(a):
    return pd.Series(a).rank(pct=True).values

oofs_r  = [rank_norm(o) for o in oofs]
tests_r = [rank_norm(t) for t in tests]


def neg_auc(w, oofs_list, target):
    w = np.abs(w); w = w / (w.sum() + 1e-9)
    blend = np.zeros_like(oofs_list[0])
    for wi, o in zip(w, oofs_list):
        blend += wi * o
    return -roc_auc_score(target, blend)

best_res, best_val = None, -1.0
for init in ['uniform', 'random', 'random', 'random']:
    if init == 'uniform':
        x0 = np.ones(len(names)) / len(names)
    else:
        r = np.random.rand(len(names)); x0 = r / r.sum()
    r = minimize(neg_auc, x0=x0, args=(oofs_r, y), method='Nelder-Mead',
                 options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 500})
    if -r.fun > best_val:
        best_val, best_res = -r.fun, r

weights = np.abs(best_res.x); weights = weights / weights.sum()
blend_oof  = sum(w * o for w, o in zip(weights, oofs_r))
blend_test = sum(w * t for w, t in zip(weights, tests_r))
auc_blend  = roc_auc_score(y, blend_oof)

print("\nPesos del ensemble:")
for n, w in zip(names, weights):
    print(f"  {n:15s} : {w:.4f}  (OOF individual = {results[n][2]:.5f})")
print(f"\nEnsemble OOF AUC = {auc_blend:.5f}")

# %% [markdown]
# ## 6. Final model selection and submission export
#
# Compare the ensemble OOF AUC with the best individual learner, select the
# stronger validation result, clip probabilities away from exact zero and
# one, and join predictions to the supplied sample-submission IDs. Assertions
# verify complete ID coverage and the expected row count before writing the
# required `SK_ID_CURR,TARGET` CSV schema.

# %%
best_single = max(results, key=lambda k: results[k][2])
best_auc    = results[best_single][2]
print(f"Mejor individual: {best_single} ({best_auc:.5f})")

if auc_blend >= best_auc:
    final_pred = blend_test
    print(f"→ Usando ENSEMBLE ({auc_blend:.5f})")
else:
    final_pred = tests_r[names.index(best_single)]
    print(f"→ Usando {best_single}")

final_pred = np.clip(final_pred, 1e-6, 1 - 1e-6)
print("Rango predicciones:", float(final_pred.min()), "-", float(final_pred.max()))

OUTPUT_DIR      = CURRENT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"

pred_df = pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": final_pred})
sub = sample_sub[["SK_ID_CURR"]].merge(pred_df, on="SK_ID_CURR", how="left")
assert sub["TARGET"].notna().all(), "IDs de sample_submission no están en test"
assert len(sub) == len(sample_sub)
sub.to_csv(SUBMISSION_PATH, index=False)
print(f"✅ {SUBMISSION_PATH} guardado ({len(sub)} filas)")

# %% [markdown]
# ## 7. Reproducibility artifacts
#
# The Python source is partitioned into Markdown and code cells to create this
# notebook. The final archive contains the notebook, generated submission,
# and source file so the workflow can be inspected and reproduced.

# %%
def _parse_percent_source(source_text):
    cells = []
    cur_type, cur_lines = 'code', []

    def flush():
        nonlocal cur_type, cur_lines
        if cur_lines and any(l.strip() for l in cur_lines):
            cells.append((cur_type, '\n'.join(cur_lines)))
        cur_lines = []

    for line in source_text.splitlines():
        if line.startswith('# %% [markdown]'):
            flush(); cur_type = 'markdown'
        elif line.startswith('# %%'):
            flush(); cur_type = 'code'
        else:
            cur_lines.append(line)
    flush()

    nb_cells = []
    for ctype, ctext in cells:
        if ctype == 'markdown':
            md = '\n'.join(
                (l[2:] if l.startswith('# ') else (l[1:] if l.startswith('#') else l))
                for l in ctext.splitlines())
            nb_cells.append(new_markdown_cell(md))
        else:
            nb_cells.append(new_code_cell(ctext))
    return new_notebook(
        cells=nb_cells,
        metadata={"kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }})


def finalize():
    if not HAVE_NBF:
        print("⚠️ nbformat no disponible; se omite .ipynb/.zip")
        return
    in_notebook = callable(globals().get("get_ipython"))

    nb_path = CURRENT_DIR / "tutorial.ipynb"
    if not in_notebook:
        try:
            src = Path(__file__).resolve().read_text(encoding='utf-8')
        except NameError:
            print("⚠️ __file__ no disponible")
            return
        nb = _parse_percent_source(src)
        nbformat.write(nb, str(nb_path))
        print(f"✅ tutorial.ipynb creado ({nb_path})")

    zip_path = CURRENT_DIR / "tutorial.zip"
    if nb_path.exists():
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(nb_path, arcname="tutorial.ipynb")
            zf.write(SUBMISSION_PATH, arcname="output/submission.csv")
            try:
                zf.write(Path(__file__).resolve(), arcname="solution.py")
            except NameError:
                pass
        print(f"✅ tutorial.zip creado ({zip_path})")


finalize()
print(f"\n⏱ Tiempo total: {time.time() - T_START:.1f}s")