# =========================
# Two-Model Architecture for CI/CD Failure Prediction (FIXED)
# =========================

import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

print("=" * 70)
print("TWO-MODEL CI/CD PREDICTION SYSTEM")
print("=" * 70)

# =========================
# MODEL 1: Line-Level Defect Prediction
# =========================
print("\n" + "=" * 70)
print("MODEL 1: LINE-LEVEL DEFECT PREDICTION")
print("=" * 70)

print("\nLoading train-lines data...")
line_dfs = []
file_count = 0

for root, dirs, files in os.walk('data/train-lines'):
    for filename in files:
        if filename.endswith('.csv'):
            filepath = os.path.join(root, filename)
            try:
                # Try reading with $ delimiter
                temp_df = pd.read_csv(filepath, delimiter='$', quotechar='"')

                # Check if parsing worked
                if len(temp_df.columns) < 3:
                    # Try without delimiter specification
                    temp_df = pd.read_csv(filepath)

                # Clean column names
                temp_df.columns = temp_df.columns.str.strip().str.strip('"')

                # Check if this is actually data
                if 'class_value' in temp_df.columns or 'build_status' in temp_df.columns:
                    line_dfs.append(temp_df)
                    file_count += 1
                    if file_count % 10 == 0:
                        print(f"  Loaded {file_count} files...")
                else:
                    print(f"  Skipping {filename} - no target column found")

            except Exception as e:
                print(f"  Error loading {filename}: {e}")
                continue

if not line_dfs:
    print("\n⚠️  WARNING: Could not load train-lines data!")
    print("Skipping Model 1 and proceeding with Model 2 only.\n")
    model_1 = None
else:
    df_lines = pd.concat(line_dfs, ignore_index=True)
    print(f"\n✓ Total lines: {len(df_lines):,}")
    print(f"✓ Columns: {df_lines.columns.tolist()}")

    # Inspect the data
    print("\nFirst few rows:")
    print(df_lines.head(3))

    print("\nData types:")
    print(df_lines.dtypes)

    # Target variable - check what we actually have
    print("\n" + "=" * 50)
    print("IDENTIFYING TARGET VARIABLE")
    print("=" * 50)

    target_col = None
    if 'class_value' in df_lines.columns:
        target_col = 'class_value'
    elif 'build_status' in df_lines.columns:
        target_col = 'build_status'

    if target_col:
        print(f"Using target column: {target_col}")
        print(f"Unique values: {df_lines[target_col].unique()[:20]}")
        print(f"Value counts:\n{df_lines[target_col].value_counts(dropna=False).head(10)}")
        print(f"Data type: {df_lines[target_col].dtype}")

        # Convert target to binary
        try:
            # Remove rows where target column equals the column name (header leak)
            df_lines = df_lines[df_lines[target_col] != target_col].copy()

            # Convert to numeric, coercing errors
            df_lines['is_buggy'] = pd.to_numeric(df_lines[target_col], errors='coerce')

            # Fill NaN with 0 (assume non-buggy)
            df_lines['is_buggy'] = df_lines['is_buggy'].fillna(0).astype(int)

            # Remove any remaining invalid rows
            df_lines = df_lines[df_lines['is_buggy'].isin([0, 1])].copy()

            print(f"\n✓ Target created successfully")
            print(f"  Buggy lines: {df_lines['is_buggy'].sum():,} ({df_lines['is_buggy'].mean():.2%})")
            print(f"  Clean lines: {(df_lines['is_buggy'] == 0).sum():,}")

        except Exception as e:
            print(f"\n✗ Error creating target: {e}")
            print("Skipping Model 1")
            model_1 = None
            df_lines = None
    else:
        print("✗ No target column found!")
        model_1 = None
        df_lines = None

    # Feature engineering for lines
    if df_lines is not None and len(df_lines) > 0:
        print("\n" + "=" * 50)
        print("FEATURE ENGINEERING")
        print("=" * 50)

        # Basic features
        if 'contents' in df_lines.columns:
            print("Creating content-based features...")
            df_lines['line_length'] = df_lines['contents'].astype(str).str.len()
            df_lines['has_comment'] = df_lines['contents'].astype(str).str.contains('//', regex=False).astype(int)
            df_lines['complexity_indicators'] = df_lines['contents'].astype(str).str.count(r'if|for|while|switch')
            df_lines['has_semicolon'] = df_lines['contents'].astype(str).str.contains(';', regex=False).astype(int)

        if 'path' in df_lines.columns:
            print("Creating path-based features...")
            df_lines['file_extension'] = df_lines['path'].astype(str).str.extract(r'\.([^.]+)$')[0]
            df_lines['is_test_file'] = df_lines['path'].astype(str).str.contains('test', case=False, na=False).astype(
                int)
            df_lines['path_depth'] = df_lines['path'].astype(str).str.count('/')

        if 'class_name' in df_lines.columns:
            df_lines['class_name_length'] = df_lines['class_name'].astype(str).str.len()

        # Select features for line model
        line_features = []
        potential_features = [
            'line_length', 'has_comment', 'complexity_indicators',
            'is_test_file', 'path_depth', 'has_semicolon', 'class_name_length'
        ]

        for feat in potential_features:
            if feat in df_lines.columns and df_lines[feat].dtype in ['int64', 'float64']:
                line_features.append(feat)

        if len(line_features) < 2:
            print(f"\n⚠️  Not enough valid features ({len(line_features)}). Skipping Model 1.")
            model_1 = None
        else:
            print(f"\n✓ Using {len(line_features)} features: {line_features}")

            # Prepare data
            X_lines = df_lines[line_features].fillna(0)
            y_lines = df_lines['is_buggy']

            print(f"\nClass distribution:")
            print(f"  Class 0 (clean): {(y_lines == 0).sum():,} ({(y_lines == 0).mean():.1%})")
            print(f"  Class 1 (buggy): {(y_lines == 1).sum():,} ({(y_lines == 1).mean():.1%})")

            # Check if we have enough samples
            if len(X_lines) < 100 or y_lines.sum() < 10:
                print("\n⚠️  Insufficient data for training. Skipping Model 1.")
                model_1 = None
            else:
                # Split
                X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
                    X_lines, y_lines, test_size=0.2, random_state=42,
                    stratify=y_lines if y_lines.nunique() > 1 else None
                )

                print(f"\nTrain: {len(X_train_l):,} | Test: {len(X_test_l):,}")

                # Train Model 1
                print("\nTraining line-level defect model...")
                model_1 = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', XGBClassifier(
                        n_estimators=200,
                        max_depth=5,
                        learning_rate=0.1,
                        random_state=42,
                        eval_metric='logloss'
                    ))
                ])

                model_1.fit(X_train_l, y_train_l)

                # Evaluate Model 1
                y_pred_l = model_1.predict(X_test_l)
                y_prob_l = model_1.predict_proba(X_test_l)[:, 1]

                print("\n" + "=" * 50)
                print("MODEL 1 RESULTS")
                print("=" * 50)
                print(classification_report(y_test_l, y_pred_l, digits=4))
                print(f"ROC-AUC: {roc_auc_score(y_test_l, y_prob_l):.4f}")

                # Save Model 1
                os.makedirs('processed_data', exist_ok=True)
                joblib.dump(model_1, 'processed_data/line_defect_model.pkl')
                print("\n✓ Saved Model 1 to processed_data/line_defect_model.pkl")
    else:
        model_1 = None

# =========================
# MODEL 2: Build-Level Prediction (WITHOUT prev_build_failed)
# =========================
print("\n" + "=" * 70)
print("MODEL 2: BUILD-LEVEL FAILURE PREDICTION")
print("=" * 70)

print("\nLoading TSM data...")
metric_dfs = []
for root, dirs, files in os.walk('data/tsm'):
    for filename in files:
        if filename.endswith('.csv'):
            filepath = os.path.join(root, filename)
            try:
                temp_df = pd.read_csv(filepath, delimiter='$', quotechar='"')
                temp_df.columns = temp_df.columns.str.strip().str.strip('"')
                metric_dfs.append(temp_df)
            except:
                pass

df_builds = pd.concat(metric_dfs, ignore_index=True)
print(f"✓ Total builds: {len(df_builds):,}")

# Target
df_builds['failed'] = df_builds['tr_status'].astype(int)
print(f"Failure rate: {df_builds['failed'].mean():.2%}")

# Timestamps
df_builds['ts'] = pd.to_datetime(df_builds['gh_build_started_at'], errors='coerce')
df_builds = df_builds.dropna(subset=['ts']).sort_values('ts').reset_index(drop=True)
print(f"After timestamp cleaning: {len(df_builds):,} rows")

# Feature engineering (NO prev_build_failed!)
print("\nEngineering build features...")

df_builds['hour'] = df_builds['ts'].dt.hour
df_builds['day_of_week'] = df_builds['ts'].dt.dayofweek
df_builds['month'] = df_builds['ts'].dt.month
df_builds['is_weekend'] = df_builds['day_of_week'].isin([5, 6]).astype(int)
df_builds['is_night'] = df_builds['hour'].isin(range(0, 6)).astype(int)

# Interaction features
df_builds['files_per_commit'] = df_builds['gh_diff_files_modified'] / (df_builds['gh_num_commits_in_push'] + 1)
df_builds['churn_per_file'] = df_builds['git_diff_src_churn'] / (df_builds['gh_diff_src_files'] + 1)
df_builds['team_commits_ratio'] = df_builds['gh_num_commits_in_push'] / (df_builds['gh_team_size'] + 1)

# Complexity indicators
df_builds['large_change'] = (df_builds['git_diff_src_churn'] > df_builds['git_diff_src_churn'].quantile(0.9)).astype(
    int)
df_builds['many_files_touched'] = (df_builds['gh_diff_files_modified'] > 10).astype(int)

# Select features (NO prev_build_failed!)
build_features = [
    'gh_num_commits_in_push',
    'git_prev_commit_resolution_status',
    'gh_team_size',
    'git_num_all_built_commits',
    'gh_num_commit_comments',
    'git_diff_src_churn',
    'gh_diff_files_added',
    'gh_diff_files_deleted',
    'gh_diff_files_modified',
    'gh_diff_src_files',
    'gh_diff_doc_files',
    'gh_diff_other_files',
    'gh_num_commits_on_files_touched',
    'gh_sloc',
    'hour',
    'day_of_week',
    'month',
    'is_weekend',
    'is_night',
    'files_per_commit',
    'churn_per_file',
    'team_commits_ratio',
    'large_change',
    'many_files_touched'
]

build_features = [c for c in build_features if c in df_builds.columns]
print(f"✓ Using {len(build_features)} features")

# Prepare data
X_builds = df_builds[build_features].fillna(0)
y_builds = df_builds['failed']

# Chronological split
cutoff = df_builds['ts'].quantile(0.75)
train_mask = df_builds['ts'] <= cutoff
test_mask = df_builds['ts'] > cutoff

X_train_b = X_builds[train_mask]
X_test_b = X_builds[test_mask]
y_train_b = y_builds[train_mask]
y_test_b = y_builds[test_mask]

print(f"\nTrain: {len(X_train_b):,} rows ({y_train_b.mean():.2%} fail rate)")
print(f"Test:  {len(X_test_b):,} rows ({y_test_b.mean():.2%} fail rate)")

# Train Model 2
print("\nTraining build-level failure model...")
model_2 = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=len(y_train_b[y_train_b == 0]) / len(y_train_b[y_train_b == 1])
    ))
])

model_2.fit(X_train_b, y_train_b)

# Evaluate Model 2
y_pred_b = model_2.predict(X_test_b)
y_prob_b = model_2.predict_proba(X_test_b)[:, 1]

print("\n" + "=" * 70)
print("MODEL 2 RESULTS")
print("=" * 70)
print(classification_report(y_test_b, y_pred_b, digits=4))
print(f"ROC-AUC: {roc_auc_score(y_test_b, y_prob_b):.4f}")

# Feature importance
importance_df = pd.DataFrame({
    'feature': build_features,
    'importance': model_2.named_steps['classifier'].feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "=" * 70)
print("TOP 10 MOST IMPORTANT FEATURES")
print("=" * 70)
print(importance_df.head(10).to_string(index=False))

# Save Model 2
os.makedirs('processed_data', exist_ok=True)
joblib.dump(model_2, 'processed_data/build_failure_model_clean.pkl')
importance_df.to_csv('processed_data/feature_importance_clean.csv', index=False)
print("\n✓ Saved Model 2 to processed_data/build_failure_model_clean.pkl")
print("✓ Saved feature importance to processed_data/feature_importance_clean.csv")

# =========================
# SUMMARY
# =========================
print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

if model_1 is not None:
    print("\n✓ Model 1 (Line-level): TRAINED")
    print("  → Use to predict which lines of code are buggy")
else:
    print("\n✗ Model 1 (Line-level): SKIPPED")
    print("  → train-lines data unavailable or insufficient")

print("\n✓ Model 2 (Build-level): TRAINED")
print("  → Use to predict build failures from build metrics")
print("  → Does NOT use previous build history")
print(f"  → ROC-AUC: {roc_auc_score(y_test_b, y_prob_b):.4f}")

print("\nFiles saved in: processed_data/")
print("\n✓ Done!")