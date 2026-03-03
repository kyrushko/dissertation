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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
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
            if len(df_lines) > 100000:
                print(f"\n⚠️  Large dataset detected ({len(df_lines):,} lines)")
                print("Performing stratified sampling to keep class balance...")

                df_lines = df_lines.groupby('is_buggy', group_keys=False).apply(
                    lambda x: x.sample(min(len(x), 50000), random_state=42)
                ).reset_index(drop=True)

                # print(f"✓ Sampled to {len(df_lines):,} lines")
                # print(f"  Buggy: {df_lines['is_buggy'].sum():,} ({df_lines['is_buggy'].mean():.2%})")
                # print(f"  Clean: {(df_lines['is_buggy'] == 0).sum():,}")

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

            # Clean content first
            df_lines['content_clean'] = df_lines['contents'].astype(str).str.strip()
            df_lines['line_length'] = df_lines['content_clean'].str.len()

            # === IDENTIFY COMMENTS FIRST ===
            df_lines['is_comment'] = df_lines['content_clean'].str.contains(
                r'^\s*(?://|/\*|\*|#)', regex=True, na=False
            ).astype(int)

            # === VADER ONLY ON COMMENTS (MAXIMUM MEMORY EFFICIENCY) ===
            print("\nPerforming sentiment analysis on COMMENTS ONLY...")
            analyzer = SentimentIntensityAnalyzer()

            comment_mask = df_lines['is_comment'] == 1
            num_comments = comment_mask.sum()

            print(
                f"  Found {num_comments:,} comment lines out of {len(df_lines):,} total ({num_comments / len(df_lines):.1%})")

            # Pre-allocate NumPy arrays (most memory efficient)
            sentiment_neg = np.zeros(len(df_lines), dtype=np.float32)
            sentiment_pos = np.zeros(len(df_lines), dtype=np.float32)
            sentiment_neu = np.zeros(len(df_lines), dtype=np.float32)
            sentiment_compound = np.zeros(len(df_lines), dtype=np.float32)

            if num_comments > 0:
                print("  Processing comments (maximum memory efficiency)...")

                # Get comment data
                comment_indices = df_lines[comment_mask].index.tolist()
                comment_texts = df_lines.loc[comment_mask, 'content_clean'].tolist()

                batch_size = 5000
                for i in range(0, len(comment_indices), batch_size):
                    batch_end = min(i + batch_size, len(comment_indices))

                    for j in range(i, batch_end):
                        idx = comment_indices[j]
                        text = comment_texts[j]
                        scores = analyzer.polarity_scores(str(text))

                        sentiment_neg[idx] = scores['neg']
                        sentiment_pos[idx] = scores['pos']
                        sentiment_neu[idx] = scores['neu']
                        sentiment_compound[idx] = scores['compound']

                    print(
                        f"  Processed {batch_end:,} / {num_comments:,} comments ({batch_end / num_comments * 100:.1f}%)")

                print(f"✓ Sentiment analysis complete on {num_comments:,} comments!")

            # Assign arrays to DataFrame
            df_lines['sentiment_neg'] = sentiment_neg
            df_lines['sentiment_pos'] = sentiment_pos
            df_lines['sentiment_neu'] = sentiment_neu
            df_lines['sentiment_compound'] = sentiment_compound

            # Free memory
            del sentiment_neg, sentiment_pos, sentiment_neu, sentiment_compound

            # Show examples
            if num_comments >= 3:
                print("\n--- Sample Comment Sentiments ---")
                comment_samples = df_lines[comment_mask].nlargest(3, 'sentiment_neg')[
                    ['content_clean', 'sentiment_neg', 'sentiment_compound']
                ]
                for idx, row in comment_samples.iterrows():
                    print(f"Neg: {row['sentiment_neg']:.3f} | Compound: {row['sentiment_compound']:.3f}")
                    print(f"  → {row['content_clean'][:80]}\n")
            # === END VADER ===


            # Other comment features
            df_lines['is_todo_comment'] = df_lines['content_clean'].str.contains(
                r'\b(?:TODO|FIXME|HACK|XXX|BUG)\b', case=False, regex=True, na=False
            ).astype(int)

            # Code complexity
            df_lines['control_flow_count'] = df_lines['content_clean'].str.count(
                r'\b(?:if|else|for|while|switch|case|catch|try)\b'
            )

            df_lines['method_call_count'] = df_lines['content_clean'].str.count(r'\w+\(')

            df_lines['has_exception'] = df_lines['content_clean'].str.contains(
                r'\b(?:Exception|Error|throw|throws)\b', regex=True, na=False
            ).astype(int)

            # Git diff markers
            df_lines['is_diff_header'] = df_lines['content_clean'].str.contains(
                r'^(?:\+\+|--|@@)', regex=True, na=False
            ).astype(int)

            # Code smells
            df_lines['nested_depth'] = df_lines['content_clean'].apply(
                lambda x: len(x) - len(x.lstrip()) if isinstance(x, str) else 0
            ) // 4

            df_lines['has_magic_number'] = df_lines['content_clean'].str.contains(
                r'\b\d{2,}\b', regex=True, na=False
            ).astype(int)

            df_lines['has_logging'] = df_lines['content_clean'].str.contains(
                r'\b(?:Log\.|logger|println)\b', regex=True, na=False
            ).astype(int)

            df_lines['has_null_check'] = df_lines['content_clean'].str.contains(
                r'\b(?:null|NULL|isNull|isEmpty)\b', regex=True, na=False
            ).astype(int)

        if 'path' in df_lines.columns:
            print("Creating path-based features...")
            df_lines['file_extension'] = df_lines['path'].astype(str).str.extract(r'\.([^.]+)$')[0]
            df_lines['is_test_file'] = df_lines['path'].astype(str).str.contains(
                r'(?:test|spec|mock)', case=False, na=False
            ).astype(int)
            df_lines['is_config_file'] = df_lines['path'].astype(str).str.contains(
                r'\.(?:yml|yaml|gradle|xml|properties)$', case=False, na=False
            ).astype(int)
            df_lines['path_depth'] = df_lines['path'].astype(str).str.count('/')

        if 'class_name' in df_lines.columns:
            df_lines['class_name_length'] = df_lines['class_name'].astype(str).str.len()

        # Select features for line model
        # Select features for line model
        line_features = []
        potential_features = [
            # Size metrics
            'line_length',
            'nested_depth',

            # VADER sentiment features
            'sentiment_neg',
            'sentiment_pos',
            'sentiment_neu',
            'sentiment_compound',

            # Comment indicators
            'is_comment',
            'is_todo_comment',

            # Code complexity
            'control_flow_count',
            'method_call_count',
            'has_exception',

            # Git diff
            'is_diff_header',

            # Code quality
            'has_magic_number',
            'has_logging',
            'has_null_check',

            # File context
            'is_test_file',
            'is_config_file',
            'path_depth',
            'class_name_length'
        ]

        for feat in potential_features:
            if feat in df_lines.columns and df_lines[feat].dtype in ['int64', 'float64', 'float32']:
                line_features.append(feat)

        # Check if we have enough features
        if len(line_features) < 2:
            print(f"\n⚠️  Not enough valid features ({len(line_features)}). Skipping Model 1.")
            model_1 = None
        else:
            print(f"\n✓ Using {len(line_features)} features: {line_features}")

            # DIAGNOSTIC: Check if sentiment features made it in
            sentiment_features = [f for f in line_features if f.startswith('sentiment_')]
            if len(sentiment_features) == 0:
                print("⚠️  WARNING: No sentiment features included! Check dtypes:")
                for sf in ['sentiment_neg', 'sentiment_pos', 'sentiment_neu', 'sentiment_compound']:
                    if sf in df_lines.columns:
                        print(f"  {sf}: {df_lines[sf].dtype}")
            else:
                print(f"✓ Sentiment features included: {sentiment_features}")

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


# previous versioon - potentiall an error here. uncomment if the acc drops
df_builds['failed'] = df_builds['tr_status'].astype(int)
print(f"Failure rate: {df_builds['failed'].mean():.2%}")

# =========================
# LINK LINES TO BUILDS (Timestamp Matching)
# =========================
print("\n" + "=" * 70)
print("LINKING LINE DATA TO BUILD DATA")
print("=" * 70)

if model_1 is not None and len(df_builds) > 0:

    # Parse timestamps
    print("Parsing timestamps...")
    df_lines['line_timestamp'] = pd.to_datetime(df_lines['date'], errors='coerce')
    df_builds['build_timestamp'] = pd.to_datetime(df_builds['gh_build_started_at'], errors='coerce')

    # Clean data
    df_lines_clean = df_lines.dropna(subset=['line_timestamp']).copy()
    df_builds_clean = df_builds.dropna(subset=['build_timestamp']).copy()

    print(f"Line data: {len(df_lines_clean):,} lines")
    print(f"Build data: {len(df_builds_clean):,} builds")

    # Add build ID
    df_builds_clean['build_id'] = range(len(df_builds_clean))

    # Check timestamp overlap
    line_start, line_end = df_lines_clean['line_timestamp'].min(), df_lines_clean['line_timestamp'].max()
    build_start, build_end = df_builds_clean['build_timestamp'].min(), df_builds_clean['build_timestamp'].max()

    print(f"\nTimestamp ranges:")
    print(f"  Lines:  {line_start} to {line_end}")
    print(f"  Builds: {build_start} to {build_end}")

    if line_end < build_start or build_end < line_start:
        print("\n❌ NO OVERLAP - Cannot link datasets")
        df_lines_with_build = None
    else:
        overlap_start = max(line_start, build_start)
        overlap_end = min(line_end, build_end)
        print(f"  Overlap: {overlap_start} to {overlap_end}")

        # Use merge_asof for timestamp matching
        print("\nMatching lines to builds by timestamp...")

        df_lines_sorted = df_lines_clean.sort_values('line_timestamp').reset_index(drop=True)
        df_builds_sorted = df_builds_clean.sort_values('build_timestamp').reset_index(drop=True)

        # Match each line to the nearest build (within 5 minutes)
        df_lines_with_build = pd.merge_asof(
            df_lines_sorted,
            df_builds_sorted[['build_id', 'build_timestamp', 'tr_status', 'tr_original_commit']],
            left_on='line_timestamp',
            right_on='build_timestamp',
            direction='nearest',  # Find closest build
            tolerance=pd.Timedelta('5 minutes')  # Within 5 minutes
        )

        # Count matches
        matched = df_lines_with_build['build_id'].notna().sum()
        match_pct = matched / len(df_lines_with_build) * 100

        print(f"\n✓ Matched {matched:,} / {len(df_lines_with_build):,} lines ({match_pct:.1f}%)")

        if match_pct > 50:
            # Calculate time differences
            time_diff = (df_lines_with_build['build_timestamp'] - df_lines_with_build[
                'line_timestamp']).dt.total_seconds().abs()
            exact_matches = (time_diff == 0).sum()

            print(f"\nMatching quality:")
            print(f"  Exact timestamp matches: {exact_matches:,} ({exact_matches / matched * 100:.1f}%)")
            print(f"  Average time gap: {time_diff.mean():.1f} seconds")
            print(f"  Max time gap: {time_diff.max():.1f} seconds")

            # Show unique builds matched
            unique_builds = df_lines_with_build['build_id'].nunique()
            print(f"  Unique builds matched: {unique_builds:,} / {len(df_builds_clean):,}")

            # Save linked data
            df_lines_with_build.to_csv('processed_data/lines_with_builds.csv', index=False)
            print("\n✓ Saved linked data to processed_data/lines_with_builds.csv")

        else:
            print(f"\n⚠️  Low match rate ({match_pct:.1f}%) - linking may not be reliable")
            df_lines_with_build = None

else:
    print("\n✗ Skipping linking (Model 1 not trained or no build data)")
    df_lines_with_build = None

# Timestamps
df_builds['ts'] = pd.to_datetime(df_builds['gh_build_started_at'], errors='coerce')
df_builds = df_builds.dropna(subset=['ts']).sort_values('ts').reset_index(drop=True)
print(f"After timestamp cleaning: {len(df_builds):,} rows")

df_builds['build_id'] = range(len(df_builds))  # ✓ Create build_id
print(f"✓ Assigned build_id to {len(df_builds):,} builds")

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
        # scale_pos_weight=len(y_train_b[y_train_b == 0]) / len(y_train_b[y_train_b == 1])
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

# ... Model 2 training ends here ...
print("\n✓ Saved Model 2 to processed_data/build_failure_model_clean.pkl")
print("✓ Saved feature importance to processed_data/feature_importance_clean.csv")


# TRYING ALTERNATIVE MODEL 2 CONFIG
print("\n" + "=" * 50)
print("TRYING ALTERNATIVE MODEL 2 CONFIGURATIONS")
print("=" * 50)

from sklearn.ensemble import RandomForestClassifier

# Try Random Forest
model_2_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=10,
        random_state=42,
        class_weight='balanced'
    ))
])

model_2_rf.fit(X_train_b, y_train_b)
y_prob_rf = model_2_rf.predict_proba(X_test_b)[:, 1]
auc_rf = roc_auc_score(y_test_b, y_prob_rf)

print(f"\nRandom Forest AUC: {auc_rf:.4f}")
print(f"XGBoost AUC:       {roc_auc_score(y_test_b, y_prob_b):.4f}")

if auc_rf > roc_auc_score(y_test_b, y_prob_b):
    print("✓ Random Forest is better - saving it instead")
    joblib.dump(model_2_rf, 'processed_data/build_failure_model_clean.pkl')

#     model 2 alternative trained


# =========================
# AGGREGATE LINE PREDICTIONS TO BUILD LEVEL (MEMORY EFFICIENT)
# =========================
print("\n" + "=" * 70)
print("STEP: AGGREGATING LINE PREDICTIONS FOR ENSEMBLE")
print("=" * 70)

if df_lines_with_build is not None and model_1 is not None:

    print(f"Processing {len(df_lines_with_build):,} lines...")

    # Check how many builds we're working with
    num_builds = df_lines_with_build['build_id'].nunique()
    print(f"Unique builds: {num_builds:,}")

    if num_builds == 0:
        print("✗ No builds to aggregate - skipping ensemble")
        build_line_agg = None
    else:
        # Get line features (dynamically detect which exist)
        potential_features = [
            'line_length', 'nested_depth', 'sentiment_neg', 'sentiment_pos',
            'sentiment_neu', 'sentiment_compound', 'is_comment', 'is_todo_comment',
            'control_flow_count', 'method_call_count', 'has_exception',
            'is_diff_header', 'has_magic_number', 'has_logging', 'has_null_check',
            'is_test_file', 'is_config_file', 'path_depth', 'class_name_length'
        ]

        available_features = [f for f in potential_features if f in df_lines_with_build.columns]
        print(f"Using {len(available_features)} features for prediction")

        # Predict in batches to manage memory
        print("\nPredicting bug probability in batches...")
        batch_size = 50000
        predictions = []

        for i in range(0, len(df_lines_with_build), batch_size):
            batch = df_lines_with_build.iloc[i:i + batch_size]
            X_batch = batch[available_features].fillna(0)
            batch_preds = model_1.predict_proba(X_batch)[:, 1]
            predictions.extend(batch_preds)

            if (i + batch_size) % 100000 == 0 or i + batch_size >= len(df_lines_with_build):
                processed = min(i + batch_size, len(df_lines_with_build))
                print(f"  Processed {processed:,} / {len(df_lines_with_build):,} lines ({processed / len(df_lines_with_build) * 100:.1f}%)")

        df_lines_with_build['bug_probability'] = predictions
        print("✓ Predictions complete")

        # Aggregate by build (memory efficient)
        print("\nAggregating to build level...")

        # Only keep columns we need for aggregation
        agg_cols = ['build_id', 'bug_probability']
        if 'is_todo_comment' in df_lines_with_build.columns:
            agg_cols.append('is_todo_comment')
        if 'sentiment_compound' in df_lines_with_build.columns:
            agg_cols.append('sentiment_compound')
        if 'has_exception' in df_lines_with_build.columns:
            agg_cols.append('has_exception')

        df_agg = df_lines_with_build[agg_cols].copy()

        # Aggregate
        build_line_agg = df_agg.groupby('build_id').agg({
            'bug_probability': ['mean', 'max', 'std', 'count']
        }).reset_index()

        # Flatten column names
        build_line_agg.columns = ['build_id', 'mean_line_risk', 'max_line_risk', 'std_line_risk', 'num_lines']

        # Add optional features if they exist
        if 'is_todo_comment' in df_agg.columns:
            build_line_agg['num_todo_comments'] = df_agg.groupby('build_id')['is_todo_comment'].sum().values

        if 'sentiment_compound' in df_agg.columns:
            build_line_agg['mean_sentiment'] = df_agg.groupby('build_id')['sentiment_compound'].mean().values

        if 'has_exception' in df_agg.columns:
            build_line_agg['has_exception_handling'] = (df_agg.groupby('build_id')['has_exception'].sum() > 0).astype(int).values

        # Count high-risk lines (do this separately to manage memory)
        print("Calculating high-risk line counts...")
        high_risk_counts = df_agg.groupby('build_id')['bug_probability'].apply(lambda x: (x > 0.7).sum())
        build_line_agg['num_high_risk_lines'] = high_risk_counts.values

        # Calculate percentage
        build_line_agg['pct_risky_lines'] = build_line_agg['num_high_risk_lines'] / build_line_agg['num_lines']

        print(f"✓ Aggregated to {len(build_line_agg):,} builds")
        print(f"✓ Created {len(build_line_agg.columns) - 1} aggregate features")

        # Show summary statistics
        print("\n--- Aggregation Summary ---")
        print(f"Mean lines per build: {build_line_agg['num_lines'].mean():.0f}")
        print(f"Mean line risk: {build_line_agg['mean_line_risk'].mean():.3f}")
        print(f"Mean high-risk lines per build: {build_line_agg['num_high_risk_lines'].mean():.1f}")

        # Clean up memory
        del df_agg
        del predictions

        # Save aggregates
        build_line_agg.to_csv('processed_data/build_line_aggregates.csv', index=False)
        print("\n✓ Saved to processed_data/build_line_aggregates.csv")

else:
    print("\n✗ Skipping aggregation (no linked data or Model 1 not trained)")
    build_line_agg = None


# =========================
# TWO-STAGE PIPELINE: MODEL 1 → MODEL 2
# =========================
# =========================
# TWO-STAGE PIPELINE: MODEL 1 → MODEL 2
# =========================
print("\n" + "=" * 70)
print("TWO-STAGE PIPELINE: LINE FILTERING → BUILD PREDICTION")
print("=" * 70)

if build_line_agg is not None and model_2 is not None:

    # Merge line aggregates with build data
    print("Setting up pipeline...")
    df_builds_with_lines = df_builds.merge(
        build_line_agg,
        on='build_id',
        how='inner'  # Only matched builds
    )

    print(f"✓ Working with {len(df_builds_with_lines):,} builds with line data")

    if len(df_builds_with_lines) < 20:
        print("\n❌ Insufficient data for pipeline evaluation")
    else:
        # Sort by timestamp and split
        df_builds_with_lines = df_builds_with_lines.sort_values('ts').reset_index(drop=True)

        n = len(df_builds_with_lines)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)

        train_data = df_builds_with_lines.iloc[:train_end]
        val_data = df_builds_with_lines.iloc[train_end:val_end]
        test_data = df_builds_with_lines.iloc[val_end:]

        print(f"\nData splits:")
        print(f"  Training: {len(train_data)} builds")
        print(f"  Validation: {len(val_data)} builds")
        print(f"  Test: {len(test_data)} builds")


        # =========================
        # DEFINE PIPELINE STRATEGIES
        # =========================

        def pipeline_stage1_filter(df, threshold=0.5):
            """
            Stage 1: Filter builds based on line-level risk

            Returns:
            - high_risk_mask: Boolean mask of high-risk builds
            - low_risk_mask: Boolean mask of low-risk builds
            """
            # Multiple filtering criteria
            high_risk_mask = (
                    (df['max_line_risk'] > 0.8) |  # Any line very risky
                    (df['mean_line_risk'] > threshold) |  # Average risk high
                    (df['pct_risky_lines'] > 0.3) |  # >30% lines risky
                    (df['num_todo_comments'] > 3)  # Many TODOs
            )

            low_risk_mask = ~high_risk_mask

            return high_risk_mask, low_risk_mask


        def pipeline_stage2_predict(df_high_risk, df_low_risk, model2, features):
            """
            Stage 2: Model 2 makes final decision

            Strategy:
            - Low risk builds: Predict PASS (skip Model 2)
            - High risk builds: Use Model 2 to decide
            """
            predictions = []
            probabilities = []

            # Low-risk builds: Assume pass (override Model 2)
            for idx in df_low_risk.index:
                predictions.append(0)  # Predict PASS
                probabilities.append(0.2)  # Low probability of failure

            # High-risk builds: Use Model 2
            if len(df_high_risk) > 0:
                X_high_risk = df_high_risk[features].fillna(0)
                probs = model2.predict_proba(X_high_risk)[:, 1]
                preds = (probs > 0.5).astype(int)

                for i in range(len(df_high_risk)):
                    predictions.append(preds[i])
                    probabilities.append(probs[i])

            return np.array(predictions), np.array(probabilities)


        def pipeline_hybrid_predict(df, model2, features, alpha=0.5):
            """
            Hybrid: Combine Model 1 signals with Model 2

            Strategy: Model 2 gets BOOSTED by line-level signals
            """
            X = df[features].fillna(0)
            model2_probs = model2.predict_proba(X)[:, 1]

            # Boost based on line-level signals
            line_risk_boost = (
                    0.3 * df['max_line_risk'].values +  # Critical lines
                    0.2 * df['mean_line_risk'].values +  # Average risk
                    0.1 * (df['num_todo_comments'] > 2).astype(int).values +  # TODOs
                    0.1 * (df['pct_risky_lines'] > 0.3).astype(int).values  # High % risky
            )

            # Combine: Model 2 + line signals
            final_prob = np.clip(model2_probs + line_risk_boost, 0, 1)
            final_pred = (final_prob > 0.5).astype(int)

            return final_pred, final_prob


        # =========================
        # TUNE PIPELINE ON VALIDATION SET
        # =========================

        print("\n" + "=" * 50)
        print("TUNING PIPELINE STRATEGIES")
        print("=" * 50)

        # Try different Stage 1 thresholds
        best_threshold = 0.5
        best_strategy_auc = 0
        best_strategy_name = ""

        strategies = {
            'Stage1-Filter (threshold=0.3)': {'threshold': 0.3, 'type': 'filter'},
            'Stage1-Filter (threshold=0.5)': {'threshold': 0.5, 'type': 'filter'},
            'Stage1-Filter (threshold=0.7)': {'threshold': 0.7, 'type': 'filter'},
            'Hybrid (alpha=0.5)': {'alpha': 0.5, 'type': 'hybrid'},
        }

        results = []

        for name, params in strategies.items():
            if params['type'] == 'filter':
                # Filter strategy
                threshold = params['threshold']
                high_risk_mask, low_risk_mask = pipeline_stage1_filter(val_data, threshold)

                df_high_risk = val_data[high_risk_mask]
                df_low_risk = val_data[low_risk_mask]

                # Predict
                preds, probs = pipeline_stage2_predict(df_high_risk, df_low_risk, model_2, build_features)

                # Sort back to original order
                all_indices = list(df_low_risk.index) + list(df_high_risk.index)
                sort_order = np.argsort(all_indices)
                probs_sorted = probs[sort_order]

            elif params['type'] == 'hybrid':
                # Hybrid strategy
                preds, probs_sorted = pipeline_hybrid_predict(val_data, model_2, build_features)

            # Evaluate
            auc = roc_auc_score(val_data['failed'].values, probs_sorted)
            results.append({'strategy': name, 'auc': auc, 'params': params})

            print(f"{name}: AUC = {auc:.4f}")

            if auc > best_strategy_auc:
                best_strategy_auc = auc
                best_strategy_name = name
                best_params = params

        print(f"\n✓ Best strategy: {best_strategy_name}")
        print(f"  Validation AUC: {best_strategy_auc:.4f}")

        # =========================
        # EVALUATE ON TEST SET
        # =========================

        print("\n" + "=" * 70)
        print("PIPELINE EVALUATION ON TEST SET")
        print("=" * 70)

        # Apply best strategy to test set
        if best_params['type'] == 'filter':
            threshold = best_params['threshold']
            high_risk_mask, low_risk_mask = pipeline_stage1_filter(test_data, threshold)

            df_high_risk = test_data[high_risk_mask]
            df_low_risk = test_data[low_risk_mask]

            print(f"\nStage 1 Filtering (threshold={threshold}):")
            print(f"  Low-risk builds: {len(df_low_risk)} ({len(df_low_risk) / len(test_data) * 100:.1f}%)")
            print(f"  High-risk builds: {len(df_high_risk)} ({len(df_high_risk) / len(test_data) * 100:.1f}%)")

            preds, probs = pipeline_stage2_predict(df_high_risk, df_low_risk, model_2, build_features)

            # Sort back
            all_indices = list(df_low_risk.index) + list(df_high_risk.index)
            sort_order = np.argsort(all_indices)
            probs_sorted = probs[sort_order]
            preds_sorted = preds[sort_order]

        elif best_params['type'] == 'hybrid':
            preds_sorted, probs_sorted = pipeline_hybrid_predict(test_data, model_2, build_features)

        y_test = test_data['failed'].values

        # Compare to baselines
        print("\n--- BASELINE: Model 1 Only (Aggregated) ---")
        auc_m1 = roc_auc_score(y_test, test_data['mean_line_risk'].values)
        print(f"ROC-AUC: {auc_m1:.4f}")

        print("\n--- BASELINE: Model 2 Only ---")
        X_test_m2 = test_data[build_features].fillna(0)
        m2_probs = model_2.predict_proba(X_test_m2)[:, 1]
        auc_m2 = roc_auc_score(y_test, m2_probs)
        print(f"ROC-AUC: {auc_m2:.4f}")

        print(f"\n--- PIPELINE: {best_strategy_name} ---")
        auc_pipeline = roc_auc_score(y_test, probs_sorted)
        print(f"ROC-AUC: {auc_pipeline:.4f}")
        print(classification_report(y_test, preds_sorted, digits=4))

        # Summary
        print("\n" + "=" * 70)
        print("PIPELINE RESULTS SUMMARY")
        print("=" * 70)
        print(f"Model 1 (Line agg):  AUC = {auc_m1:.4f}")
        print(f"Model 2 (Build):     AUC = {auc_m2:.4f}")
        print(f"Pipeline:            AUC = {auc_pipeline:.4f}")

        best_baseline = max(auc_m1, auc_m2)
        improvement = auc_pipeline - best_baseline
        improvement_pct = (improvement / best_baseline) * 100

        print(f"\nBest baseline:       AUC = {best_baseline:.4f}")
        print(f"Pipeline improvement: {improvement:+.4f} ({improvement_pct:+.1f}%)")

        if improvement > 0.02:
            print("\n✓ Pipeline provides meaningful improvement!")
        elif improvement > 0:
            print("\n⚠️  Pipeline provides marginal improvement")
        else:
            print("\n✗ Pipeline does not improve over baselines")

        # Save results
        pipeline_metadata = {
            'best_strategy': best_strategy_name,
            'best_params': best_params,
            'auc_model1': auc_m1,
            'auc_model2': auc_m2,
            'auc_pipeline': auc_pipeline,
            'improvement': improvement
        }

        import json

        with open('processed_data/pipeline_metadata.json', 'w') as f:
            json.dump(pipeline_metadata, f, indent=2)

        print("\n✓ Saved pipeline metadata to processed_data/pipeline_metadata.json")

else:
    print("\n✗ Cannot create pipeline (missing line aggregates or Model 2)")

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


# model ensemble; voting mechanisms sikit learn has gread setups for it - look up that documentation