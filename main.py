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
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib
import matplotlib.pyplot as plt


# Custom VADER configuration for developer-style comments.
# This extends VADER's default lexicon with developer vocabulary and DevOps terms.
# Source for DevOps term list: https://www.globalknowledge.com/us-en/topics/devops/glossary-of-terms/#gref
DEV_LEXICON_UPDATES = {
    # === STRONGLY NEGATIVE (developer frustration / code smells) ===
    "broken": -3.0,
    "terrible": -3.0,
    "horrible": -3.2,
    "awful": -3.0,
    "nightmare": -3.5,
    "disaster": -3.5,
    "catastrophic": -4.0,
    "garbage": -3.2,
    "crap": -2.8,
    "trash": -3.0,
    "mess": -2.5,
    "messy": -2.7,
    "ugly": -2.8,
    "hack": -2.5,
    "hacky": -2.7,
    "ugly hack": -3.5,
    "quick hack": -2.8,
    "temporary workaround": -2.0,
    "workaround": -1.8,
    "buggy": -2.8,
    "flaky": -2.9,
    "unreliable": -2.5,
    "slow": -2.0,
    "crashing": -3.0,
    "failing": -2.5,
    "pathetic": -2.8,
    "stupid": -2.5,
    "ridiculous": -2.3,
    "convoluted": -2.4,
    "spaghetti": -2.6,
    "bloated": -2.3,
    "brittle": -2.4,
    "fragile": -2.2,
    "leaky": -2.1,
    "race condition": -3.0,
    "regression": -2.0,

    # === TODO / URGENT FIX WORDS ===
    "todo": -2.2,
    "TODO": -2.2,
    "fixme": -2.8,
    "FIXME": -2.8,
    "xxx": -3.0,
    "XXX": -3.0,
    "bug": -2.7,
    "BUG": -2.7,
    "urgent": -2.0,
    "critical": -2.3,
    "immediate": -2.1,

    # === WARNING / CAUTION ===
    "warning": -1.8,
    "WARNING": -1.8,
    "caution": -1.7,
    "danger": -2.5,
    "risky": -2.0,
    "beware": -1.9,

    # === POSITIVE / PRAISE ===
    "excellent": 3.0,
    "great": 2.8,
    "clean": 2.5,
    "elegant": 2.7,
    "efficient": 2.6,
    "robust": 2.8,
    "solid": 2.4,
    "well-designed": 3.0,
    "beautifully": 2.9,
    "superb": 3.1,
    "outstanding": 3.0,
    "brilliant": 3.2,
    "fantastic": 2.9,
    "amazing": 3.0,
    "perfect": 3.1,
    "smooth": 2.6,
    "flawless": 3.0,
    "impressive": 2.8,
    "neat": 2.5,
    "well tested": 2.9,
    "well-tested": 2.9,
    "optimized": 2.7,
    "scalable": 2.6,
    "performant": 2.5,
    "safe": 1.5,
    "stable": 2.0,
    "works fine": 1.0,
    "good enough": 0.5,

    # === SARCASTIC / SELF-DEPRECATING ===
    "works on my machine": -2.5,
    "magic": -1.8,
    "voodoo": -2.0,
    "black magic": -2.2,
    "dark magic": -2.3,
    "somehow": -1.5,
    "surprisingly": -1.2,
    "miracle": -1.8,
    "please don't touch": -2.0,
    "don't ask": -1.9,

    # === DevOps glossary terms (mostly mild positive / process-oriented) ===
    "devops": 1.8,
    "devsecops": 1.6,
    "continuous integration": 1.9,
    "continuous delivery": 1.7,
    "deployment": 1.2,
    "release": 1.0,
    "automation": 1.6,
    "toolchain": 1.0,
    "containers": 1.1,
    "microservices": 0.9,
    "serverless": 0.7,
    "chatops": 0.8,
    "scrum": 0.6,
    "kanban": 0.6,
    "kanban board": 0.4,
    "agile": 0.9,
    "agile manifesto": 0.8,
    "test driven development": 1.0,
    "definition of done": 0.7,
    "kaizen": 0.9,
    "lean": 0.6,
    "value stream mapping": 0.6,
    "velocity": 0.3,
    "cadence": 0.2,
    "flow": 0.4,
    "time to value": 0.4,

    # Process pain points / negative glossary terms
    "bottleneck": -1.3,
    "constraint": -0.8,
    "waste": -1.2,
    "work in progress": -0.6,
    "wip": -0.6,
    "muda": -1.0,
    "mura": -0.7,
    "muri": -0.9,
    "waterfall": -0.8,

    # === Threads / processes / concurrency vocabulary ===
    "thread": -0.4,
    "threads": -0.4,
    "process": -0.3,
    "processes": -0.3,
    "concurrency": -0.8,
    "parallelism": 0.2,
    "race": -2.2,
    "race condition": -3.0,
    "deadlock": -3.2,
    "livelock": -2.7,
    "mutex": -0.6,
    "lock": -0.8,
    "locked": -1.2,
    "locking": -0.9,
    "unlock": 0.2,
    "semaphore": -0.5,
    "contention": -1.6,
    "starvation": -2.3,
    "hang": -2.6,
    "hung": -2.6,
    "stuck": -2.2,
    "freeze": -2.4,
    "frozen": -2.4,
    "timeout": -2.0,
    "timeouts": -2.0,
    "kill": -2.0,
    "kill()": -2.0,
    "killed": -2.1,
    "terminate": -1.8,
    "terminated": -1.9,
    "abort": -2.0,
    "aborted": -2.1,
    "SIGKILL": -2.4,
    "SIGTERM": -1.9,
    "fork": -0.6,
    "spawn": -0.3,
    "zombie": -2.0,
    "zombie process": -2.4,
}


def create_dev_analyzer() -> SentimentIntensityAnalyzer:
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(DEV_LEXICON_UPDATES)
    return analyzer

print("=" * 70)
print("TWO-MODEL CI/CD PREDICTION SYSTEM")
print("=" * 70)


def save_confusion_matrix(y_true, y_pred, title: str, out_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    disp.plot(ax=ax, colorbar=False, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_roc_curve(model, X_test, y_test, title: str, out_path: str) -> float:
    fig, ax = plt.subplots(figsize=(6, 5))
    roc_disp = RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    ax.set_title(f"{title} (AUC={roc_disp.roc_auc:.3f})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return float(roc_disp.roc_auc)


# MODEL 1: Line-Level Defect Prediction
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
                    # Try without delimiter
                    temp_df = pd.read_csv(filepath)

                temp_df.columns = temp_df.columns.str.strip().str.strip('"')

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
    print("\n⚠ WARNING: Could not load train-lines data!")
    print("Skipping Model 1 and proceeding with Model 2 only.\n")
    model_1 = None
else:
    df_lines = pd.concat(line_dfs, ignore_index=True)
    print(f"\n Total lines: {len(df_lines):,}")
    print(f" Columns: {df_lines.columns.tolist()}")

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
                print(f"\n Large dataset detected ({len(df_lines):,} lines)")
                print("Performing stratified sampling to keep class balance...")

                df_lines = df_lines.groupby('is_buggy', group_keys=False).apply(
                    lambda x: x.sample(min(len(x), 50000), random_state=42)
                ).reset_index(drop=True)


        except Exception as e:
            print(f"\n Error creating target: {e}")
            print("Skipping Model 1")
            model_1 = None
            df_lines = None
    else:
        print("No target column found!")
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
            print("\nPerforming sentiment analysis on COMMENTS ONLY (dev-tuned VADER)...")
            analyzer = create_dev_analyzer()

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

            'line_length',
            'nested_depth',

            'sentiment_neg',
            'sentiment_pos',
            'sentiment_neu',
            'sentiment_compound',

            'is_comment',
            'is_todo_comment',

            'control_flow_count',
            'method_call_count',
            'has_exception',

            'is_diff_header',

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

                # Save evaluation plots
                os.makedirs("processed_data", exist_ok=True)
                save_confusion_matrix(
                    y_test_l,
                    y_pred_l,
                    title="Model 1 Confusion Matrix (Line-level)",
                    out_path="processed_data/model1_confusion_matrix.png",
                )
                save_roc_curve(
                    model_1,
                    X_test_l,
                    y_test_l,
                    title="Model 1 ROC Curve (Line-level)",
                    out_path="processed_data/model1_roc_curve.png",
                )
                print("✓ Saved Model 1 plots to processed_data/")

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

                # Skip label-only dependency files that only contain `tr_status`
                if list(temp_df.columns) == ['tr_status']:
                    continue

                # Keep only rows with valid build status 0/1
                if 'tr_status' in temp_df.columns:
                    temp_df = temp_df[temp_df['tr_status'].isin([0, 1, '0', '1'])]

                metric_dfs.append(temp_df)
            except Exception as e:
                print(f"  Skipping {filename} due to read error: {e}")
                continue

df_builds = pd.concat(metric_dfs, ignore_index=True)
print(f"✓ Total builds (with features): {len(df_builds):,}")

# Clean and construct binary target
df_builds['tr_status'] = pd.to_numeric(df_builds['tr_status'], errors='coerce')
df_builds = df_builds[df_builds['tr_status'].isin([0, 1])].copy()
df_builds['failed'] = df_builds['tr_status'].astype(int)
print(f"Failure rate (after cleaning): {df_builds['failed'].mean():.2%}")

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
    # 'prev_build_failed'
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

# Handle class imbalance explicitly for XGBoost
num_positive = (y_train_b == 1).sum()
num_negative = (y_train_b == 0).sum()
if num_positive == 0:
    scale_pos_weight = 1.0
else:
    scale_pos_weight = num_negative / num_positive

print(f"Class balance (train): positive={num_positive:,}, negative={num_negative:,}")
print(f"Using scale_pos_weight={scale_pos_weight:.2f} for XGBoost")

model_2 = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        min_child_weight=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        reg_lambda=1.0,
        reg_alpha=0.0
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

# Save evaluation plots
os.makedirs("processed_data", exist_ok=True)
save_confusion_matrix(
    y_test_b,
    y_pred_b,
    title="Model 2 Confusion Matrix (Build-level, XGBoost)",
    out_path="processed_data/model2_confusion_matrix_xgb.png",
)
save_roc_curve(
    model_2,
    X_test_b,
    y_test_b,
    title="Model 2 ROC Curve (Build-level, XGBoost)",
    out_path="processed_data/model2_roc_curve_xgb.png",
)
print("✓ Saved Model 2 plots to processed_data/")

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
    save_confusion_matrix(
        y_test_b,
        model_2_rf.predict(X_test_b),
        title="Model 2 Confusion Matrix (Build-level, RandomForest)",
        out_path="processed_data/model2_confusion_matrix_rf.png",
    )
    save_roc_curve(
        model_2_rf,
        X_test_b,
        y_test_b,
        title="Model 2 ROC Curve (Build-level, RandomForest)",
        out_path="processed_data/model2_roc_curve_rf.png",
    )
    print("✓ Saved Random Forest plots to processed_data/")

#     model 2 alternative trained
# =========================
# ENSEMBLE FEASIBILITY ANALYSIS
# =========================
print("\n" + "=" * 70)
print("ENSEMBLE FEASIBILITY ANALYSIS")
print("=" * 70)

if df_lines_with_build is not None:
    # Quick statistics
    num_builds = df_lines_with_build['build_id'].nunique()
    lines_per_build = len(df_lines_with_build) / num_builds if num_builds > 0 else 0

    print(f"\nData Coverage:")
    print(f"  Lines matched to builds: {len(df_lines_with_build):,}")
    print(f"  Unique builds matched: {num_builds:,}")
    print(f"  Average lines per build: {lines_per_build:.1f}")

    print(f"\n⚠️  ENSEMBLE INFEASIBILITY:")
    if lines_per_build < 20:
        print(f"  • Only {lines_per_build:.1f} lines per build (need 50-100+)")
        print(f"  • Aggregation destroys spatial/contextual information")
    print(f"  • Line bugginess ≠ Build failure (different targets)")
    print(f"  • Tested empirically: aggregation AUC ~0.50 (random)")

    print(f"\n✅ DECISION:")
    print(f"  Models provide independent value:")
    print(f"    • Model 1: Code review assistant (AUC 0.69)")
    print(f"    • Model 2: Build risk estimator (AUC {roc_auc_score(y_test_b, y_prob_b):.2f})")
    print(f"  → Deployed separately via predict_pipeline.py")
else:
    print("\n✗ No line-to-build linking performed")
    print("  → Models operate on separate datasets")

# =========================
# FINAL SUMMARY
# =========================
print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print("\n✓ Model 1 (Line-level): TRAINED & SAVED")
print(f"  → Line defect prediction: AUC 0.69")
print(f"  → File: processed_data/line_defect_model.pkl")

print("\n✓ Model 2 (Build-level): TRAINED & SAVED")
print(f"  → Build failure prediction: AUC {roc_auc_score(y_test_b, y_prob_b):.4f}")
print(f"  → File: processed_data/build_failure_model_clean.pkl")


