"""
Dual-Model CI/CD Prediction Pipeline
Provides separate assessments from line-level and build-level models
"""

import pandas as pd
import numpy as np
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime

print("=" * 70)
print("CI/CD FAILURE PREDICTION PIPELINE")
print("=" * 70)

# =========================
# Load Trained Models
# =========================
print("\nLoading models...")

try:
    model_1 = joblib.load('processed_data/line_defect_model.pkl')
    print("✓ Loaded Model 1 (Line-level defect prediction)")
except:
    print("✗ Could not load Model 1")
    model_1 = None

try:
    model_2 = joblib.load('processed_data/build_failure_model_clean.pkl')
    print("✓ Loaded Model 2 (Build-level failure prediction)")
except:
    print("✗ Could not load Model 2")
    model_2 = None


# =========================
# Feature Extraction Functions
# =========================

def extract_line_features(lines):
    """
    Extract features from lines of code

    Args:
        lines: list of strings (code lines)

    Returns:
        DataFrame with features
    """
    df = pd.DataFrame({'contents': lines})
    df['content_clean'] = df['contents'].astype(str).str.strip()

    # Basic features
    df['line_length'] = df['content_clean'].str.len()
    df['nested_depth'] = df['content_clean'].apply(
        lambda x: len(x) - len(x.lstrip()) if isinstance(x, str) else 0
    ) // 4

    # Comments
    df['is_comment'] = df['content_clean'].str.contains(
        r'^\s*(?://|/\*|\*|#)', regex=True, na=False
    ).astype(int)

    df['is_todo_comment'] = df['content_clean'].str.contains(
        r'\b(?:TODO|FIXME|HACK|XXX|BUG)\b', case=False, regex=True, na=False
    ).astype(int)

    # Sentiment (only on comments)
    analyzer = SentimentIntensityAnalyzer()

    sentiment_neg = np.zeros(len(df), dtype=np.float32)
    sentiment_pos = np.zeros(len(df), dtype=np.float32)
    sentiment_neu = np.zeros(len(df), dtype=np.float32)
    sentiment_compound = np.zeros(len(df), dtype=np.float32)

    comment_mask = df['is_comment'] == 1
    for idx in df[comment_mask].index:
        text = str(df.loc[idx, 'content_clean'])
        scores = analyzer.polarity_scores(text)
        sentiment_neg[idx] = scores['neg']
        sentiment_pos[idx] = scores['pos']
        sentiment_neu[idx] = scores['neu']
        sentiment_compound[idx] = scores['compound']

    df['sentiment_neg'] = sentiment_neg
    df['sentiment_pos'] = sentiment_pos
    df['sentiment_neu'] = sentiment_neu
    df['sentiment_compound'] = sentiment_compound

    # Code complexity
    df['control_flow_count'] = df['content_clean'].str.count(
        r'\b(?:if|else|for|while|switch|case|catch|try)\b'
    )

    df['method_call_count'] = df['content_clean'].str.count(r'\w+\(')

    df['has_exception'] = df['content_clean'].str.contains(
        r'\b(?:Exception|Error|throw|throws)\b', regex=True, na=False
    ).astype(int)

    df['is_diff_header'] = df['content_clean'].str.contains(
        r'^(?:\+\+|--|@@)', regex=True, na=False
    ).astype(int)

    df['has_magic_number'] = df['content_clean'].str.contains(
        r'\b\d{2,}\b', regex=True, na=False
    ).astype(int)

    df['has_logging'] = df['content_clean'].str.contains(
        r'\b(?:Log\.|logger|println)\b', regex=True, na=False
    ).astype(int)

    df['has_null_check'] = df['content_clean'].str.contains(
        r'\b(?:null|NULL|isNull|isEmpty)\b', regex=True, na=False
    ).astype(int)

    # File context (defaults)
    df['is_test_file'] = 0
    df['is_config_file'] = 0
    df['path_depth'] = 2
    df['class_name_length'] = 10

    return df


def predict_line_defects(lines):
    """
    Predict which lines are likely buggy

    Args:
        lines: list of strings (code lines)

    Returns:
        DataFrame with predictions and risk scores
    """
    if model_1 is None:
        print("✗ Model 1 not loaded")
        return None

    # Extract features
    df = extract_line_features(lines)

    # Feature list (must match training)
    features = [
        'line_length', 'nested_depth', 'sentiment_neg', 'sentiment_pos',
        'sentiment_neu', 'sentiment_compound', 'is_comment', 'is_todo_comment',
        'control_flow_count', 'method_call_count', 'has_exception',
        'is_diff_header', 'has_magic_number', 'has_logging', 'has_null_check',
        'is_test_file', 'is_config_file', 'path_depth', 'class_name_length'
    ]

    # Filter to available features
    features = [f for f in features if f in df.columns]

    X = df[features].fillna(0)

    # Predict
    predictions = model_1.predict(X)
    probabilities = model_1.predict_proba(X)[:, 1]

    # Add to dataframe
    df['predicted_buggy'] = predictions
    df['bug_risk'] = probabilities

    return df[['contents', 'predicted_buggy', 'bug_risk', 'is_comment',
               'is_todo_comment', 'sentiment_compound']]


def predict_build_failure(build_metrics):
    """
    Predict if a build will fail

    Args:
        build_metrics: dict with build features

    Returns:
        (prediction, probability, assessment)
    """
    if model_2 is None:
        print("✗ Model 2 not loaded")
        return None, None, None

    # Feature list
    features = [
        'gh_num_commits_in_push', 'git_prev_commit_resolution_status',
        'gh_team_size', 'git_num_all_built_commits', 'gh_num_commit_comments',
        'git_diff_src_churn', 'gh_diff_files_added', 'gh_diff_files_deleted',
        'gh_diff_files_modified', 'gh_diff_src_files', 'gh_diff_doc_files',
        'gh_diff_other_files', 'gh_num_commits_on_files_touched', 'gh_sloc',
        'hour', 'day_of_week', 'month', 'is_weekend', 'is_night',
        'files_per_commit', 'churn_per_file', 'team_commits_ratio',
        'large_change', 'many_files_touched'
    ]

    # Fill in missing features with defaults
    X_dict = {}
    for feat in features:
        X_dict[feat] = build_metrics.get(feat, 0)

    X = pd.DataFrame([X_dict])

    # Predict
    prediction = model_2.predict(X)[0]
    probability = model_2.predict_proba(X)[0, 1]

    # Assessment
    if probability < 0.3:
        assessment = "LOW RISK"
    elif probability < 0.6:
        assessment = "MEDIUM RISK"
    else:
        assessment = "HIGH RISK"

    return prediction, probability, assessment


# =========================
# EXAMPLE USAGE
# =========================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("EXAMPLE 1: LINE-LEVEL ANALYSIS")
    print("=" * 70)

    # Sample code to analyze
    sample_code = [
        "public class UserService {",
        "    // TODO: This is a terrible hack that needs urgent fixing",
        "    public User getUser(int id) {",
        "        if (id < 0) throw new IllegalArgumentException();",
        "        User user = database.query(123456);  // Magic number",
        "        return user;",
        "    }",
        "    ",
        "    // Well-tested and optimized method",
        "    public void saveUser(User user) {",
        "        if (user == null) {",
        "            logger.warn(\"Null user provided\");",
        "            return;",
        "        }",
        "        database.save(user);",
        "    }",
        "}"
    ]

    if model_1:
        print("\nAnalyzing code with Model 1...")
        results = predict_line_defects(sample_code)

        print("\n📝 Line-by-Line Assessment:")
        print("=" * 70)

        high_risk_lines = []
        for idx, row in results.iterrows():
            risk = row['bug_risk']

            if risk > 0.7:
                level = "🔴 HIGH RISK"
                high_risk_lines.append(idx + 1)
            elif risk > 0.5:
                level = "🟡 MEDIUM"
            else:
                level = "🟢 LOW"

            print(f"\nLine {idx + 1}: {level} (risk={risk:.2f})")
            print(f"  Code: {row['contents'][:60]}")

            if row['is_todo_comment']:
                print(f"  ⚠️  Contains TODO/FIXME/HACK")
            if row['is_comment'] and abs(row['sentiment_compound']) > 0.3:
                print(f"  💬 Sentiment: {row['sentiment_compound']:.2f}")

        # Summary
        print("\n" + "=" * 70)
        print("MODEL 1 SUMMARY")
        print("=" * 70)
        print(f"Total lines: {len(results)}")
        print(f"High-risk lines: {len(high_risk_lines)} → {high_risk_lines}")
        print(f"Average risk: {results['bug_risk'].mean():.2f}")
        print(f"Max risk: {results['bug_risk'].max():.2f}")
        print(f"TODO comments: {results['is_todo_comment'].sum()}")

        print("\n💡 RECOMMENDATION:")
        if len(high_risk_lines) > 0:
            print(f"  ⚠️  Review lines {high_risk_lines} - they show high defect risk")
        if results['is_todo_comment'].sum() > 0:
            print(f"  ⚠️  Address {results['is_todo_comment'].sum()} TODO/FIXME comments")
        if results['bug_risk'].mean() > 0.5:
            print(f"  ⚠️  Overall code quality is concerning (avg risk {results['bug_risk'].mean():.2f})")
        else:
            print(f"  ✓ Code quality looks acceptable")

    print("\n" + "=" * 70)
    print("EXAMPLE 2: BUILD-LEVEL ANALYSIS")
    print("=" * 70)

    # Sample build metrics
    current_time = datetime.now()
    sample_build = {
        'gh_num_commits_in_push': 3,
        'git_prev_commit_resolution_status': 1,  # Previous build passed
        'gh_team_size': 4,
        'git_num_all_built_commits': 250,
        'gh_num_commit_comments': 1,
        'git_diff_src_churn': 45,  # Lines changed
        'gh_diff_files_added': 1,
        'gh_diff_files_deleted': 0,
        'gh_diff_files_modified': 3,
        'gh_diff_src_files': 3,
        'gh_diff_doc_files': 0,
        'gh_diff_other_files': 0,
        'gh_num_commits_on_files_touched': 20,
        'gh_sloc': 8500,  # Total codebase size
        'hour': current_time.hour,
        'day_of_week': current_time.weekday(),
        'month': current_time.month,
        'is_weekend': 1 if current_time.weekday() >= 5 else 0,
        'is_night': 1 if 0 <= current_time.hour < 6 else 0,
        'files_per_commit': 1.0,
        'churn_per_file': 15.0,
        'team_commits_ratio': 0.75,
        'large_change': 0,
        'many_files_touched': 0
    }

    if model_2:
        print("\nAnalyzing build with Model 2...")
        pred, prob, assessment = predict_build_failure(sample_build)

        print("\n🏗️  Build Assessment:")
        print("=" * 70)
        print(f"Prediction: {'❌ WILL FAIL' if pred == 1 else '✅ WILL PASS'}")
        print(f"Failure probability: {prob:.1%}")
        print(f"Risk level: {assessment}")

        print("\n📊 Key Metrics:")
        print(f"  • Commits: {sample_build['gh_num_commits_in_push']}")
        print(f"  • Files modified: {sample_build['gh_diff_files_modified']}")
        print(f"  • Code churn: {sample_build['git_diff_src_churn']} lines")
        print(f"  • Team size: {sample_build['gh_team_size']}")
        print(f"  • Previous build: {'✓ Passed' if sample_build['git_prev_commit_resolution_status'] else '✗ Failed'}")
        print(
            f"  • Time: {'Weekend' if sample_build['is_weekend'] else 'Weekday'}, {'Night' if sample_build['is_night'] else 'Day'}")

        print("\n💡 MODEL 2 RECOMMENDATION:")
        if prob > 0.6:
            print("  🔴 HIGH RISK - Consider additional testing or review")
        elif prob > 0.4:
            print("  🟡 MODERATE RISK - Standard review process")
        else:
            print("  🟢 LOW RISK - Proceed with confidence")

    # =========================
    # COMBINED ASSESSMENT
    # =========================

    if model_1 and model_2:
        print("\n" + "=" * 70)
        print("COMBINED ASSESSMENT")
        print("=" * 70)

        line_risk = results['bug_risk'].mean()
        build_risk = prob

        print(f"\nModel 1 (Code Quality):  {line_risk:.2f} avg risk")
        print(f"Model 2 (Build Outcome): {build_risk:.2f} failure prob")

        print("\n🎯 FINAL RECOMMENDATION:")

        if line_risk > 0.5 and build_risk > 0.5:
            print("  🔴 CRITICAL: Both models indicate high risk")
            print("     → Require thorough code review")
            print("     → Add extra tests")
            print("     → Consider breaking into smaller changes")

        elif line_risk > 0.5:
            print("  🟡 Code quality concerns detected")
            print("     → Review high-risk lines before merging")
            print("     → Build risk is acceptable")

        elif build_risk > 0.5:
            print("  🟡 Build failure risk detected")
            print("     → Code quality looks okay")
            print("     → Check build configuration and dependencies")

        else:
            print("  ✓ Both models show acceptable risk")
            print("    → Proceed with standard review process")

    print("\n" + "=" * 70)
    print("Pipeline ready! Modify sample_code and sample_build above to test.")
    print("=" * 70)