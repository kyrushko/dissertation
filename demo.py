import pandas as pd
import numpy as np
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime

# ANSI color codes for terminal output
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BOLD = "\033[1m"
RESET = "\033[0m"

print("=" * 70)
print("CI/CD FAILURE PREDICTION PIPELINE")
print("=" * 70)



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


# Feature Extraction Functions

def extract_line_features(lines):
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



# EXAMPLE USAGE
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
    mixed_sentiment_code = [
        "# This function works reasonably well but has some issues",
        "def calculate_discount(price, customer_type):",
        "    '''",
        "    Calculates discount based on customer type",
        "    TODO: Need to add VIP customer handling - currently missing",
        "    '''",
        "    # Basic discount logic - works fine for standard cases",
        "    if customer_type == 'regular':",
        "        # Standard discount rate - this is acceptable",
        "        return price * 0.05",
        "    elif customer_type == 'premium':",
        "        # Great discount for premium customers!",
        "        return price * 0.15",
        "    elif customer_type == 'employee':",
        "        # FIXME: Employee discount is broken - gives wrong amount",
        "        # TODO: Fix this calculation - it's causing complaints",
        "        return price * 0.25  # This is wrong and needs correction",
        "    else:",
        "        # Default case - no discount applied",
        "        return 0",
    ]

    edge_case_sarcasm_code = [
        "# This is an extraordinarily wonderful and absolutely magnificent implementation",
        "# that demonstrates excellent coding practices and brilliant design patterns",
        "# showcasing the developer's outstanding skills and impressive knowledge",
        "",
        "# WARNING: This is a catastrophically broken piece of garbage code that is",
        "# completely useless and dangerously unstable causing massive failures and",
        "# horrible bugs throughout the entire system destroying everything it touches",
        "def edge_case_function():",
        "    pass  # This is fine 🙂  (it's not fine at all)",
    ]
    if model_1:
        print("\nAnalyzing code with Model 1...")
        results = predict_line_defects(sample_code)


        print("\n Line-by-Line Assessment:")
        print("=" * 70)

        high_risk_lines = []
        # If the model is very conservative, absolute probabilities may be tiny.
        # Use both absolute thresholds and percentile thresholds to surface relative risk.
        p90 = float(results["bug_risk"].quantile(0.90))
        p75 = float(results["bug_risk"].quantile(0.75))
        print(f"\nRisk distribution: min={results['bug_risk'].min():.4f} "
              f"p75={p75:.4f} p90={p90:.4f} max={results['bug_risk'].max():.4f}")
        for idx, row in results.iterrows():
            risk = row['bug_risk']
            # Prefer absolute thresholds if probabilities are well-spread,
            # otherwise fall back to relative thresholds.
            if risk > 0.7 or (p90 > 0 and risk >= p90):
                color = RED
                level = "HIGH RISK"
                high_risk_lines.append(idx + 1)
            elif risk > 0.5 or (p75 > 0 and risk >= p75):
                color = YELLOW
                level = "MEDIUM RISK"
            else:
                color = GREEN
                level = "LOW RISK"

            print(f"\n{color}Line {idx+1}: {level} (risk={risk:.4f}){RESET}")
            print(f"{color}  Code: {row['contents'][:60]}{RESET}")

            if row['is_todo_comment']:
                print(f"{YELLOW}  ⚠️  Contains TODO/FIXME/HACK{RESET}")
            if row['is_comment'] and abs(row['sentiment_compound']) > 0.3:
                print(f"{YELLOW}  Sentiment: {row['sentiment_compound']:.2f}{RESET}")

        # Summary
        print("\n" + "=" * 70)
        print("MODEL 1 SUMMARY")
        print("=" * 70)
        print(f"Total lines: {len(results)}")
        print(f"High-risk lines: {len(high_risk_lines)} → {high_risk_lines}")
        print(f"Average risk: {results['bug_risk'].mean():.4f}")
        print(f"Max risk: {results['bug_risk'].max():.4f}")
        print(f"TODO comments: {results['is_todo_comment'].sum()}")

        print("\n💡 RECOMMENDATION:")
        if len(high_risk_lines) > 0:
            print(f"  ⚠️  Review lines {high_risk_lines} - they show high defect risk")
        if results['is_todo_comment'].sum() > 0:
            print(f"  ⚠️  Address {results['is_todo_comment'].sum()} TODO/FIXME comments")
        if results['bug_risk'].mean() > 0.5:
            print(f"  ⚠️  Overall code quality is concerning (avg risk {results['bug_risk'].mean():.2f})")
        else:
            print(f"  Code quality looks acceptable")
        print("\n📌 Additional code improvement suggestions:")
        print("  • Strengthen unit tests around high-risk control flow and error paths.")
        print("  • Replace magic numbers and hard-coded values with named constants or configuration.")
        print("  • Reduce deeply nested logic by extracting helper methods and using guard clauses.")
        print("  • Add or refine logging around risky sections to aid debugging in CI/CD.")

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
    sample_build_mixed = {
    "gh_num_commits_in_push": 2,
    "git_prev_commit_resolution_status": 1,  # previous build passed
    "gh_team_size": 3,
    "git_num_all_built_commits": 120,
    "gh_num_commit_comments": 1,
    "git_diff_src_churn": 35,        # moderate lines changed
    "gh_diff_files_added": 0,
    "gh_diff_files_deleted": 0,
    "gh_diff_files_modified": 2,
    "gh_diff_src_files": 2,
    "gh_diff_doc_files": 0,
    "gh_diff_other_files": 0,
    "gh_num_commits_on_files_touched": 15,
    "gh_sloc": 5000,
    "hour": 14,                      # afternoon
    "day_of_week": 2,                # Wednesday
    "month": 5,
    "is_weekend": 0,
    "is_night": 0,
    "files_per_commit": 1.0,
    "churn_per_file": 17.5,
    "team_commits_ratio": 0.5,
    "large_change": 0,
    "many_files_touched": 0,
    }
    sample_build_edge_case = {
    "gh_num_commits_in_push": 5,
    "git_prev_commit_resolution_status": 0,  # previous build failed
    "gh_team_size": 6,
    "git_num_all_built_commits": 450,
    "gh_num_commit_comments": 4,
    "git_diff_src_churn": 320,       # large change
    "gh_diff_files_added": 2,
    "gh_diff_files_deleted": 1,
    "gh_diff_files_modified": 9,
    "gh_diff_src_files": 9,
    "gh_diff_doc_files": 1,
    "gh_diff_other_files": 0,
    "gh_num_commits_on_files_touched": 40,
    "gh_sloc": 18000,
    "hour": 2,                       # night deploy
    "day_of_week": 6,                # Sunday
    "month": 11,
    "is_weekend": 1,
    "is_night": 1,
    "files_per_commit": 9 / (5 + 1),
    "churn_per_file": 320 / (9 + 1),
    "team_commits_ratio": 5 / (6 + 1),
    "large_change": 1,
    "many_files_touched": 1,
    }
    if model_2:
        print("\nAnalyzing build with Model 2...")
        pred, prob, assessment = predict_build_failure(sample_build_edge_case)

        print("\n  Build Assessment:")
        print("=" * 70)
        print(f"Prediction: {'❌ WILL FAIL' if pred == 1 else '✅ WILL PASS'}")
        print(f"Failure probability: {prob:.1%}")
        print(f"Risk level: {assessment}")

        print("\n Key Metrics:")
        print(f"  • Commits: {sample_build['gh_num_commits_in_push']}")
        print(f"  • Files modified: {sample_build['gh_diff_files_modified']}")
        print(f"  • Code churn: {sample_build['git_diff_src_churn']} lines")
        print(f"  • Team size: {sample_build['gh_team_size']}")
        print(f"  • Previous build: {'✓ Passed' if sample_build['git_prev_commit_resolution_status'] else '✗ Failed'}")
        print(f"  • Time: {'Weekend' if sample_build['is_weekend'] else 'Weekday'}, {'Night' if sample_build['is_night'] else 'Day'}")

        print("\n MODEL 2 RECOMMENDATION:")
        if prob > 0.6:
            color = RED
            level = "HIGH RISK"
            guidance = "Consider additional testing, peer review, and staging rollouts."
        elif prob > 0.4:
            color = YELLOW
            level = "MODERATE RISK"
            guidance = "Follow your standard review process and monitor post-deploy telemetry."
        else:
            color = GREEN
            level = "LOW RISK"
            guidance = "Proceed with confidence, but keep lightweight monitoring in place."

        print(f"{color}  {level} - {guidance}{RESET}")
        print("  • Keep CI jobs fast and deterministic to spot regressions early.")
        print("  • Prioritize tests around high-churn areas and frequently touched files.")
        print("  • Review build configuration (caching, dependencies, environment flags) for flakiness.")


    # COMBINED ASSESSMENT
    if model_1 and model_2:
        print("\n" + "=" * 70)
        print("COMBINED ASSESSMENT")
        print("=" * 70)

        line_risk = results['bug_risk'].mean()
        build_risk = prob

        print(f"\nModel 1 (Code Quality):  {line_risk:.2f} avg risk")
        print(f"Model 2 (Build Outcome): {build_risk:.2f} failure prob")

        print("\n FINAL RECOMMENDATION:")

        if line_risk > 0.5 and build_risk > 0.5:
            print(f"{RED}  CRITICAL: Both models indicate high risk{RESET}")
            print("     → Require thorough code review and sign-off from senior reviewers.")
            print("     → Add targeted regression and integration tests for risky areas.")
            print("     → Consider breaking this change into smaller, independently deployable pieces.")

        elif line_risk > 0.5:
            print(f"{YELLOW}  Code quality concerns detected{RESET}")
            print("     → Review high-risk lines before merging and refactor where possible.")
            print("     → Build risk is acceptable; focus on improving readability and test coverage.")

        elif build_risk > 0.5:
            print(f"{YELLOW}  Build failure risk detected{RESET}")
            print("     → Code quality looks okay; investigate build scripts, environments, and dependencies.")
            print("     → Introduce canary deployments or gradual rollouts to limit blast radius.")

        else:
            print("  ✓ Both models show acceptable risk")
            print("    → Proceed with standard review process and routine CI monitoring.")

    print("\n" + "=" * 70)
    print("Pipeline ready! Modify sample_code and sample_build above to test.")
    print("=" * 70)
