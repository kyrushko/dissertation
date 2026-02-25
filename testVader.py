import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

print("\n" + "=" * 70)
print("VADER SENTIMENT TEST - SINGLE FILE")
print("=" * 70)

# Load just one file
test_file = 'data/train-lines/diff_AcDisplay.csv'
df_test = pd.read_csv(test_file, delimiter='$', quotechar='"')

print(f"\n✓ Loaded {len(df_test):,} lines from {test_file}")


df_sample = df_test.head(10000).copy()

print(f"Testing on first {len(df_sample):,} lines...")

# Clean content
df_sample['content_clean'] = df_sample['contents'].astype(str).str.strip()

# Run VADER
analyzer = SentimentIntensityAnalyzer()
sentiments = df_sample['content_clean'].apply(
    lambda x: analyzer.polarity_scores(str(x))
)

df_sample['sentiment_neg'] = sentiments.apply(lambda x: x['neg'])
df_sample['sentiment_pos'] = sentiments.apply(lambda x: x['pos'])
df_sample['sentiment_neu'] = sentiments.apply(lambda x: x['neu'])
df_sample['sentiment_compound'] = sentiments.apply(lambda x: x['compound'])

# Show results
print("\n" + "=" * 50)
print("VADER RESULTS")
print("=" * 50)

print("\nSentiment Statistics:")
print(df_sample[['sentiment_neg', 'sentiment_pos', 'sentiment_neu', 'sentiment_compound']].describe())

print("\nExample Lines with High Negative Sentiment:")
negative_lines = df_sample.nlargest(5, 'sentiment_neg')[['content_clean', 'sentiment_neg', 'sentiment_compound']]
for idx, row in negative_lines.iterrows():
    print(f"\nNeg: {row['sentiment_neg']:.3f} | Compound: {row['sentiment_compound']:.3f}")
    print(f"Line: {row['content_clean'][:100]}")

print("\nExample Lines with High Positive Sentiment:")
positive_lines = df_sample.nlargest(5, 'sentiment_pos')[['content_clean', 'sentiment_pos', 'sentiment_compound']]
for idx, row in positive_lines.iterrows():
    print(f"\nPos: {row['sentiment_pos']:.3f} | Compound: {row['sentiment_compound']:.3f}")
    print(f"Line: {row['content_clean'][:100]}")

# Check correlation with bugs
if 'class_value' in df_sample.columns:
    df_sample['is_buggy'] = pd.to_numeric(df_sample['class_value'], errors='coerce').fillna(0).astype(int)

    print("\n" + "=" * 50)
    print("CORRELATION WITH BUGS")
    print("=" * 50)

    print("\nMean Sentiment by Bug Status:")
    print(df_sample.groupby('is_buggy')[['sentiment_neg', 'sentiment_pos', 'sentiment_compound']].mean())

    from scipy.stats import ttest_ind

    buggy = df_sample[df_sample['is_buggy'] == 1]
    clean = df_sample[df_sample['is_buggy'] == 0]

    if len(buggy) > 0 and len(clean) > 0:
        t_stat, p_val = ttest_ind(buggy['sentiment_compound'], clean['sentiment_compound'])
        print(f"\nT-test for sentiment_compound difference:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_val:.4f}")

        if p_val < 0.05:
            print("  ✓ Sentiment is SIGNIFICANTLY different between buggy and clean lines!")
        else:
            print("  ✗ No significant difference in sentiment.")

print("\n✓ VADER test complete!")



# how to modify vocabulary of vader to keep the correct sentiment