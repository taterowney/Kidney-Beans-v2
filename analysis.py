import os
import math
from collections import Counter, defaultdict
import pandas as pd
import re
from typing import Optional, Set
from functools import lru_cache

def load_df(path: str = "outputs/9_000000_000000.csv"):
    """Load a model output CSV and split answer into thoughts and response.

    Returns three DataFrames: (safe, unsafe, benign). Safe/unsafe are filtered by the
    boolean 'is_safe' column from the provided CSV. Benign is loaded from a directory
    of benign outputs if present.
    """
    df = pd.read_csv(path)
    if "answer" not in df.columns:
        raise KeyError("Expected column 'answer' in the CSV")
    if "is_safe" not in df.columns:
        raise KeyError("Expected column 'is_safe' in the CSV")

    # Robust splitting even if answer is NaN (coerce to str first)
    answers = df["answer"].fillna("").astype(str)
    df["thoughts"] = answers.apply(lambda x: x.split("</think>")[0])
    df["response"] = answers.apply(lambda x: x.split("</think>")[-1])

    df_safe = df[df["is_safe"] == True].reset_index(drop=True)
    df_unsafe = df[df["is_safe"] == False].reset_index(drop=True)
    
    
    df_benign = pd.DataFrame()
    benign_dir = "outputs/benign_outputs/1753111087.5420048"
    if os.path.isdir(benign_dir):
        for file in os.listdir(benign_dir):
            if file.lower().endswith(".csv"):
                file_path = os.path.join(benign_dir, file)
                try:
                    temp_df = pd.read_csv(file_path)
                    df_benign = pd.concat([df_benign, temp_df], ignore_index=True)
                except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, OSError, ValueError):
                    # Skip files that can't be read; keep analysis resilient
                    continue
    # Derive thoughts/response for benign only if 'answer' exists
    if not df_benign.empty and "answer" in df_benign.columns:
        answers_b = df_benign["answer"].fillna("").astype(str)
        df_benign["thoughts"] = answers_b.apply(lambda x: x.split("</think>")[0])
        df_benign["response"] = answers_b.apply(lambda x: x.split("</think>")[-1])

    return df_safe, df_unsafe, df_benign

def different_chars_used(df: pd.DataFrame, column: str = "thoughts") -> float:
    """Mean number of distinct characters per row in the specified column.

    Ignores missing values.
    """
    if column not in df.columns or df.empty:
        return 0.0
    series = df[column].dropna().astype(str)
    if series.empty:
        return 0.0
    processed = series.apply(lambda x: len(set(x)))
    return processed.mean(), processed.std()

def char_frequencies(df: pd.DataFrame, column: str = "thoughts") -> pd.Series:
    """Normalized character frequencies across the entire column.

    Returns a Series indexed by character with frequency (sum to 1.0). Empty Series if no text.
    """
    if column not in df.columns or df.empty:
        return pd.Series(dtype=float)
    all_text = "".join(df[column].dropna().astype(str).tolist())
    if len(all_text) == 0:
        return pd.Series(dtype=float)
    counts = pd.Series(list(all_text)).value_counts()
    return counts / counts.sum()

def biggest_char_frequency_differences(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    column: str = "thoughts",
    top_n: int = 100,
) -> pd.DataFrame:
    """Per-entry absolute char-frequency differences vs other group's aggregate (top_n).

    For each row in df1, compute its character frequency distribution and take the absolute
    difference from df2's aggregate character frequencies. Do the symmetric operation for
    each row in df2 vs df1's aggregate. Aggregate these per-entry difference vectors by
    taking the mean and std for each character. Return a DataFrame with columns
    ['mean', 'std'], indexed by character, sorted by descending mean and truncated to top_n.

    Characters absent in one side are treated as frequency 0. If both inputs are empty or
    the column is missing, returns an empty DataFrame.
    """
    # Quick exits for missing columns or empty inputs
    if column not in df1.columns and column not in df2.columns:
        return pd.DataFrame(columns=["mean", "std"])  # empty

    s1 = df1[column].dropna().astype(str) if column in df1.columns else pd.Series([], dtype=str)
    s2 = df2[column].dropna().astype(str) if column in df2.columns else pd.Series([], dtype=str)
    if s1.empty and s2.empty:
        return pd.DataFrame(columns=["mean", "std"])  # empty

    # Aggregate reference frequencies for each group
    ref1 = char_frequencies(df1, column) if column in df1.columns else pd.Series(dtype=float)
    ref2 = char_frequencies(df2, column) if column in df2.columns else pd.Series(dtype=float)

    def _row_char_freq(text: str) -> pd.Series:
        if not text:
            return pd.Series(dtype=float)
        counts = pd.Series(list(text)).value_counts()
        return counts / counts.sum()

    diffs_list = []

    # df1 rows vs df2 aggregate
    for text in s1:
        row_freq = _row_char_freq(text)
        all_chars = row_freq.index.union(ref2.index)
        diff = (row_freq.reindex(all_chars, fill_value=0) - ref2.reindex(all_chars, fill_value=0)).abs()
        diffs_list.append(diff)

    # df2 rows vs df1 aggregate
    for text in s2:
        row_freq = _row_char_freq(text)
        all_chars = row_freq.index.union(ref1.index)
        diff = (row_freq.reindex(all_chars, fill_value=0) - ref1.reindex(all_chars, fill_value=0)).abs()
        diffs_list.append(diff)

    if not diffs_list:
        return pd.DataFrame(columns=["mean", "std"])  # empty

    # Align all diff vectors to the global union of characters
    all_chars_global = pd.Index([])
    for s in diffs_list:
        all_chars_global = all_chars_global.union(s.index)
    aligned = [s.reindex(all_chars_global, fill_value=0) for s in diffs_list]

    diffs_df = pd.DataFrame(aligned)
    mean = diffs_df.mean(axis=0)
    std = diffs_df.std(axis=0)
    out = pd.DataFrame({"mean": mean, "std": std}).sort_values("mean", ascending=False)
    if top_n is not None and top_n > 0:
        out = out.head(top_n)
    return out

def proportion_non_utf8_chars_used(df: pd.DataFrame, column: str = "thoughts") -> tuple[float, float]:
    """Mean and std of per-entry non-ASCII character proportions.

    For each row in the specified text column, compute:
        proportion = (# of characters with ord(c) > 127) / (total # of characters in the entry)
    Empty strings contribute 0.0. Missing values are ignored. Returns (0.0, 0.0) if no text.
    """
    if column not in df.columns or df.empty:
        return 0.0, 0.0
    series = df[column].dropna().astype(str)
    if series.empty:
        return 0.0, 0.0

    def _prop(text: str) -> float:
        if not text:
            return 0.0
        non_ascii = sum(1 for c in text if ord(c) > 127)
        return non_ascii / len(text)

    proportions = series.apply(_prop)
    return proportions.mean(), proportions.std()


def max_non_ascii_streak_stats(df: pd.DataFrame, column: str = "thoughts") -> tuple[float, float]:
    """Mean and std of the maximum consecutive non-ASCII streak per entry.

    For each row's text, find the longest run of consecutive characters with ord(c) > 127.
    Empty strings contribute 0.0. Missing values are ignored. Returns (0.0, 0.0) if no text.
    """
    if column not in df.columns or df.empty:
        return 0.0, 0.0
    series = df[column].dropna().astype(str)
    if series.empty:
        return 0.0, 0.0

    def _max_streak(text: str) -> int:
        max_run = 0
        cur = 0
        for ch in text:
            if ord(ch) > 127:
                cur += 1
                if cur > max_run:
                    max_run = cur
            else:
                cur = 0
        return max_run

    streaks = series.apply(_max_streak)
    return streaks.mean(), streaks.std()


@lru_cache(maxsize=1)
def _load_common_english_words(max_words: int = 50000) -> Set[str]:
    """Load a set of common English words.

    Tries to use the optional 'wordfreq' package (MIT licensed). If unavailable,
    falls back to a small built-in high-frequency list. Words are lowercased.
    """
    from wordfreq import top_n_list  # type: ignore
    words = {w.lower() for w in top_n_list("en", n=max_words)}
    return words

_WORD_RE = re.compile(r"[A-Za-z']+")

def proportion_oov_words(
    df: pd.DataFrame,
    column: str = "thoughts",
    word_set: Optional[Set[str]] = None,
    lowercase: bool = True,
) -> tuple[float, float]:
    """Mean and std of per-entry out-of-vocabulary (OOV) word proportions.

    For each row, tokenize using a simple regex (A-Z letters and apostrophes), optionally
    lowercase tokens, and compute:
        OOV proportion = (# tokens not in dictionary) / (total # tokens in that entry)
    Entries with zero tokens contribute 0.0. Missing values are ignored.
    Returns (0.0, 0.0) if the column is absent/empty.

    Parameters
    ----------
    df : DataFrame containing the text column.
    column : Column name (default 'thoughts').
    word_set : Optional set of valid English words. If None, a cached common
        English word list is loaded (prefers 'wordfreq', falls back otherwise).
    lowercase : Whether to lowercase tokens before dictionary lookup.
    """
    if column not in df.columns or df.empty:
        return 0.0, 0.0
    series = df[column].dropna().astype(str)
    if series.empty:
        return 0.0, 0.0

    if word_set is None:
        word_set = _load_common_english_words()

    def _oov_prop(text: str) -> float:
        total_tokens = 0
        oov_tokens = 0
        for match in _WORD_RE.finditer(text):
            token = match.group(0)
            token_cmp = token.lower() if lowercase else token
            total_tokens += 1
            if token_cmp not in word_set:
                oov_tokens += 1
        if total_tokens == 0:
            return 0.0
        return oov_tokens / total_tokens

    proportions = series.apply(_oov_prop)
    return proportions.mean(), proportions.std()


def top_common_word_count_stats(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    column: str = "thoughts",
    top_n: int = 10000,
    lowercase: bool = True,
    sigma_threshold: float = 2.0,
) -> pd.DataFrame:
    """Compute per-word count stats for the top-N most common words across two dataframes.

    Steps:
    - Tokenize each row's text from both dataframes using the same regex as OOV.
    - Build a combined frequency counter and select the top-N words.
    - For each selected word and for each dataframe separately, compute the mean and
      sample standard deviation of the per-row word counts.
        - Return a DataFrame indexed by word with columns:
        ['mean_df1', 'std_df1', 'mean_df2', 'std_df2', 'mean_diff', 'abs_mean_diff']
            sorted by 'abs_mean_diff' descending. The output is filtered to words where
            abs_mean_diff >= sigma_threshold * std_df2 (default sigma_threshold=2.0).

    If the column is missing or both inputs are empty, returns an empty DataFrame.
    """
    # Quick exits
    if (column not in df1.columns and column not in df2.columns) or (df1.empty and df2.empty):
        return pd.DataFrame(
            columns=[
                "mean_df1",
                "std_df1",
                "mean_df2",
                "std_df2",
                "mean_diff",
                "abs_mean_diff",
            ]
        )

    s1 = df1[column].dropna().astype(str) if column in df1.columns else pd.Series([], dtype=str)
    s2 = df2[column].dropna().astype(str) if column in df2.columns else pd.Series([], dtype=str)

    # Tokenize and build frequency counters
    def _tokenize(text: str) -> list[str]:
        toks = _WORD_RE.findall(text)
        if lowercase:
            toks = [t.lower() for t in toks]
        return toks

    counter1: Counter[str] = Counter()
    counter2: Counter[str] = Counter()

    # We won't store per-row tokens permanently to limit memory
    tokens_list_1 = []
    tokens_list_2 = []
    for text in s1:
        toks = _tokenize(text)
        tokens_list_1.append(toks)
        counter1.update(toks)
    for text in s2:
        toks = _tokenize(text)
        tokens_list_2.append(toks)
        counter2.update(toks)

    combined = counter1 + counter2
    if not combined:
        return pd.DataFrame(
            columns=[
                "mean_df1",
                "std_df1",
                "mean_df2",
                "std_df2",
                "mean_diff",
                "abs_mean_diff",
            ]
        )

    top_words = [w for w, _ in combined.most_common(top_n if top_n is not None else None)]
    top_set = set(top_words)

    # Aggregate sums and sums of squares of counts per word for each dataframe
    N1 = len(tokens_list_1)
    N2 = len(tokens_list_2)

    sums1 = defaultdict(int)
    sumsq1 = defaultdict(int)
    sums2 = defaultdict(int)
    sumsq2 = defaultdict(int)

    for toks in tokens_list_1:
        if not toks:
            continue
        c = Counter(toks)
        for w, cnt in c.items():
            if w in top_set:
                sums1[w] += cnt
                sumsq1[w] += cnt * cnt
    for toks in tokens_list_2:
        if not toks:
            continue
        c = Counter(toks)
        for w, cnt in c.items():
            if w in top_set:
                sums2[w] += cnt
                sumsq2[w] += cnt * cnt

    # Compute mean and sample std (ddof=1). If N <= 1, std=0.0
    def _stats(sums: dict[str, int], sumsq: dict[str, int], N: int) -> tuple[pd.Series, pd.Series]:
        means = {}
        stds = {}
        for w in top_words:
            s = sums.get(w, 0)
            ss = sumsq.get(w, 0)
            if N > 0:
                mean = s / N
            else:
                mean = 0.0
            if N > 1:
                var = max((ss - N * (mean * mean)) / (N - 1), 0.0)
                std = math.sqrt(var)
            else:
                std = 0.0
            means[w] = mean
            stds[w] = std
        return pd.Series(means), pd.Series(stds)

    mean1, std1 = _stats(sums1, sumsq1, N1)
    mean2, std2 = _stats(sums2, sumsq2, N2)

    out = pd.DataFrame(
        {
            "mean_df1": mean1,
            "std_df1": std1,
            "mean_df2": mean2,
            "std_df2": std2,
        }
    )
    out["mean_diff"] = out["mean_df1"] - out["mean_df2"]
    out["abs_mean_diff"] = out["mean_diff"].abs()
    # Apply sigma-based filtering using df2's std as requested
    if sigma_threshold is not None:
        thresh = sigma_threshold * out["std_df2"].fillna(0.0)
        out = out[out["abs_mean_diff"] >= thresh]
    out["z-score"] = out["abs_mean_diff"] / (out["std_df2"].replace(0.0, float("nan")))
    out = out.sort_values("z-score", ascending=False)
    return out


if __name__ == "__main__":
    data_path = "outputs/v4.csv"
    # data_path = "outputs/ministral_reasoning_delimiters.csv"
    try:
        safe_df, unsafe_df, benign_df = load_df(path=data_path)
    except (FileNotFoundError, KeyError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as e:
        print(f"Failed to load {data_path}: {e}")
        raise SystemExit(1) from e

    print("Mean different characters used in safe (mean, std):", different_chars_used(safe_df))
    print("Mean different characters used in unsafe (mean, std):", different_chars_used(unsafe_df))
    print("Mean different characters used in benign (mean, std):", different_chars_used(benign_df))

    print("Proportion of non-ASCII characters in safe (mean, std):", proportion_non_utf8_chars_used(safe_df))
    print("Proportion of non-ASCII characters in unsafe (mean, std):", proportion_non_utf8_chars_used(unsafe_df))
    print("Proportion of non-ASCII characters in benign (mean, std):", proportion_non_utf8_chars_used(benign_df))

    print("Max non-ASCII streak in safe (mean, std):", max_non_ascii_streak_stats(safe_df))
    print("Max non-ASCII streak in unsafe (mean, std):", max_non_ascii_streak_stats(unsafe_df))
    print("Max non-ASCII streak in benign (mean, std):", max_non_ascii_streak_stats(benign_df))

    print("Proportion OOV words in safe (mean, std):", proportion_oov_words(safe_df))
    print("Proportion OOV words in unsafe (mean, std):", proportion_oov_words(unsafe_df))
    print("Proportion OOV words in benign (mean, std):", proportion_oov_words(benign_df))

    pd.set_option('display.max_rows', None)
    stats = top_common_word_count_stats(unsafe_df, benign_df)
    print("Most common word comparison (jailbreak vs benign): ", stats.head(100))
    stats.to_csv("outputs/common_word_stats_jailbreak_vs_benign.csv")
    print("Most common word comparison (successful vs failed jailbreak): ", top_common_word_count_stats(unsafe_df, safe_df).head(100))
    # print("Biggest character frequency differences (top 100):")
    # print(biggest_char_frequency_differences(safe_df, unsafe_df).to_dict())