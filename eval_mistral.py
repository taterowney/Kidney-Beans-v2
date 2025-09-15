import random
import pandas as pd
import requests
import time
import re


def add_distractors(df):
    """Add distractor examples to the DataFrame."""
    with open("potentially_harmful_jailbreak_distractors.txt", "r", encoding="utf-8") as f:
        distractors = f.readlines()
    df["distractor"] = random.choices(distractors, k=len(df))
    return df

def mk_distractors(n: int = 1):
    """Create a distractors DataFrame and duplicate each prompt n times.

    Parameters:
        n (int): Number of times to duplicate each row (>=1).
    Returns:
        pd.DataFrame: DataFrame with a 'prompt' column (and any other retained cols) duplicated n times.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    distractors_df = pd.read_csv("advbench_educational_obfuscated_version_3.csv")
    distractors_df["prompt"] = distractors_df["instruction"] + distractors_df["example_prompt"] + distractors_df["guidance"]
    distractors_df["request"] = distractors_df["example_prompt"].apply(lambda x: re.findall('[A-Z][^A-Z]*', x.split(":")[1])[0]) # Original malicious requests got corrupted for some reason
    distractors_df = distractors_df.drop(columns=["instruction", "example_prompt", "guidance", "original_label"], errors="ignore")
    if n > 1:
        distractors_df = pd.concat([distractors_df]*n, ignore_index=True)
    return distractors_df

def batch_inference(df, server_url="http://localhost:8000/v1/completions", prompt_col="prompt", max_tokens=512):
    """
    Send prompts in df[prompt_col] to vLLM server and collect completions.
    Args:
        df (pd.DataFrame): DataFrame with prompts.
        server_url (str): vLLM server endpoint.
        prompt_col (str): Column name for prompts.
        max_tokens (int): Max tokens to generate.
    Returns:
        pd.DataFrame: DataFrame with completions added as 'completion' column.
    """
    completions = []
    for prompt in df[prompt_col]:
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.95,
            "n": 1
        }
        try:
            response = requests.post(server_url, json=payload)
            response.raise_for_status()
            result = response.json()
            # vLLM returns 'choices' list with 'text' field
            completion = result["choices"][0]["text"] if "choices" in result and result["choices"] else ""
        except Exception as e:
            completion = f"ERROR: {e}"
        completions.append(completion)
    df["answer"] = completions
    print(df[[prompt_col, "answer"]].head())
    df.to_csv(f"{time.time()}.csv", index=False)

def run_jailbreaks():
    # df = load_advbench()
    # df = add_distractors(df)

    # df["prompt"] = df.apply(lambda row: row["distractor"].format(row["instruction"]), axis=1)
    df = mk_distractors(n=10)
    print(df.head())

    # Run batch inference
    batch_inference(df)