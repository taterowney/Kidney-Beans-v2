from agent_boilerplate import Client
import json

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

def load_prompts_iter():
    # Simulate a dataloader that yields batches of data
    # Text of the prompt, whether it is benign or not
    for i in range(5):
        yield "Prompt", True

def save_responses(loader):
    client = Client("local", MODEL_NAME)
    all_responses = []
    for prompt, is_benign in loader:
        raw = client.get_response([{"role": "user", "message": prompt}])
        text = f"""<|USER|>{prompt}
        <|ASSISTANT|><think>{raw.choices[0].message.reasoning_content}</think>
        {raw.choices[0].message.content}"""
        all_responses.append({
            "text": text,
            "benign": is_benign
        })
    with open("data/responses.json", "w", encoding="utf-8") as f:
        json.dump(all_responses, f, indent=4, ensure_ascii=False)

def load_responses_iter():
    # A bit silly if we're working with a lot of data, but keeping it like this for now
    with open("data/responses.json", "r", encoding="utf-8") as f:
        all_responses = json.load(f)
    for response in all_responses:
        yield response["text"], response["benign"]

