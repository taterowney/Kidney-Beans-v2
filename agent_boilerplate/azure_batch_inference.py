# Organization for batch inference API in Microsoft Azure's AI foundry
# Written by Tate Rowney (CMU L3 lab)

from openai import AzureOpenAI
from dotenv import load_dotenv
import os, json, time
from datetime import datetime
from pydantic import BaseModel, Field



load_dotenv()
if not os.getenv("AZURE_OPENAI_API_KEY"):
    raise EnvironmentError("AZURE_OPENAI_API_KEY environment variable not set.")
if not os.getenv("AZURE_OPENAI_ENDPOINT"):
    raise EnvironmentError("AZURE_OPENAI_ENDPOINT environment variable not set.")
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2025-03-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

def upload_data(model, messages_list, job_tags={}, **kwargs):
    """
    Creates an appropriately formatted jsonl file for batch inference at a new folder in batch-inference/jobs/(time)/
    Uploads it to Azure OpenAI for batch inference.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_path = f"batch-inference/jobs/{timestamp}/"
    os.makedirs(folder_path, exist_ok=True)

    jsonl_path = os.path.join(folder_path, "messages.jsonl")

    with open(jsonl_path, 'w') as f:
        for i, messages in enumerate(messages_list):
            for j in range(kwargs["n"]):
                j = json.dumps(
                    {
                            "custom_id": f"job-{timestamp}-{i}-{j}",
                            "method": "POST",
                            "url": "/chat/completions",
                            "body": {
                                "model": model,
                                "messages": messages,
                            }
                        }
                )
                j = j.encode('utf-16','surrogatepass').decode('utf-16')
                f.write(j)
                f.write("\n")

    response = client.files.create(
        file=open(jsonl_path, "rb"),
        purpose="batch",
        extra_body={"expires_after":{"seconds": 1209600, "anchor": "created_at"}}
    )
    print(f"File uploaded successfully. File ID: {response.id}")
    job_path = os.path.join(folder_path, "status.json")
    with open(job_path, 'w') as f:
        json.dump({
            "model": model,
            "file_id": response.id,
            "status": "pending",
            "batch_id": "",
            "messages": "",
            "output_file_id": "",
            "job_tags": job_tags,
        }, f, indent=4)

    return folder_path

def get_status(folder_path):
    """
    Checks the status of the batch inference job.
    """
    job_path = os.path.join(folder_path, "status.json")
    if not os.path.exists(job_path):
        raise FileNotFoundError(f"Job status file not found at {job_path}")

    with open(job_path, 'r') as f:
        job_info = json.load(f)

    return job_info

def upload_batch(folder_path):
    """
    Uploads the batch inference job to Azure OpenAI.
    """
    job_path = os.path.join(folder_path, "status.json")
    if not os.path.exists(job_path):
        raise FileNotFoundError(f"Job status file not found at {job_path}")

    with open(job_path, 'r') as f:
        job_info = json.load(f)
        # if "extra_body" not in kwargs:
        #     kwargs["extra_body"] = {}
        # kwargs["extra_body"]["output_expires_after"] = {"seconds": 1209600, "anchor": "created_at"}
        # print(kwargs)

    if not (input("You are about to upload a batch inference job. This may cost quite a bit. Are you sure? (y/N): ").strip().lower().startswith("y")):
        import sys
        sys.exit(0)

    batch_response = client.batches.create(
        input_file_id=job_info["file_id"],
        endpoint="/chat/completions",
        completion_window="24h",
        extra_body={"output_expires_after": {"seconds": 1209600, "anchor": "created_at"}},
    )
    print(f"Batch job created successfully. Batch ID: {batch_response.id}")
    job_info["status"] = "running"
    job_info["batch_id"] = batch_response.id
    with open(job_path, 'w') as f:
        json.dump(job_info, f, indent=4)

def get_batch_results(folder_path):
    job_info = get_status(folder_path)
    status = "validating"
    while status not in ("completed", "failed", "canceled"):
        batch_response = client.batches.retrieve(job_info["batch_id"])
        status = batch_response.status
        print(f"{datetime.now()} Batch Id: {job_info["batch_id"]},  Status: {status}")
        if status in ("completed", "failed", "canceled"):
            break
        time.sleep(120)

    if status == "failed":
        for error in batch_response.errors.data:
            print(f"Error code {error.code} Message {error.message}")
            job_info["status"] = "failed"
            job_info["messages"] += f"Error code {error.code} Message {error.message}\n\n"
    elif status == "completed":
        job_info["status"] = "completed"
        job_info["output_file_id"] = batch_response.output_file_id
        job_info["messages"] += f"Batch completed successfully. Output file ID: {batch_response.output_file_id}\n\n"
    elif status == "canceled":
        job_info["status"] = "canceled"
        job_info["messages"] += "Batch was canceled.\n\n"


    with open(os.path.join(folder_path, "status.json"), 'w') as f:
        json.dump(job_info, f, indent=4)

    return job_info

def download_results(folder_path):
    job_info = get_status(folder_path)
    if job_info["status"] != "completed":
        raise ValueError("Batch job has not completed successfully (yet).")

    output_file_id = job_info["output_file_id"]
    output = client.files.content(output_file_id).text.strip()
    # formatted_output = [json.loads(line) for line in output.split("\n") if line.strip()]
    formatted_output = {}
    for line in output.split("\n"):
        if line.strip():
            res = json.loads(line)
            query_number = int(res["custom_id"].split("-")[-2])
            if query_number not in formatted_output:
                formatted_output[query_number] = ChatCompletionResponse(choices=[])
            formatted_output[query_number].choices.append(
                ChatCompletionChoice(
                    finish_reason=res["response"]["body"]["choices"][0]["finish_reason"],
                    index=len(formatted_output[query_number].choices),
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=res["response"]["body"]["choices"][0]["message"]["content"],
                    )
                )
            )
    formatted_output = [formatted_output[i] for i in sorted(formatted_output.keys())]

    output_path = os.path.join(folder_path, "output.jsonl")
    with open(output_path, 'wb') as f:
        f.write(output.encode('utf-8'))

    return formatted_output

class ChatCompletionMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender, e.g., 'user', 'assistant', 'system'.")
    content: str = Field(..., description="Content of the message.")
    name: str | None = Field(None, description="Optional name of the message sender.")
    function_call: dict | None = Field(None, description="Optional function call information if applicable.")

class ChatCompletionChoice(BaseModel):
    finish_reason: str
    index: int
    message: ChatCompletionMessage

class ChatCompletionResponse(BaseModel):
    choices: list[ChatCompletionChoice]


def batch_inference(model, messages, job_tags={}, **kwargs):
    """ Performs batch inference using Azure OpenAI. Call again to resume in-progress jobs after termination, or to re-download completed jobs. """

    protected_tags = ["model", "file_id", "batch_id", "output_file_id", "status", "messages"]
    for tag in protected_tags:
        job_tags.pop(tag, None)

    if not os.path.exists(os.path.join(os.getcwd(), "batch-inference/jobs")) or not os.listdir(os.path.join(os.getcwd(), "batch-inference/jobs")):
        folder_path = upload_data(model, messages, **kwargs)
    else:
        jobs = sorted(os.listdir(os.path.join(os.getcwd(), "batch-inference/jobs")))[::-1]
        folder_path = None
        for job in jobs:
            status = get_status(os.path.join(os.getcwd(), "batch-inference/jobs", job))
            if (status["job_tags"] == job_tags or (not job_tags)) and (status["model"] == model):
                folder_path = os.path.join(os.getcwd(), "batch-inference/jobs", job)
                break
        if not folder_path:
            folder_path = upload_data(model, messages, job_tags=job_tags, **kwargs)

    status = get_status(folder_path)
    if status["status"] == "pending":
        upload_batch(folder_path)
        status = get_batch_results(folder_path)
    if status["status"] == "running" or status["status"] == "validating":
        status = get_batch_results(folder_path)
    if status["status"] == "completed":
        return download_results(folder_path)
    else:
        raise ValueError(f"Batch job failed with status: {status['status']}. Check the status file at {os.path.join(folder_path, 'status.json')} for more details.")
