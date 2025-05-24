import openai
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

class Client:
    def __init__(self, model_source, model_name, endpoint="http://localhost:8000/v1"):
        self.model_source = model_source
        self.model_name = model_name
        if model_source == "local":
            self.client = openai.OpenAI(
                api_key="EMPTY",
                base_url=endpoint
            )
            return
        load_dotenv()
        if model_source == "openai":
            self.client = openai.OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        elif model_source == "google":
            self.client = genai.Client(
                api_key=os.getenv("GENAI_API_KEY"),
            )

    def get_response(self, messages, raw=False, **kwargs):
        if self.model_source == "local" or self.model_source == "openai":
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs,
            )
            if raw:
                return res
            return res.choices[0].message.content
        elif self.model_source == "google":
            content = []
            for m in messages:
                if m["role"] in ["user", "model"]:
                    content.append(
                        types.Content(
                            role=m["role"],
                            parts=[types.Part.from_text(text=m["message"])]
                        )
                    )
                elif m["role"] == "system":
                    kwargs["config"] = types.GenerateContentConfig(
                        system_instruction=m["message"]
                    )

            # print(content)
            res = self.client.models.generate_content(
                model='gemini-2.0-flash-001',
                contents=content,
                **kwargs
            )
            if raw:
                return res
            return res.text