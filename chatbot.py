import os
import openai
from io import StringIO

from dotenv import load_dotenv, find_dotenv
load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")


def get_response(prompt: str) -> str:
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].text


def get_cheaper_response(messages: list[str]) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,)

    return response['choices'][0]['message']['content']
