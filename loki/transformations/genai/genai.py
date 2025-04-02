import requests
from ollama import chat, ChatResponse

from loki.backend.fgen import fgen
from loki.subroutine import Subroutine
from loki.batch import Transformation

class PromptedGenAITransformation(Transformation):
    def __init__(self, command):
        self.command = command

    def transform_subroutine(self, routine, **kwargs):
        return generic_prompt(routine, self.command)


def query_llm(prompt: str, api_endpoint: str):
    """Query llm listening on an api endpoint with a prompt

    Args:
        prompt (str): prompt to provide
        api_endpoint (str): api_endpoint of llm

    Returns:
        _type_: _description_
    """

    data = {'model': 'codestral', 'prompt': prompt, 'stream': False}

    response = requests.post(api_endpoint, json=data)
    return response.json()


def test_api():
    return query_llm("Tell me a joke.")

def generic_prompt(routine: Subroutine, command: str, api_endpoint: str):
    """Builds a prompt based on a command + a routine

    Args:
        routine (Subroutine): routine to transform
        command (str): instruction to prompt

    Returns:
        _type_: _description_
    """

    prompt = f"{command} :" + fgen(routine)

    # Call API
    response_data = query_llm(prompt, api_endpoint)
    response_content = response_data["response"]

    # Cut response
    raw_answer = response_content.split("```")[1]

    return raw_answer

if __name__ == "__main__":
    
#     response: ChatResponse = chat(model='codestral:22b', messages=[
#   {
#     'role': 'user',
#     'content': 'Why is the sky blue?',
#   },
#     ])
#     print(response['message']['content'])
#     # or access fields directly from the response object
#     print(response.message.content)
    
    response: ChatResponse = chat(model='codestral:22b', messages=[
  {
    'role': 'user',
    'content': 'Generate a matrix multiplication in python',
  },
    ])
    print(response['message']['content'])
    # or access fields directly from the response object
    print(response.message.content)

