from loki.batch import Transformation
from functools import partial
from ollama import chat, ChatResponse, ResponseError
from gt4py.eve.codegen import format_source
import logging

class DaceOptimisationTransformation(Transformation):
    def __init__(self):
        self.python2dace = python2dace

    def transform_file(self, code_string: str):
        return self.python2dace(code_string)


def python2dace(python_code: str):
    """Generate dace code from a python code.

    Args:
        python_code (str): _description_

    Returns:
        _type_: _description_
    """
    
    # Call API
    try: 
        response: ChatResponse = chat(model='codestral:22b', messages=[
        {
            'role': 'user',
            'content': f'Optimize this python function with dace : {python_code}',
        },
    ])
    except ResponseError as e:
        logging.error(f"Error {e.error}")
        

    print("Dace Transformation output")   
    print(response.message.content)

    # Cut python code in response
    dace_code = response.message.content.split("```python")[1].split("```")[0]

    return format_source("python", dace_code)

