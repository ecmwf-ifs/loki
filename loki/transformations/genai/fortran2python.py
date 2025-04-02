from loki.backend.fgen import fgen
from loki.subroutine import Subroutine
from loki.batch import Transformation
from functools import partial
from ollama import ChatResponse, chat, ResponseError
from gt4py.eve.codegen import format_source
import logging

from loki.transformations.genai.genai import query_llm

class Fortran2PythonTransformation(Transformation):
    def __init__(self):
        self.fortran2python = fortran2python

    def transform_subroutine(self, routine, **kwargs):
        return self.fortran2python(routine)
    
def fortran2python(routine: Subroutine) -> str:
    """Return python code given a fortran subroutine

    Args:
        fsub (Subroutine): fortran subroutine

    Returns:
        str: python generated code
    """

    # Call API
    try: 
        response: ChatResponse = chat(model='codestral:22b', messages=[
        {
            'role': 'user',
            'content': f'Translate this routine from fortran in python {fgen(routine)}',
        },
    ])
    except ResponseError as e:
        logging.error(f"Error {e.error}")

    
    # Cut python code in response
    
    response_msg = response.message.content
    logging.info(f"Response content : {response_msg}")
    
    
    python_code = response_msg.split("```python")[1].split("```")[0]
    rendered_code = format_source("python", python_code)
    
    return rendered_code
