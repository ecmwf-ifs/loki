from loki.backend.fgen import fgen
from loki.subroutine import Subroutine
from loki.batch import Transformation
from functools import partial
import logging
from ollama import ResponseError, ChatResponse, chat

class AccInsertionTransformation(Transformation):
    def __init__(self):
        self.insert_openacc_directives = insert_openacc_directives

    def transform_subroutine(self, routine, **kwargs):
        return self.insert_openacc_directives(routine)
    

def insert_openacc_directives(routine: Subroutine):
    """Generate openacc directives for a fortran subroutine

    Args:
        routine (Subroutine): _description_

    Returns:
        _type_: _description_
    """
    
    # Call API
    try: 
        response: ChatResponse = chat(model='codestral:22b', messages=[
        {
            'role': 'user',
            'content': f'Add OpenACC directives to this routine {fgen(routine)}',
        },
    ])
    except ResponseError as e:
        logging.error(f"Error {e.error}")
        
    logging.info(f"Response content {response.message.content}")
    print(response.message.content)
    content = response.message.content

    # Cut response
    if "```fortran" in content:
        fortran_code = content.split("```fortran")[1].split("```")[0]
    elif "```Fortran" in content:
        fortran_code = content.split("```Fortran")[1].split("```")[0]
    transformed_routine = Subroutine.from_source(fortran_code, preprocess=True)
    
    return transformed_routine

