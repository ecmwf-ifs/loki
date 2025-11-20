import requests
from gt4py.eve.codegen import format_source

from loki.backend.fgen import fgen
from loki.subroutine import Subroutine
from loki.batch import Transformation
from functools import partial
import logging


class AccGenAITransformation(Transformation):
    def __init__(self, api_endpoint: str):
        self.insert_openacc_directives = partial(
            insert_openacc_directives, api_endpoint=api_endpoint
        )

    def transform_subroutine(self, routine, **kwargs):
        return self.insert_openacc_directives(routine)


class PythonGenAITransformation(Transformation):
    def __init__(self, api_endpoint: str):
        self.fortran2python = partial(fortran2python, api_endpoint=api_endpoint)

    def transform_subroutine(self, routine, **kwargs):
        return self.fortran2python(routine)


class DaceGenAITransformation(Transformation):
    def __init__(self, api_endpoint: str):
        self.python2dace = partial(python2dace, api_endpoint=api_endpoint)

    def transform_file(self, code_string: str):
        return self.python2dace(code_string)


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

    data = {"model": "codestral", "prompt": prompt, "stream": False}

    response = requests.post(api_endpoint, json=data)
    return response.json()


def test_api():
    return query_llm("Tell me a joke.")


def fortran2python(routine: Subroutine, api_endpoint: str) -> str:
    """Return python code given a fortran subroutine

    Args:
        fsub (Subroutine): fortran subroutine

    Returns:
        str: python generated code
    """

    prompt = f"Translate this routine from fortran in python {fgen(routine)}"

    # Call API
    response_data = query_llm(prompt, api_endpoint)
    response_content = response_data["response"]
    
    print(response_content)

    # Cut python code in response
    python_code = response_content.split("```python")[1].split("```")[0]

    rendered_code = format_source("python", python_code)
    return rendered_code


def python2dace(python_code: str, api_endpoint: str):
    """Generate dace code from a python code.

    Args:
        python_code (str): _description_

    Returns:
        _type_: _description_
    """

    prompt = f"Optimize this routine with dace {python_code}"

    # Call API
    response_data = query_llm(prompt, api_endpoint)
    response_content = response_data["response"]

    # Cut python code in response
    dace_code = response_content.split("```python")[1].split("```")[0]

    rendered_code = format_source("python", dace_code)
    return rendered_code


def insert_openacc_directives(routine: Subroutine, api_endpoint):
    """Generate openacc directives for a fortran subroutine

    Args:
        routine (Subroutine): _description_

    Returns:
        _type_: _description_
    """

    prompt = f"Add OpenACC directives to this routine {fgen(routine)}"

    # Call API
    response_data = query_llm(prompt, api_endpoint)
    response_content = response_data["response"]

    # Cut response
    fortran_code = response_content.split("```fortran")[1].split("```")[0].split("fortran")[-1]

    transformed_routine = Subroutine.from_source(fortran_code, preprocess=True)
    return transformed_routine


def generic_prompt(routine: Subroutine, command: str, api_endpoint: str):
    """Builds a prompt based on a command + a routine

    Args:
        routine (Subroutine): routine to transform
        command (str): instruction to prompt

    Returns:
        _type_: _description_
    """

    prompt = command + fgen(routine)

    # Call API
    response_data = query_llm(prompt, api_endpoint)
    response_content = response_data["response"]

    # Cut response
    fortran_code = response_content.split("```")[1].split("```")[0].split("fortran")[-1]

    transformed_routine = Subroutine.from_source(fortran_code, preprocess=True)
    return transformed_routine
