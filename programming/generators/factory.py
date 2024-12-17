from .py_generate import PyGenerator
from .model import CodeLlama, ModelBase

def model_factory(model_name: str, port: str = "", key: str = "") -> ModelBase:
    if model_name == "codellama/CodeLlama-7b-Python-hf":
        return CodeLlama(port)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
