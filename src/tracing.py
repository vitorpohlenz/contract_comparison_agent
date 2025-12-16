from langfuse import Langfuse
import os
from uuid import uuid4
from dotenv import load_dotenv
load_dotenv()

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

def start_trace(name: str, input: dict, as_type: str = "span"):
    return langfuse.start_as_current_observation(
        name=name,
        input=input,
        as_type=as_type
    )
