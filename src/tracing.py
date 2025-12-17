from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler
import os
from contextlib import contextmanager
from dotenv import load_dotenv
load_dotenv()

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

@contextmanager
def start_trace(name: str, input: dict):
    """
    Context manager for creating Langfuse traces with proper input/output handling.
    Uses start_as_current_observation with as_type="trace" (modern Langfuse API).
    
    Args:
        name: Name of the trace
        input: Input data (will be serialized)
    
    Yields:
        Trace object that can be used to create child spans and set output
    """
    langfuse_client = get_client()
    with langfuse_client.start_as_current_observation(
        name=name,
        input=_serialize_input(input),
        as_type="trace"
    ) as trace:
        try:
            yield trace
        finally:
            pass

@contextmanager
def start_span(name: str, input: dict):
    """
    Context manager for creating child spans within an existing trace.
    Uses start_as_current_observation with as_type="span" which automatically attaches to the current trace context.
    
    Args:
        name: Name of the span
        input: Input data (will be serialized)
    
    Yields:
        Span object that can be used to set output
    """
    langfuse_client = get_client()
    with langfuse_client.start_as_current_observation(
        name=name,
        input=_serialize_input(input),
        as_type="span"
    ) as span:
        try:
            yield span
        finally:
            pass

def _serialize_input(input_data):
    """Serialize input data to be JSON-serializable for Langfuse."""
    if isinstance(input_data, dict):
        return {k: _serialize_value(v) for k, v in input_data.items()}
    return _serialize_value(input_data)

def _serialize_value(value):
    """Serialize a single value to be JSON-serializable."""
    if hasattr(value, 'model_dump'):
        # Pydantic model
        return value.model_dump()
    elif hasattr(value, 'dict'):
        # Pydantic model (older versions)
        return value.dict()
    elif isinstance(value, (str, int, float, bool, type(None))):
        return value
    elif isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    elif isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    else:
        return str(value)

def get_langfuse_callback_handler(trace_name: str = None):
    """
    Get a Langfuse callback handler for LangChain integration.
    This will automatically capture all LangChain LLM calls as generations within the current trace.
    
    Args:
        trace_name: Optional name for the trace. If None, will use the current trace context.
    
    Returns:
        CallbackHandler instance
    """
    # CallbackHandler automatically uses environment variables if not provided
    # It will automatically attach to the current trace context
    return CallbackHandler()
