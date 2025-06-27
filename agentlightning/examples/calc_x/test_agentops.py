import agentops
from agentops.sdk.decorators import operation

@operation
def process_data(data):
    # Your function logic here
    processed_result = data.upper()
    # agentops.record(Events("Processed Data", result=processed_result)) # Optional: record specific events
    return processed_result

agentops.init()
process_data("hello")
