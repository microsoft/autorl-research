import agentops
from agentlightning.reward import reward

@reward
def process_data(data):
    # Your function logic here
    processed_result = data.upper()
    # agentops.record(Events("Processed Data", result=processed_result)) # Optional: record specific events
    return 1.0

agentops.init()
process_data("hello")
