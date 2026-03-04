

inference_store = []

def save_inference(data):
    inference_store.append(data)
    return True

def get_all_inferences():
    return inference_store