import time, os.path, numpy as np, io, traceback, sys, requests
from lib import to_str, to_bytes, serialize, deserialize


class ModelClient(object):
    
    def __init__(self, model_name, url_head):
        self.URLHead = url_head
        self.ModelName = model_name
        
    def get(self):
        response = requests.get(self.URLHead + "/" + self.ModelName)
        response.raise_for_status()
        return deserialize(response.content)
    
    def update(self, params):
        response = requests.put(self.URLHead + "/" + self.ModelName, data=serialize(params))
        if response.status_code // 100 != 2:
            print(response)
            print(response.text)
            response.raise_for_status()
        return deserialize(response.content)
        
    def reset(self):
        response = requests.delete(self.URLHead + "/" + self.ModelName, data=serialize(params))
        response.raise_for_status()
