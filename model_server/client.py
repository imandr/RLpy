import time, os.path, numpy as np, io, traceback, sys
from socket import socket, AF_INET, SOCK_STREAM, SOL_SOCKET, SO_REUSEADDR

def to_bytes(s):
    if isinstance(s, str):
        s = s.encode("utf-8")
    return s

def to_str(s):
    if isinstance(s, bytes):
        s = s.decode("utf-8")
    return s

class ModelKeeperClient(object):
    
    def __init__(self, project_name, addr)
        self.ServerAddr = addr
        self.ProjectName = project_name
        
    def communicate(self, command, data=b''):
        sock = socket(AF_INET, SOCK_STREAM)
        sock.connect(self.ServerAddr)
        n = len(data)
        sock.sendall(to_bytes(f"{self.ProjectName}:{command}:{n}:"))
        if data:
            sock.sendall(data)
        fragments = []
        eof = False
        while not eof:
            fragment = sock.recv(128*1024)
            if not fragment:
                eof = True
            fragments.append(fragment)
        return b''.join(fragments)

    @staticmethod
    def deserialize(serialized_params):
        buf = io.BytesIO(serialized_params)
        loaded = np.load(buf)
        buf.close()
        return [loaded[k] for k in loaded]

    @staticmethod
    def serialize(params):
        if not params:
            return b''
        buf = io.BytesIO()
        np.savez(buf, *params)
        return buf.getvalue()

    def get_params(self):
        return self.deserialize(self.communicate("get"))
    
    def update_params(self, params):
        return self.deserialize(self.communicate("update", self.serialize(params)))
        
    def reset(self):
        return self.deserialize(self.communicate("reset"))
    