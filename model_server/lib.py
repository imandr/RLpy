import io, numpy as np


def to_bytes(s):
    if isinstance(s, str):
        s = s.encode("utf-8")
    return s

def to_str(s):
    if isinstance(s, memoryview):
        s = s.tobytes()
    if isinstance(s, bytes):
        s = s.decode("utf-8")
    return s

HeaderFormatVersion = "1.0"

def data_header(array):
    if isinstance(array, np.ndarray):
        return "#__header:version=%s;dtype=%s;shape=%s;size=%d#" % (HeaderFormatVersion, array.dtype.str, array.shape, array.nbytes)
    else:
        return "#__header:version=%s;dtype=bytes;size=%d#" % (HeaderFormatVersion, len(array))

def serialize_array(arr):
    return to_bytes(data_header(arr)) + arr.data

def deserialize_array(data):
    assert data[:10] == b"#__header:"
    iend = 10
    while iend < len(data) and data[iend:iend+1] != b'#':
        iend += 1
    if iend >= len(data):
        raise ValueError("Header parsing error: ", repr(data[:50]))
    hdr = to_str(data[10:iend])
    data = data[iend+1:]
    parts = hdr.split(";")
    shape = (-1,)
    dtype = "bytes"
    size = None
    for part in parts:
        name, value = part.split("=",1)
        if name == "shape":
            value = value[1:-1]     # remove outer parenthesis
            #print("shape: value:", value)
            shape = tuple([int(x) for x in value.split(",") if x.strip()])
        elif name == "dtype":
            dtype = value
        elif name == "size":
            size = int(value)
    #print("deserialize_array: size, shape, dtype:", size, shape, dtype)
    if size is not None:
        #print("    data:", len(data))
        tail = data[size:]
        out = data[:size]
        #print("    tail:", len(tail))
    else:
        tail = b''
    if dtype != "bytes":
        out = np.frombuffer(out, dtype).reshape(shape)
    return out, tail
	
def serialize(params):
    return b''.join([serialize_array(p) for p in params])

def deserialize(inp_bytes):
    if not inp_bytes:
        print("deserialize: empty inp_bytes:", inp_bytes)
        return None
    inp_view = memoryview(inp_bytes)
    params = []
    while len(inp_view):
        array, inp_view = deserialize_array(inp_view)
        params.append(array)
    return params

