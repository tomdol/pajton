import numpy as np
import onnxruntime as ort

def _to_dtype(t):
    if "int8" in t:
        return np.int8
    if "int32" in t:
        return np.int32
    if "int64" in t:
        return np.int64
    if "float32" in t:
        return np.float32
    if "float" in t:
        return np.float32
    if "float64" in t:
        return np.float64
    else:
        raise Exception("Unknown data type: " + t)

def _randomize_inputs(session_inputs):
    random_inputs = []
    for input in session_inputs:
        t = _to_dtype(input.type)
        if np.issubdtype(t, np.integer):
            i = {input.name: np.random.randint(low=0, high=1000, size=input.shape, dtype=t)}
        else:
            i = {input.name: np.random.rand(*(input.shape)).astype(t)}
        random_inputs.append(i[input.name])

    return random_inputs

def infer_onnxrt_random_inputs(model_path):
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_names = [i.name for i in session.get_inputs()]
    random_inputs = _randomize_inputs(session.get_inputs())
    session_inputs = {input_name: random_data for input_name, random_data in zip(input_names, random_inputs)}
    res = session.run(None, session_inputs)
    output_names = [o.name for o in session.get_outputs()]
    return {name: output for name, output in zip(output_names, res)}