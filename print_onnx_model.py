import sys
import onnx

m = onnx.load(sys.argv[1])

def print_ops():
    ops = list(map(lambda n: n.op_type, m.graph.node))
    print(",".join(ops))

def print_inputs():
    print(m.graph.input)

def print_outputs():
    print(m.graph.output)

if "--ops" in sys.argv:
    print_ops()
elif "--inputs" in sys.argv:
    print_inputs()
elif "--outputs" in sys.argv:
    print_outputs()
else:
    print(m)
