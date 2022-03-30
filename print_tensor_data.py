import sys
import onnx
import onnx.numpy_helper
onnx_tensor = onnx.TensorProto.FromString(open(sys.argv[1], "rb").read())
np_array = onnx.numpy_helper.to_array(onnx_tensor)

print("shape: ", np_array.shape)
print("dtype: ", np_array.dtype)
print()
print(np_array)
