import onnxruntime as ort

sess = ort.InferenceSession("yolo11n-pose.onnx")

print("输入：")
for i in sess.get_inputs():
    print(i.name, i.shape)

print("输出：")
for o in sess.get_outputs():
    print(o.name, o.shape)