from roboflow import Roboflow

rf = Roboflow(api_key="028QZz9LBPRsKJXDFnY7")
project = rf.workspace("roboflow-100").project("road-signs-6ih4y")
dataset = project.version(1).download("yolov8")
