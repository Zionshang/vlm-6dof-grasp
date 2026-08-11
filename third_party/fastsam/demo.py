from ultralytics import FastSAM  # type: ignore

# Define an inference source
source = "/home/zishang/py-workspace/vlm-6dof-grasp/example_data/color.png"

# Create a FastSAM model
model = FastSAM("./weight/FastSAM-s.pt")  # or FastSAM-x.pt

# Run inference with bboxes prompt
bbox_results = model(source, bboxes=[439, 437, 524, 709], conf=0.5)
