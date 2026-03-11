This repository was created for the course Edge Computing and Internet of Things - DT8040 given by Halmstad University, Sweden.

# Instructions

1. create a directory named "onnx_models" and place the models with ".onnx" extension there.
2. create a decoder for the model you want to test if it does not already exist in "/models".

## Real-time inference

3. run webcam.py to preview a model live before benchmarking.
4. run run_webcam_benchmark.py to benchmark a model

## Batch inference
3. create a new directory "batchInference/annotations" and place the annotations from the COCO dataset in .json format inside.
4. create a new directory "batchInference/val2017" and place all validation images from the COCO dataset as .jpg.
5. run the NAME_OF_MODEL_Dataset_test.py to benchmark a model.