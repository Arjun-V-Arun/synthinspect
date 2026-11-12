"""Explanations for detections, scored against the true defect masks.

odam    gradient, instance-specific (one backward pass), PyTorch YOLO only
drise   black-box random masking, works on any runtime including INT8
metrics pointing game, energy in a region, deletion/insertion faithfulness
"""
