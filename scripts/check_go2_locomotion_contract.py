"""Inspect the Go2 model and check a supplied Unitree locomotion export."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

JOINTS = {f"{leg}_{joint}_joint"
          for leg in ("FR", "FL", "RR", "RL")
          for joint in ("hip", "thigh", "calf")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--target-speed", type=float, default=1.60)
    args = parser.parse_args()
    root = ET.parse(args.urdf).getroot()
    movable = {joint.attrib["name"] for joint in root.findall("joint")
               if joint.get("type") in ("revolute", "continuous")}
    legs = {"FR": "rf", "FL": "lf", "RR": "rh", "RL": "lh"}
    joints = {"hip": "hip", "thigh": "upper_leg", "calf": "lower_leg"}
    aliases = {f"{leg}_{joint}_joint": f"{alias}_{joints[joint]}_joint"
               for leg, alias in legs.items() for joint in joints}
    mapping_by_name = {name: name if name in movable else aliases[name]
                       for name in sorted(JOINTS) if name in movable or aliases[name] in movable}
    report = {"urdf": str(args.urdf), "go2_joint_count": len(mapping_by_name),
              "joint_name_candidates": mapping_by_name,
              "missing_joints": sorted(JOINTS - mapping_by_name.keys()),
              "target_speed_mps": args.target_speed}
    if args.parameters:
        import yaml

        parameters = yaml.safe_load(args.parameters.read_text())
        mapping = parameters["joint_ids_map"]
        if len(mapping) != 12 or sorted(mapping) != list(range(12)):
            raise ValueError("Joint mapping must be a permutation of twelve indices.")
        report["policy_period_s"] = parameters["step_dt"]
        speed_range = parameters["commands"]["base_velocity"]["ranges"]["lin_vel_x"]
        report["forward_command_range_mps"] = speed_range
        report["target_within_command_range"] = speed_range[0] <= args.target_speed <= speed_range[1]
        report["observation_order"] = list(parameters["observations"])
        report["observation_width"] = sum(
            len(term["scale"]) * term["history_length"]
            for term in parameters["observations"].values()
        )
    if args.policy:
        if not args.parameters:
            parser.error("--policy requires --parameters")
        import onnxruntime

        session = onnxruntime.InferenceSession(str(args.policy), providers=["CPUExecutionProvider"])
        inputs, outputs = session.get_inputs(), session.get_outputs()
        report["onnx_input_shapes"] = [tensor.shape for tensor in inputs]
        report["onnx_output_shapes"] = [tensor.shape for tensor in outputs]
        report["onnx_shape_match"] = (
            len(inputs) == len(outputs) == 1
            and len(inputs[0].shape) == len(outputs[0].shape) == 2
            and inputs[0].shape[-1] == report["observation_width"]
            and outputs[0].shape[-1] == 12
        )
    print(json.dumps(report, indent=2))
    if report["missing_joints"] or report.get("onnx_shape_match") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
