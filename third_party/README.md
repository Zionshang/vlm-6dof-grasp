# Third-party algorithms

Vendored external algorithm projects live here:

- `Fast-FoundationStereo/`: stereo depth
- `fastsam/`: prompted segmentation
- `EfficientSAM/`: lightweight prompted segmentation
- `economic_grasp/`: 6-DoF grasp generation
- `vlm/`: VLM detection and selection

Keep project-specific configuration, adapters and component registration in
`config/` and `core/components/`. Do not add application or robot-control
workflow code to these vendored directories.
