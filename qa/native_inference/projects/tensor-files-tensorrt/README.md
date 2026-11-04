# Tensor upload and artifact QA (tensorrt)

Three independent static FP32 cases cover multipart upload, raw tensor decoding,
raw output registration, actual artifact downloads, and golden byte identities.
The affine model computes `input * 2 + 1`; outputs are manually specified in
`fixtures/cases.json` and `fixtures/golden-*.f32`.

This project is an acceptance fixture. Lepton controllers select matched build
and runtime tools in their owned source stage; authored files remain unchanged.
