# Free model and source

The automatic 1–5 instar measurement mode uses the externally pretrained YOLOv8-DSRI silkworm instance segmentation checkpoint. This is not described as training solely on 蚕CHANNEL images.

- Upstream: https://github.com/BillTheDing/YOLOv8-DSRI/tree/18085596e7e950cac86b377807f05046c6b9a2cf/yolov8-main
- Pinned checkpoint SHA-256: `ddd6e7d9fb2911588f4e6d13fc78a2680f585c95ec13c1e651ba6b3197a8e217`
- License: GNU AGPL-3.0; full text in `kaiko_dsri/LICENSE.txt`.
- Modified inference source is included in `kaiko_dsri/` in this repository and in the program download. Namespace changes, omitted training/settings writers and the tensor rearrangement replacement are described in `kaiko_dsri/NOTICE.md`.
- The application combined with this inference code is distributed under the AGPL-3.0 terms. The original MIT notice in `LICENSE` is retained for the independently authored parts that were previously MIT-licensed.

Source for the distributed program is available on the `parsonal` and `school` branches of https://github.com/sasakinoriyuki/kaiko_size_check . Users may inspect, modify and redistribute under the applicable free licenses. No paid inference API, account, subscription or enterprise license is used or required for this source-distributed application.

The channel's own texture model is used as a foreground veto in small-body search. It does not supply instance identity, hidden endpoints or physical scale. The older self-trained modes remain available.
