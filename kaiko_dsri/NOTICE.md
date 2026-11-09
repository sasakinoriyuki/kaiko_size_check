# YOLOv8-DSRI inference code

This directory contains an inference-only extraction of Ultralytics / YOLOv8-DSRI code, under GNU AGPL version 3. The package name was changed to `kaiko_dsri`; training, configuration writers, download helpers and package initialisation were omitted. A fixed tensor rearrangement replaces the single required einops operation.

Source: https://github.com/BillTheDing/YOLOv8-DSRI/tree/18085596e7e950cac86b377807f05046c6b9a2cf/yolov8-main

Original Ultralytics authors and the YOLOv8-DSRI authors retain their copyrights. See LICENSE.txt for the GNU Affero General Public License. The original code and the modified inference code are available with this application's corresponding public source. This is an externally pretrained model, not a model trained solely on this channel's images.
