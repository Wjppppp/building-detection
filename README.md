# building-detection

Detect OSM missing buildings

To build ohsome2label docker image or prepare training sets from OSM and satellite images, please refer to *[./data_prepare](https://github.com/Wjppppp/building-detection/tree/main/data_preparing)*.



## Folder Tree

```
building-detection/
├── data_preparing/
│   ├── label/
│   ├── Dockerfile
│   └── README
└── object_detection/
    ├── cocoapi/
    ├── models/
    ├── workspace/
    │   └── training_demo/
    │       ├── annotations/
    │       ├── exported-models/
    │       │   ├── label_map.pbtxt
    │       │   ├── train.record
    │       │   └── valid.record          
    │       ├── images/
    │       │   ├── test/
    │       │   └── train/
    │       ├── models/
    │       ├── pre-trained-models/
    │       ├── exporter_main_v2.py
    │       ├── model_main_tf2.py
    │       └── walkthrough.ipynb
    ├── Dockerfile
    └── README
```