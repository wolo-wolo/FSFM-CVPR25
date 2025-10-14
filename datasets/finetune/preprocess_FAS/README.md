We follow the [flip](https://github.com/koushiksrivats/FLIP/blob/main/docs/datasets.md) and [few_shot_fas](https://github.com/hhsinping/few_shot_fas) to prepare the face anti-spoofing datasets. 
We provide this preprocessing pipeline to facilitate the reproduction.

1) install mtcnn for face detection.
    ```bash
    cd /prerocess/tools
    git clone https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection.git
    ```

2) Modify the following code in ```/tools/mxnet_mtcnn_face_detection/mtcnn_detector.py``` to be compatible with Python 3.X version.

    ```
    Replace:
        from itertools import izip
        from helper import nms, adjust_input, generate_bbox, detect_first_stage_warpper
    with:
        from six.moves import zip as izip
        from .helper import nms, adjust_input, generate_bbox, detect_first_stage_warpper
    ```
    
    ```
    Replace:
        from_shape_points = from_shape.reshape(from_shape.shape[0] / 2, 2)
        to_shape_points = to_shape.reshape(to_shape.shape[0] / 2, 2)
        for k in range(len(p)/2):
        for i in range(len(shape)/2):
    with:
        from_shape_points = from_shape.reshape(from_shape.shape[0] // 2, 2)
        to_shape_points = to_shape.reshape(to_shape.shape[0] // 2, 2)
        for k in range(len(p) // 2):
        for i in range(len(shape) //  2):
    ```

3) Download the following data and put it in ```finetune_datasets/face_anti_spoofing```
    ```bash
    cd finetune_datasets/face_anti_spoofing
    wget https://www.dropbox.com/s/2mxh5r8hf0m8m1n/data.tgz
    tar zxvf data.tgz
    ```
    you will get ```finetune_datasets/face_anti_spoofing/data```


4) Then, run  ```preprocess_{MCIO/WCS}.py```