## Extracting the bosh dataset
```bash
echo 'test'
```
## Set environment variable for model
```bash
```
## Training the model for object detection
```bash
nohup python /workspace/tensorflow/models/research/object_detection/train.py \
    --logtostderr \
    --train_dir=output/train/autti \
    --pipeline_config_path=config/autti_ssd_mobilenet_v1_coco.config &
```
## Spin up tensorboard
```bash
tensorboard --logdir output/train/autti --port 4567
```

## Evaluate model performance for given model
```bash
python /workspace/tensorflow/models/research/object_detection/eval.py \
    --logtostderr \
    --checkpoint_dir=output/train/autti \
    --eval_dir=output/eval \
    --pipeline_config_path=config/autti_ssd_mobilenet_v1_coco.config \
    --cpu_only
```


```bash
python /workspace/tensorflow/models/research/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path config/autti_ssd_mobilenet_v1_coco.config \
    --trained_checkpoint_prefix output/train/autti/model.ckpt-1847 \
    --output_directory output/frozen-model/autti-1847

python export_inference_graph \
    --input_type image_tensor \
    --pipeline_config_path path/to/ssd_inception_v2.config \
    --trained_checkpoint_prefix path/to/model.ckpt \
    --output_directory path/to/exported_model_directory
```

## Extract autti dataset for training
```bash
python extractor/extract_autti_dataset.py --label_map_path=data/crowdai/label_map_path.pbtxt \
        --input_path=data/crowdai/labels.csv        \
        --train_path=data/crowdai/train.record      \
        --val_path=data/crowdai/val.record          \
        --data_dir=data/crowdai/
```

## Sync code from local to ROS node
rsync -avn ~/workspace/sdc-object-detection/ desktop:/workspace/sdc-object-detection --exclude=data --exclude=images --exclude=models --exclude .idea
