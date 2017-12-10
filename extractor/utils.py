from object_detection.utils import label_map_util
import os
import logging.config

import yaml

def setup_logging(
    default_path='config/log_config.yaml',
    default_level=logging.INFO,
    env_key='LOG_CFG'
):
    """Setup logging configuration

    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

def get_all_labels(label_map_path):
    label_map = label_map_util.load_labelmap(label_map_path)
    max_num_classes = max([item.id for item in label_map.item])
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes)

    labelToIdMap = {}

    for item in categories:
        labelToIdMap[item.get('name')] = item.get('id')

    return labelToIdMap