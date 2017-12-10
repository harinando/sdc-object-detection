import yaml
import os
import os
import logging.config
import yaml

def setup_logging(
    default_path='config/log_config.yaml',
    default_level=logging.INFO,
    env_key='LOG_CFG'
):
    """Setup logging configuration
       Usage: LOG_CFG=my_logging.yaml python my_server.py
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

def convert(in_filename):

    width = 800
    height = 600

    sim_data = yaml.load(open(in_filename, "r"))
    bosch_data = []

    for data in sim_data:
        sim_row = data.get('annotations', [])

        bosch_row = {
            'boxes': [],
            'path': os.path.join('.', data.get('filename', '').replace('\\', '/'))
        }

        for bbox_info in sim_row:
            bosh_bbox_info = {
                'label': bbox_info.get('class', None).encode(),
                'occluded': False,
                'x_max': float((bbox_info.get('xmin', 0) + bbox_info.get('x_width', 0))),
                'x_min': float(bbox_info.get('xmin', 0)),
                'y_max': float((bbox_info.get('ymin', 0) + bbox_info.get('y_height'))),
                'y_min': float(bbox_info.get('ymin', 0))
            }
            bosch_row.get('boxes', []).append(bosh_bbox_info)
        bosch_data.append(bosch_row)
    return bosch_data

def save(data, out_filename):
    with open(out_filename, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


if __name__ == '__main__':
    in_filename = os.path.join(os.path.join(os.path.dirname(__file__)), "../data/sim_data_large.yaml")
    out_filename = os.path.join(os.path.join(os.path.dirname(__file__)), '../data/sim_data_large_bosch_format.yml')

    print out_filename

    data = convert(in_filename)
    save(data, out_filename)