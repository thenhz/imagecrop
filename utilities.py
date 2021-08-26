from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
import numpy as np
from six import BytesIO

from object_detection.utils import label_map_util





def get_keypoint_tuples(eval_config):
    """Return a tuple list of keypoint edges from the eval config.

    Args:
      eval_config: an eval config containing the keypoint edges

    Returns:
      a list of edge tuples, each in the format (start, end)
    """
    tuple_list = []
    kp_list = eval_config.keypoint_edge
    for edge in kp_list:
        tuple_list.append((edge.start, edge.end))
    return tuple_list


def load_label_map(configs):
    label_map_path = configs['eval_input_config'].label_map_path
    label_map = label_map_util.load_labelmap(
        'models/research/object_detection/data/mscoco_complete_label_map.pbtxt')
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=label_map_util.get_max_label_map_index(label_map),
        use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    label_map_dict = label_map_util.get_label_map_dict(
        label_map, use_display_name=True)
    return label_map_dict, label_map_path, category_index


def get_heatmap(predictions_dict, class_name, label_map_dict, label_id_offset):
    """Grabs class center logits and apply inverse logit transform.

    Args:
      predictions_dict: dictionary of tensors containing a `object_center`
        field of shape [1, heatmap_width, heatmap_height, num_classes]
      class_name: string name of category (e.g., `horse`)

    Returns:
      heatmap: 2d Tensor heatmap representing heatmap of centers for a given class
        (For CenterNet, this is 128x128 or 256x256) with values in [0,1]
    """
    class_index = label_map_dict[class_name]
    class_center_logits = predictions_dict['object_center'][0]
    class_center_logits = class_center_logits[0][
        :, :, class_index - label_id_offset]
    heatmap = tf.exp(class_center_logits) / (tf.exp(class_center_logits) + 1)
    return heatmap


def unpad_heatmap(heatmap, image_np, detection_model, shapes):
    """Reshapes/unpads heatmap appropriately.

    Reshapes/unpads heatmap appropriately to match image_np.

    Args:
      heatmap: Output of `get_heatmap`, a 2d Tensor
      image_np: uint8 numpy array with shape (img_height, img_width, 3).  Note
        that due to padding, the relationship between img_height and img_width
        might not be a simple scaling.

    Returns:
      resized_heatmap_unpadded: a resized heatmap (2d Tensor) that is the same
        size as `image_np`
    """
    heatmap = tf.tile(tf.expand_dims(heatmap, 2), [1, 1, 3]) * 255
    pre_strided_size = detection_model._stride * heatmap.shape[0]
    resized_heatmap = tf.image.resize(
        heatmap, [pre_strided_size, pre_strided_size],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    resized_heatmap_unpadded = tf.slice(
        resized_heatmap, begin=[0, 0, 0], size=shapes)
    return tf.image.resize(
        resized_heatmap_unpadded,
        [image_np.shape[0], image_np.shape[1]],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[:, :, 0]
