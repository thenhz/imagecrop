import os 
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from six import BytesIO

class CenterNet:

    def __init__(
        self, 
        pipeline_config = 'models/research/object_detection/configs/tf2/center_net_hourglass104_512x512_coco17_tpu-8.config', 
        model_dir = 'models/research/object_detection/test_data/checkpoint/'
        ):
        
        # Load pipeline config and build a detection model
        self.configs = config_util.get_configs_from_pipeline_file(pipeline_config)
        model_config = self.configs['model']
        self.detection_model = model_builder.build(
            model_config=model_config, is_training=False)

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(
            model=self.detection_model)
        ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()

    
        def get_model_detection_function(model):
        # """Get a tf.function for detection."""

            @tf.function
            def detect_fn(image):
                """Detect objects in image."""

                image, shapes = model.preprocess(image)
                prediction_dict = model.predict(image, shapes)
                detections = model.postprocess(prediction_dict, shapes)

                return detections, prediction_dict, tf.reshape(shapes, [-1])

            return detect_fn

        self.model_detection_function = get_model_detection_function(self.detection_model)

        self.classes_weight = {
            "kite": 1,
            "person": 10,
            "surfboard":1
        }

    def load_image_into_numpy_array(self, path):
        """Load an image from file into a numpy array.

        Puts image into numpy array to feed into tensorflow graph.
        Note that by convention we put it into a numpy array with shape
        (height, width, channels), where channels=3 for RGB.

        Args:
        path: the file path to the image

        Returns:
        uint8 numpy array with shape (img_height, img_width, 3)
        """
        img_data = tf.io.gfile.GFile(path, 'rb').read()
        image = Image.open(BytesIO(img_data))
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def get_label_map_data(self):
        label_map_dict, label_map_path, category_index = self.load_label_map()
        return label_map_dict, label_map_path, category_index
    
    def get_detections(self, image_path):
        image_np = self.load_image_into_numpy_array(image_path)
        im_width = image_np.shape[0]
        im_height = image_np.shape[1]
        label_map_dict, label_map_path, category_index = self.load_label_map()
        input_tensor = tf.convert_to_tensor(
            np.expand_dims(image_np, 0), dtype=tf.float32)
        detections, predictions_dict, shapes = self.model_detection_function(input_tensor)

        classes_detected = (detections['detection_classes'][0].numpy() + 1).astype(int)
        detections['detection_classes_name'] = []
        for idx, bbox in enumerate(detections['detection_boxes'][0].numpy()):  
            detections['detection_classes_name'].append(category_index[classes_detected[idx]]['name'])
        
        np_im = np.zeros((im_width,im_height))
        scores = detections['detection_scores'][0].numpy()
        for idx, bbox in enumerate(detections['detection_boxes'][0].numpy()):  
            if scores[idx] > 0.30:
                ymin, xmin, ymax, xmax = tuple(bbox.tolist())
                (left, right, top, bottom) = (xmin * im_height, xmax * im_height, ymin * im_width, ymax * im_width)
                if detections['detection_classes_name'][idx] in self.classes_weight:
                    obj_score = self.classes_weight[detections['detection_classes_name'][idx]]#classes_weight[category_index[classes_detected[idx]]['name']]
                    np_im[int(top):int(bottom),int(left):int(right)] = obj_score
        
        return np_im

    def run_inference(self, image_np):
        # Things to try:
        # Flip horizontally
        # image_np = np.fliplr(image_np).copy()

        # Convert image to grayscale
        # image_np = np.tile(
        #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

        input_tensor = tf.convert_to_tensor(
            np.expand_dims(image_np, 0), dtype=tf.float32)
        detections, predictions_dict, shapes = self.model_detection_function(input_tensor)
        return detections, predictions_dict, shapes

    def show_detections(self,image_np_with_detections, detections):
        label_id_offset = 1

        label_map_dict, label_map_path, category_index = self.get_label_map_data()
        # Use keypoints if available in detections
        keypoints, keypoint_scores = None, None
        if 'detection_keypoints' in detections:
            keypoints = detections['detection_keypoints'][0].numpy()
            keypoint_scores = detections['detection_keypoint_scores'][0].numpy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'][0].numpy(),
            (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
            detections['detection_scores'][0].numpy(),
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False,
            keypoints=keypoints,
            keypoint_scores=keypoint_scores,
            keypoint_edges=self.get_keypoint_tuples(self.configs['eval_config']))

        return image_np_with_detections
        #plt.figure(figsize=(12,16))
        #plt.imshow(image_np_with_detections)
        #plt.show()
    
    def get_keypoint_tuples(self, eval_config):
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


    def load_label_map(self):
        label_map_path = self.configs['eval_input_config'].label_map_path
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


    def get_heatmap(self, predictions_dict, class_name, label_map_dict, label_id_offset):
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


    def unpad_heatmap(self, heatmap, image_np, shapes):
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
        pre_strided_size = self.detection_model._stride * heatmap.shape[0]
        resized_heatmap = tf.image.resize(
            heatmap, [pre_strided_size, pre_strided_size],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        resized_heatmap_unpadded = tf.slice(
            resized_heatmap, begin=[0, 0, 0], size=shapes)
        return tf.image.resize(
            resized_heatmap_unpadded,
            [image_np.shape[0], image_np.shape[1]],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[:, :, 0]
