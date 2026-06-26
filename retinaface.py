import base64
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from nets.facenet import Facenet
from nets_retinaface.retinaface import RetinaFace
from utils.anchors import Anchors
from utils.config import cfg_mnet, cfg_re50
from utils.utils import (Alignment_1, compare_faces, letterbox_image,
                         preprocess_input)
from utils.utils_bbox import (decode, decode_landm, non_max_suppression,
                              retinaface_correct_boxes)

from databases.redis_db import RedisStorage
from utils.helpers import base64_to_numpy_image


def cv2ImgAddText(img, label, left, top, textColor=(255, 255, 255)):
    img = Image.fromarray(np.uint8(img))
    font = ImageFont.truetype(font='model_data/simhei.ttf', size=20)
    draw = ImageDraw.Draw(img)
    label = label.encode('utf-8')
    draw.text((left, top), str(label, 'UTF-8'), fill=textColor, font=font)
    return np.asarray(img)


class Retinaface(object):
    _defaults = {
        "retinaface_model_path" : 'model_data/Retinaface_mobilenet0.25.pth',
        "retinaface_backbone"   : "mobilenet",
        "confidence"            : 0.5,
        "nms_iou"               : 0.3,
        "retinaface_input_shape": [640, 640, 3],
        "letterbox_image"       : True,
        "facenet_model_path"    : 'model_data/facenet_mobilenet.pth',
        "facenet_backbone"      : "mobilenet",
        "facenet_input_shape"   : [160, 160, 3],
        "facenet_threhold"      : 1.05,
        "cuda"                  : False
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        return "Unrecognized attribute name '" + n + "'"

    def __init__(self, encoding=0, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.redis_storage = RedisStorage()

        if self.retinaface_backbone == "mobilenet":
            self.cfg = cfg_mnet
        else:
            self.cfg = cfg_re50

        self.anchors = Anchors(self.cfg, image_size=(self.retinaface_input_shape[0], self.retinaface_input_shape[1])).get_anchors()
        self.generate()

        try:
            self.redis_storage.load_face_data("mobilenet")
            self.known_face_encodings = self.redis_storage.known_face_encodings
            self.known_face_names     = self.redis_storage.known_face_names
        except:
            if not encoding:
                print("载入已有人脸特征失败，请检查model_data下面是否生成了相关的人脸特征文件。")
            pass

    def generate(self):
        self.net     = RetinaFace(cfg=self.cfg, phase='eval', pre_train=False).eval()
        self.facenet = Facenet(backbone=self.facenet_backbone, mode="predict").eval()

        print('Loading weights into state dict...')
        state_dict = torch.load(self.retinaface_model_path, map_location=torch.device('cpu'))
        self.net.load_state_dict(state_dict)

        state_dict = torch.load(self.facenet_model_path, map_location=torch.device('cpu'))
        self.facenet.load_state_dict(state_dict, strict=False)

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
            self.facenet = nn.DataParallel(self.facenet)
            self.facenet = self.facenet.cuda()
        print('Finished!')

    def reload_face_feature(self, backbone):
        self.redis_storage.load_face_data(backbone)
        self.known_face_encodings = self.redis_storage.known_face_encodings
        self.known_face_names     = self.redis_storage.known_face_names

    # ------------------------------------------------------------------
    # Private pipeline helpers
    # ------------------------------------------------------------------

    def _compute_scales(self, image):
        """Return (im_height, im_width, scale, scale_for_landmarks) for coordinate conversion."""
        h, w = image.shape[:2]
        scale = [w, h, w, h]
        scale_for_landmarks = [w, h, w, h, w, h, w, h, w, h]
        return h, w, scale, scale_for_landmarks

    def _prepare_detection_input(self, image, im_height, im_width):
        """Letterbox + normalize image and fetch anchors; return (image_tensor, anchors)."""
        if self.letterbox_image:
            image   = letterbox_image(image, [self.retinaface_input_shape[1], self.retinaface_input_shape[0]])
            anchors = self.anchors
        else:
            anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

        image_tensor = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)
        if self.cuda:
            image_tensor = image_tensor.cuda()
            anchors      = anchors.cuda()
        return image_tensor, anchors

    def _run_detector(self, image_tensor, anchors, im_height, im_width, scale, scale_for_landmarks):
        """Run RetinaFace forward pass and return scaled, NMS-filtered detections."""
        with torch.no_grad():
            loc, conf, landms = self.net(image_tensor)
            boxes   = decode(loc.data.squeeze(0), anchors, self.cfg['variance'])
            conf    = conf.data.squeeze(0)[:, 1:2]
            landms  = decode_landm(landms.data.squeeze(0), anchors, self.cfg['variance'])
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

        if len(boxes_conf_landms) == 0:
            return boxes_conf_landms

        if self.letterbox_image:
            boxes_conf_landms = retinaface_correct_boxes(
                boxes_conf_landms,
                np.array([self.retinaface_input_shape[0], self.retinaface_input_shape[1]]),
                np.array([im_height, im_width])
            )

        boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
        boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks
        return boxes_conf_landms

    def _encode_face_crop(self, box, old_image):
        """Crop, align, and embed a single detected face; return 128-d numpy vector."""
        box      = np.maximum(box, 0)
        crop_img = np.array(old_image)[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        landmark = np.reshape(box[5:], (5, 2)) - np.array([int(box[0]), int(box[1])])
        crop_img, _ = Alignment_1(crop_img, landmark)

        crop_img = np.array(letterbox_image(np.uint8(crop_img), (self.facenet_input_shape[1], self.facenet_input_shape[0]))) / 255
        crop_img = np.expand_dims(crop_img.transpose(2, 0, 1), 0)

        with torch.no_grad():
            crop_tensor = torch.from_numpy(crop_img).type(torch.FloatTensor)
            if self.cuda:
                crop_tensor = crop_tensor.cuda()
            return self.facenet(crop_tensor)[0].cpu().numpy()

    def _find_largest_face(self, boxes_conf_landms):
        """Return the detection box with the largest area."""
        return max(boxes_conf_landms, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

    def _identify_face(self, face_encoding):
        """Match a face encoding against stored encodings; return name or 'Unknown'."""
        matches, face_distances = compare_faces(self.known_face_encodings, face_encoding, tolerance=self.facenet_threhold)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            return self.known_face_names[best_match_index]
        return "Unknown"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_face_dataset(self, image_paths, names):
        face_encodings = []
        for index, path in enumerate(tqdm(image_paths)):
            image     = np.array(Image.open(path), np.float32)
            old_image = image.copy()

            im_height, im_width, scale, scale_for_landmarks = self._compute_scales(image)
            image_tensor, anchors = self._prepare_detection_input(image, im_height, im_width)
            boxes_conf_landms = self._run_detector(image_tensor, anchors, im_height, im_width, scale, scale_for_landmarks)

            if len(boxes_conf_landms) == 0:
                print(names[index], "：未检测到人脸")
                continue

            best_box     = self._find_largest_face(boxes_conf_landms)
            face_encoding = self._encode_face_crop(best_box, old_image)
            face_encodings.append(face_encoding)

        np.save("model_data/{backbone}_face_encoding.npy".format(backbone=self.facenet_backbone), face_encodings)
        np.save("model_data/{backbone}_names.npy".format(backbone=self.facenet_backbone), names)

    def encode_face_image(self, name, image, backbone) -> dict:
        result = {"status": 2, "face_encoding_base64": None, "message": "Success"}

        try:
            image     = base64_to_numpy_image(image)
            old_image = image.copy()

            im_height, im_width, scale, scale_for_landmarks = self._compute_scales(np.array(image, np.float32))
            image_tensor, anchors = self._prepare_detection_input(image, im_height, im_width)
            boxes_conf_landms = self._run_detector(image_tensor, anchors, im_height, im_width, scale, scale_for_landmarks)

            if len(boxes_conf_landms) == 0:
                result["status"]  = 0
                result["message"] = "No face detected"
                return result

            best_box      = self._find_largest_face(boxes_conf_landms)
            face_encoding = self._encode_face_crop(best_box, old_image)

            self.redis_storage.store_face_data([name], [face_encoding], backbone)

            face_encoding_base64 = base64.b64encode(face_encoding.tobytes()).decode('utf-8')
            result["face_encoding_base64"] = face_encoding_base64
            result["encoding_shape"]       = face_encoding.shape

        except Exception as e:
            print("Something error ", e)
            result["status"]  = -1
            result["message"] = f"Error: {str(e)}"

        return result

    def delete_face_data(self, user_id: str, algorithm: str = None) -> bool:
        """Delete face data from Redis by user_id. Returns True on success."""
        try:
            backbone  = algorithm if algorithm else "mobilenet"
            redis_key = f"{backbone}_face_data"

            if not self.redis_storage.redis_client.hexists(redis_key, user_id):
                print(f"No face data found for user_id: {user_id} with algorithm: {backbone}")
                return False

            deleted_count = self.redis_storage.redis_client.hdel(redis_key, user_id)
            print(f"Deleted face data for user_id: {user_id}, algorithm: {backbone}")
            return deleted_count > 0

        except Exception as e:
            print(f"Error deleting face data for {user_id}: {e}")
            return False

    def search_face(self, image):
        image     = base64_to_numpy_image(image)
        old_image = image.copy()
        image     = np.array(image, np.float32)

        im_height, im_width, scale, scale_for_landmarks = self._compute_scales(image)
        image_tensor, anchors = self._prepare_detection_input(image, im_height, im_width)
        boxes_conf_landms = self._run_detector(image_tensor, anchors, im_height, im_width, scale, scale_for_landmarks)

        if len(boxes_conf_landms) == 0:
            return old_image, {}

        face_encodings = [self._encode_face_crop(box, old_image) for box in boxes_conf_landms]
        face_names     = [self._identify_face(enc) for enc in face_encodings]

        raw_data = {}
        for i, b in enumerate(boxes_conf_landms):
            text = "{:.4f}".format(b[4])
            b    = list(map(int, b))

            cv2.rectangle(old_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cv2.putText(old_image, text, (b[0], b[1] + 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            cv2.circle(old_image, (b[5],  b[6]),  1, (0, 0, 255),   4)
            cv2.circle(old_image, (b[7],  b[8]),  1, (0, 255, 255), 4)
            cv2.circle(old_image, (b[9],  b[10]), 1, (255, 0, 255), 4)
            cv2.circle(old_image, (b[11], b[12]), 1, (0, 255, 0),   4)
            cv2.circle(old_image, (b[13], b[14]), 1, (255, 0, 0),   4)

            name = face_names[i]
            raw_data[name] = text
            old_image = cv2ImgAddText(old_image, name, b[0] + 5, b[3] - 25)

        return old_image, raw_data

    def get_FPS(self, image, test_interval):
        old_image = image.copy()
        image     = np.array(image, np.float32)

        im_height, im_width, scale, scale_for_landmarks = self._compute_scales(image)
        image_tensor, anchors = self._prepare_detection_input(image, im_height, im_width)

        # Warmup run so GPU kernels are compiled before timing starts
        boxes_conf_landms = self._run_detector(image_tensor, anchors, im_height, im_width, scale, scale_for_landmarks)
        if len(boxes_conf_landms) > 0:
            face_encodings = [self._encode_face_crop(box, old_image) for box in boxes_conf_landms]
            [self._identify_face(enc) for enc in face_encodings]

        t1 = time.time()
        for _ in range(test_interval):
            boxes_conf_landms = self._run_detector(image_tensor, anchors, im_height, im_width, scale, scale_for_landmarks)
            if len(boxes_conf_landms) > 0:
                face_encodings = [self._encode_face_crop(box, old_image) for box in boxes_conf_landms]
                [self._identify_face(enc) for enc in face_encodings]
        t2 = time.time()
        return (t2 - t1) / test_interval
