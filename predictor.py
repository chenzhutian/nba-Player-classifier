import os
import os.path as osp
import cv2
import torch

from yolox.exp import get_exp
from yolox.data.data_augment import preproc
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.tracking_utils.timer import Timer

from transformers import ViTForImageClassification, TrainingArguments, Trainer, ViTFeatureExtractor
from transformers.models.vit.feature_extraction_vit import ViTFeatureExtractor 
# Player classification
feature_extractor: ViTFeatureExtractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
classifier = ViTForImageClassification.from_pretrained('/datadrive/player-classifier/vit-base-beans-demo-v5/')
classifier.cuda()

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info


def get_predictor():
    exp = get_exp(osp.join('../ByteTrack', 'exps/example/mot/yolox_x_mix_det.py'), None)
    device = torch.device("cuda")

    model = exp.get_model().to(device)
    model.eval()

    ckpt_file = osp.join('../ByteTrack', 'pretrained/bytetrack_x_mot17.pth.tar')
    ckpt = torch.load(ckpt_file, map_location="cpu")

    # load the model state dict
    model.load_state_dict(ckpt["model"])
    model = fuse_model(model)
    model = model.half()  # to FP16
    trt_file = None
    decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, device, True)
    return predictor

def is_player(img: cv2.Mat) -> bool:
    img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    features = feature_extractor([img_converted], return_tensors="pt")
    predictions = classifier(**{ k:v.cuda() for k,v in features.items() })
    return bool(predictions.logits.cpu().argmax() == 1)
    
    