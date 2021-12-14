##
# Import libraries

from effdet.config.model_config import efficientdet_model_param_dict
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
from effdet.efficientdet import HeadNet
from effdet.config.model_config import efficientdet_model_param_dict

from pytorch_lightning import LightningModule
from pytorch_lightning.core.decorators import auto_move_data


##
# Effdet model class (pytorch-lightning)

# class EfficientDetModel(LightningModule):
#     def __init__(
#             self,
#             num_classes=1,
#             img_size=512,
#             prediction_confidence_threshold=0.2,
#             learning_rate=0.0002,
#             wbf_iou_threshold=0.44,
#             model_architecture='tf_efficientnetv2_l',
#     ):
#         super().__init__()
#         self.img_size = img_size
#         self.model = create_model(
#             num_classes, img_size, architecture=model_architecture
#         )
#         self.prediction_confidence_threshold = prediction_confidence_threshold
#         self.lr = learning_rate
#         self.wbf_iou_threshold = wbf_iou_threshold
#
#     def forward(self, images, targets):
#         return self.model(images, targets)
#


##
def create_model(num_classes=1, image_size=512, architecture="tf_efficientnetv2_l"):
    efficientdet_model_param_dict['tf_efficientnetv2_l'] = dict(
        name='tf_efficientnetv2_l',
        backbone_name='tf_efficientnetv2_l',
        backbone_args=dict(drop_path_rate=0.2),
        num_classes=num_classes,
        url='', )

    config = get_efficientdet_config(architecture)
    config.update({'num_classes': num_classes})
    config.update({'image_size': (image_size, image_size)})

    # print(config)

    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
    )
    return DetBenchTrain(net, config)