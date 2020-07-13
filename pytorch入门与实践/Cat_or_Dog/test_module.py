from models import AlexNet

import models
model = models.AlexNet()

import models
model = getattr(models, 'AlexNet')()