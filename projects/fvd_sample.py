import torch
import json
from .common_metrics_on_video_quality.calculate_fvd import calculate_fvd

NUMBER_OF_VIDEOS = 8
VIDEO_LENGTH = 12
CHANNEL = 3
SIZE = 64
videos1 = torch.randn(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
videos2 = torch.randn(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

result = {}
result['fvd'] = calculate_fvd(videos1, videos2, device, method='styleganv')
print(json.dumps(result, indent=4))