from nn.mobilenetv3 import mobilenetv3_large,mobilenetv3_large_full,mobilenetv3_small
import torch
from nn.models import DarknetWithShh
from hyp import hyp

def convert_onnx():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = 'weights/mbv3_large_75_light_final.pt' #这是我们要转换的模型
    backone = mobilenetv3_large(width_mult=0.75)#mobilenetv3_small()  mobilenetv3_small(width_mult=0.75)  mobilenetv3_large(width_mult=0.75)
    model = DarknetWithShh(backone, hyp,light_head=True).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device)['model'])

    model.to(device)
    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32).to(device)#输入大小   #data type nchw
    onnx_path = 'weights/mbv3_large_75_light_final.onnx'
    torch.onnx.export(model, dummy_input, onnx_path, input_names=['input'], output_names=['output'],opset_version=11)
    print('convert retinaface to onnx finish!!!')

if __name__ == "__main__" :
    convert_onnx()