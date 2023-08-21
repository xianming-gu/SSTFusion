import argparse

parser = argparse.ArgumentParser(description='MyOption')
# Train args
parser.add_argument('--DEVICE', type=str, default='cuda:1')
parser.add_argument('--epoch', type=int, default=50)  # 1000 800 600
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--patch_size', type=int, default=128)
parser.add_argument('--seed', type=int, default=3407)

parser.add_argument('--temp_dir', type=str, default='./temp/train_ps=256')  # CT PET SPECT

# Train & Test args
parser.add_argument('--dir_train', type=str, default='./HMIFDatasets/SPECT-MRI/train/')  # CT PET SPECT
parser.add_argument('--dir_test', type=str, default='./HMIFDatasets/SPECT-MRI/test/')  # CT PET SPECT

parser.add_argument('--img_type1', type=str, default='SPECT/')  # CT PET SPECT
parser.add_argument('--img_type2', type=str, default='MRI/')

parser.add_argument('--model_save_path', type=str, default='./modelsave/train_ps=256')  # CT PET SPECT
parser.add_argument('--model_save_name', type=str, default='MyModel.pth')

# Test args
parser.add_argument('--img_save_dir', type=str, default='result/train_ps=256')  # CT PET SPECT

args = parser.parse_args()

# ######################## 检查 ！！！ 损失函数（权重、type）  共享参数  模型架构  ！！！
