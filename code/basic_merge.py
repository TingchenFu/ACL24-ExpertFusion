from transformers.trainer import get_last_checkpoint
import sys
from safetensors import safe_open
import argparse
import torch
import os
from merge_utils import basic_merging
RUN_DIR=os.getcwd()
to_merge_name = "Llama-2-7b-hf_lima_cluster{}_fromrandom_cluster4_lora_fp16_bs16lr3e-4epoch3decay0.0cosine"
merged_name =   "Llama-2-7b-hf_lima_mergedavg_fromrandom_cluster4_lora_fp16_bs16lr3e-4epoch3decay0.0cosine"


## lora niidclassification
if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_paths', type=str, default='', help='the path of the checkpoints to be merged, shoule be seperated by commas')
    parser.add_argument('--mix_ratio', type=str, default='', help='the ratio of the checkpoints to be merged, shoule be seperated by commas')
    parser.add_argument('--output_path',type=str,default='',help='the path of the merged checkpoint')
    args = parser.parse_args()
    state_dicts=[]
    assert args.ckpt_paths and args.mix_ratio and len(args.ckpt_paths.split(',')) == len(args.mix_ratio.split(','))
    args.ckpt_paths = args.ckpt_paths.split(',')
    args.mix_ratio = [float(x) for x in args.mix_ratio.split(',')]
    args.mix_ratio = [x/sum(args.mix_ratio) for x in args.mix_ratio]
 
    for ckpt_path in args.ckpt_paths:
        try:
            state_dicts.append(torch.load(ckpt_path))
        except:
            state_dict = {}
            ckpt_path = ckpt_path.replace('.bin','.safetensors')
            with safe_open(ckpt_path,framework="pt", device='cpu') as f:
                for k in f.keys():
                    state_dict[k]= f.get_tensor(k)
            state_dicts.append(state_dict)
        finally:
            print('load {} successfully'.format(ckpt_path))


    merged_state_dict = basic_merging(agg_mode='weighted',ratio=args.mix_ratio,finetune_statedicts=state_dicts)
    order1 = 'rm -rf ' + args.output_path
    order2 = 'cp -r ' + os.path.join(args.ckpt_paths[0], 'adapter_config.json') + ' ' + merged_name

    os.system(order1)
    os.makedirs(os.path.join(RUN_DIR, 'dump',current_merged_name))
    os.system(order2)
    sys.stdout.flush()
    save_path = os.path.join(args.output_path, 'adapter_model.bin')
    torch.save(merged_state_dict,save_path)

