import os
import sys

prj_path = os.path.join(os.path.dirname(__file__), '../')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import argparse
import torch
from lib.utils.misc import NestedTensor
from thop import profile
from thop.utils import clever_format
import time
import importlib


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='t2track', choices=['t2track'],
                        help='training script name')
    parser.add_argument('--config', type=str, default='t2track_12', help='yaml configure file name')
    parser.add_argument('--device', type=str, default='cpu', help='setup runing device type')

    args = parser.parse_args()

    return args


def evaluate_track(model, template, search):
    '''Speed Test'''
    macs1, params1 = profile(model, inputs=(template, search),
                             custom_ops=None, verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)

    T_w = 200
    T_t = 500
    print("testing speed ...")
    torch.cuda.synchronize()
    with torch.no_grad():
        # overall
        for i in range(T_w):
            tem_feats = model(template_list=template, search_list=search)
            # tem_feats = model.forward_encoder(template[-1])
            # search_feats = model.forward_encoder(search[-1])
            # feats = model.forward_neck(tem_feats,search_feats)
            # _ = model.forward_decoder(feats)
            # _ = model.forward_test(template, search)
            template_list = None,
            search_list = None,
            template_anno_list = None,
            text_src = None,
            task_index = None,
            feature = None,
            mode = "encoder"

        start = time.time()
        # tem_feats = model.forward_encoder(template[-1])
        for i in range(T_t):
            tem_feats = model(template_list=template, search_list=search)
            # search_feats = model.forward_encoder(search[-1])
            # feats = model.forward_neck(tem_feats,search_feats)
            # _ = model.forward_decoder(feats)
            # _ = model.forward_test(template, search)
        torch.cuda.synchronize()
        end = time.time()
        avg_lat = (end - start) / T_t
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))
        print("FPS is %.2f fps" % (1. / avg_lat))
        # for i in range(T_w):
        #     _ = model(template, search)
        # start = time.time()
        # for i in range(T_t):
        #     _ = model(template, search)
        # end = time.time()
        # avg_lat = (end - start) / T_t
        # print("The average backbone latency is %.2f ms" % (avg_lat * 1000))



def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz)
    att_mask = torch.rand(bs, sz, sz) > 0.5
    return NestedTensor(img_patch, att_mask)


if __name__ == "__main__":

    args = parse_args()
    devices = ['gpu', 'cpu']
    # Compute the Flops and Params of our STARK-S model
    for device in devices:
        for i in range(10):
            print(f'device type: {device}, model: {args.config}')
            if device =='gpu':
                torch.cuda.set_device('cuda:0')
                device = 'cuda:0'

            '''update cfg'''
            yaml_fname = 'experiments/%s/%s.yaml' % (args.script, args.config)
            config_module = importlib.import_module('lib.config.%s.config' % args.script)
            cfg = config_module.cfg
            config_module.update_config_from_file(os.path.join(os.path.abspath(prj_path), yaml_fname))
            '''set some values'''
            bs = 1
            z_sz = cfg.TEST.TEMPLATE_SIZE
            x_sz = cfg.TEST.SEARCH_SIZE

            if args.script == "uavtrack":
                model_module = importlib.import_module('lib.models.uavtrack')
                model_constructor = model_module.build_uavtrack
                model = model_constructor(cfg)
                model.eval()
                # get the template and search
                template = torch.randn(bs, 3, z_sz, z_sz).to(device)
                search = torch.randn(bs, 3, x_sz, x_sz).to(device)
                # transfer to device
                model = model.to(device)
                template = [template]
                search = [search]
                evaluate_track(model, template, search)
            elif args.script == "t2track":
                model_module = importlib.import_module('lib.models.t2track')
                model_constructor = model_module.build_t2track
                model = model_constructor(cfg)
                model.eval()
                # get the template and search
                template = torch.randn(bs, 3, z_sz, z_sz).to(device)
                search = torch.randn(bs, 3, x_sz, x_sz).to(device)
                model.is_memory = True
                for idx in range(cfg.DATA.SEARCH.HISTORY_LEN):
                    model.memory_search.append(torch.randn([1, 49, 256]).to(device))
                # transfer to device
                model = model.to(device)
                template = [template]
                search = [search]
                evaluate_track(model, template, search)
            else:
                raise NotImplementedError
