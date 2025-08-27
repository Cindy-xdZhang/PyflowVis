
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from scripts.dataGen import generate_triangle_points,generate_quad_points,generate_circle_points,draw_registration_result
from models import Point_NN
import numpy as np
import torch
from tqdm import tqdm
import argparse
from DeepUtils.utils import EasyConfig
from FLowUtils.VectorField2d import *
from FLowUtils.netCDFLoader import *

from sklearn.cluster import KMeans

from libs.flows import * #compute_streamline_euler,visualize_streamline,visualize_multiple_streamline,compute_multiple_streamlines,visualize_streamlines_labels
from libs.parallel_flows import *

# from libs.omegaconf_utils import *

from libs.curvature import *

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import logging
logger=logging.getLogger()

def get_arguments():
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='mn40')
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args= parser.parse_args()
    return args
def get_logger(config):
    # output_exp = config.output.exp
    # time_str = time.strftime("%Y_%m_%d_%H_%M_%S")
    # logdir = os.path.join('output',f'{config.dataset.name}',
    #         output_exp,
    #         time_str,
    #     )
    # config.logdir=logdir
    # os.makedirs(logdir, exist_ok=True)
    # logger.add(os.path.join(logdir, 'log.txt'), level="INFO")
    return logger


def check_group_have_same_lengths(valid_steps: torch.Tensor, nerbors: int = 5):
    """
    valid_steps: [M]，M = N*nerbors，按 (seed0的5条, seed1的5条, ...) 排序
    返回:
      same:        [N] whether we have same length
      bad_groups:  [B] inequal length group index
      g_min, g_max:[N] min steps and max steps in each group
    
    DL: be careful that we only filter whether the grup have the same length, we need to check again whether we have correct length after sampling
    """
    assert valid_steps.numel() % nerbors == 0, "lines count must be multiple of nerbors"
    g = valid_steps.view(-1, nerbors)            # [N, 5]
    same = (g == g[:, [0]]).all(dim=1)           # 行内是否全等
    bad_groups = (~same).nonzero(as_tuple=False).squeeze(1)  # [B]
    g_min = g.min(dim=1).values
    g_max = g.max(dim=1).values
    return same, bad_groups, g_min, g_max

def select_good_groups(valid_steps: torch.Tensor, nerbors: int, K: int, strict: bool = True):
    """
    选出满足条件的组：
      - same == True（组内5条长度相等）
      - g_min >  K  (strict=True)  或  g_min >= K (strict=False)

    返回：
      keep_groups: [N]  组级布尔掩码
      keep_lines:  [M]  逐线布尔掩码（按组重复 nerbors 次）
      good_idx:    [G]  满足条件的组索引
    """
    same, _, g_min, _ = check_group_have_same_lengths(valid_steps, nerbors)
    cond_len = (g_min > K) if strict else (g_min >= K)
    # keep_groups = same & cond_len                     # [N]
    keep_groups = cond_len                     # [N]
    good_idx = torch.nonzero(keep_groups, as_tuple=False).squeeze(1)  # [G]
    keep_lines = keep_groups.repeat_interleave(nerbors)               # [M]
    return keep_groups, keep_lines, good_idx

class NetCDFLoaderOBJ():
    #beads2d
    #doublegyre2d
    def __init__(self,path=rf"D:\Projects\FlowVis\PyflowVis\data\doublegyre2d.nc", timestep_begin=-1,timestep_end=-1):
        self.path=path
        self.timestep_begin=timestep_begin
        self.timestep_end=timestep_end
    def loadCDF(self):
        logger.info(f"start loading")
        resolved_path=self.path
        time_step_begin=self.timestep_begin
        time_step_end=self.timestep_end
        vectorfield=NetCDFLoader.load_vector_field2d(resolved_path,time_step_begin,time_step_end)
        
        logger.info(f"scene.getObject successfully.")
        return vectorfield


    



if __name__ == "__main__":
    
    
    
    print('==> Loading args..')
    args = get_arguments()
    logger.info(args)
    config = EasyConfig()
    config.load(args.config)
    get_logger(config)
    logger.info(config)
    
    # get the data path
    datapath=f"{config.dataset.dat_dir}\\{config.dataset.name}.{config.dataset.extension}"
    
    # load the data
    netCDF=NetCDFLoaderOBJ(path=datapath)
    vectorfield=netCDF.loadCDF()
    vectorfield.showInfo()
    config.pcds.time_slice=int(vectorfield.field.shape[0]*config.pcds.t_start)
    
    
    #np_points_ori,np_points_tsfm,#points_tsfm:torch.Size([55, 1005, 3])
    points_ori,valid_steps=run_pathlines_batched(config.pcds,vectorfield,line_method="Euler")
    points_ori_rk4,valid_steps_rk4=run_pathlines_batched(config.pcds,vectorfield,line_method="RK4")
    
    keep_groups, keep_lines, good_idx = select_good_groups(
        valid_steps, nerbors=config.pcds.sample_nerbors, K=config.pcds.line_sample, strict=True
    )
    # filter the lines：
    points_ori      = points_ori[keep_lines]
    valid_steps = valid_steps[keep_lines]
    
    
    keep_groups, keep_lines, good_idx = select_good_groups(
        valid_steps_rk4, nerbors=config.pcds.sample_nerbors, K=config.pcds.line_sample, strict=True
    )
    points_ori_rk4      = points_ori_rk4[keep_lines]
    valid_steps_rk4 = valid_steps_rk4[keep_lines]
    
    print('==> Preparing model..')
    args=config.pnn# reset the args from the configs
    args.points=points_ori.shape[1]
    point_nn = Point_NN(input_points=args.points, num_stages=args.stages,
                        embed_dim=args.dim, k_neighbors=args.k,
                        alpha=args.alpha, beta=args.beta).cuda()
    point_nn.eval()
    
    with torch.no_grad():
        vis_max_steps=0
        # for some dataset, we need to sample the lines
        if config.pcds.line_sample>1:
            
            points_ori=resample_to_fixed_count(points=points_ori,
                                     valid_steps=valid_steps,
                                     K=config.pcds.line_sample)
            
            points_ori_rk4=resample_to_fixed_count(points=points_ori_rk4,
                                     valid_steps=valid_steps_rk4,
                                     K=config.pcds.line_sample)
            # points_ori=sampleLines(line_sample=config.pcds.line_sample,sample_nerbors=config.pcds.sample_nerbors,points=points_ori)
            # points_ori_rk4=sampleLines(line_sample=config.pcds.line_sample,sample_nerbors=config.pcds.sample_nerbors,points=points_ori_rk4)
        
        points_ori=points_ori.reshape([points_ori.shape[0]//config.pcds.sample_nerbors,-1,points_ori.shape[-1]])
        points_ori_rk4=points_ori_rk4.reshape([points_ori_rk4.shape[0]//config.pcds.sample_nerbors,-1,points_ori_rk4.shape[-1]])
        
        logger.info(f"points_ori:{points_ori.shape}")
        vis_max_steps=points_ori.shape[1]//config.pcds.sample_nerbors-1
        logger.info(f"vis_max_steps:{vis_max_steps}")
        
        
        
        # Vis the comparsion between RK4 and Euler
        labels_test=np.ones(points_ori.shape[0])
        # multi_lines_vis_fast(vectorfield,[points_ori.detach().cpu(),
        #                                   points_ori_rk4.detach().cpu(),
        #                              ], \
                                        
        #                 [labels_test,labels_test],time_index=config.pcds.time_slice,max_step=vis_max_steps,\
        #                 title=["Original pds Euler",
        #                        "Original pds RK4"
        #                        ]
        #                 ,layout=[2,1])
        
        
        """
        After that, we only keep pathlines from rk4
        we pay attention to 
            points_ori_rk4 : original lines
            points_tsfm_loc_rk4 : local lines based on the first points in the center lines
            points_tsfm_nor_rk4 : normalized lines based on the first points in the center lines
            points_tsfm_t_loc_rk4 : local lines based on each point in the center lines
            points_tsfm_t_nor_rk4 : normalized lines based on each point in the center lines
        
        We seperate the visualization into three parts:
        1. Vis the above five lines 
        2. Vis the clustering using features
        3. Vis the clustering wo features
        
        """
        points_tsfm_loc_rk4=None
        points_tsfm_t_loc_rk4=None
        
        points_tsfm_nor_rk4=None
        points_tsfm_t_nor_rk4=None
        centerLines_rk4=None
        
        curvature=None
        if config.pcds.localized:
            
            points_tsfm_loc_rk4=LocLines(sample_nerbors=config.pcds.sample_nerbors,points=points_ori_rk4)
            points_tsfm_t_loc_rk4,centerLines_rk4=LocLines4Time(sample_nerbors=config.pcds.sample_nerbors,points=points_ori_rk4)
            
            # ========== 示例用法 ==========
            pathline = centerLines_rk4
            pathline= pathline
            curvature = compute_curvature_batch(pathline)
            # visualize_curvature(pathline.detach().cpu().numpy(), curvature.detach().cpu().numpy())
            
            
        if config.pcds.normalized:
            if config.pcds.localized:
                points_tsfm_nor_rk4=normalizeLines(sample_nerbors=config.pcds.sample_nerbors,points=points_tsfm_loc_rk4,vectorfield=vectorfield)
                points_tsfm_t_nor_rk4=normalizeLines(sample_nerbors=config.pcds.sample_nerbors-1,points=points_tsfm_t_loc_rk4,vectorfield=vectorfield)
            # else:
            #     points_tsfm_nor=normalizeLines(sample_nerbors=config.pcds.sample_nerbors,points=points_ori,vectorfield=vectorfield)
        
        logger.info(f"curvature:{curvature.shape}")
        
        logger.info(f"points_ori_rk4:{points_ori_rk4.shape}")
        
        logger.info(f"points_tsfm_loc_rk4:{points_tsfm_loc_rk4.shape}")
        logger.info(f"points_tsfm_t_loc_rk4:{points_tsfm_t_loc_rk4.shape}")
        
        logger.info(f"points_tsfm_nor_rk4:{points_tsfm_nor_rk4.shape}")
        logger.info(f"points_tsfm_t_nor_rk4:{points_tsfm_t_nor_rk4.shape}")
        
        
        
        #abs coords
        # === 主流程 ===
        if points_ori_rk4.shape[0] > 50:
            point_features_ori_rk4 = compute_features_in_chunks(points_ori_rk4, point_nn, chunk_size=50)
        else:
            point_features_ori_rk4= point_nn(points_ori_rk4.permute(0, 2, 1))
        print(f"point_features_ori_rk4: {point_features_ori_rk4.shape}")  # e.g. [B, C]

        if points_tsfm_loc_rk4.shape[0] > 50:
            point_features_loc_rk4 = compute_features_in_chunks(points_tsfm_loc_rk4, point_nn, chunk_size=50)
        else:
            point_features_loc_rk4 = point_nn(points_tsfm_loc_rk4.permute(0, 2, 1))
        print(f"point_features_loc_rk4: {point_features_loc_rk4.shape}")

        if points_tsfm_nor_rk4.shape[0] > 50:
            point_features_nor_rk4 = compute_features_in_chunks(points_tsfm_nor_rk4, point_nn, chunk_size=50)
        else:
            point_features_nor_rk4 = point_nn(points_tsfm_nor_rk4.permute(0, 2, 1))
        print(f"point_features_nor_rk4: {point_features_nor_rk4.shape}")
        
        
        if points_tsfm_t_loc_rk4.shape[0] > 50:
            point_features_t_loc_rk4 = compute_features_in_chunks(points_tsfm_t_loc_rk4, point_nn, chunk_size=50)
        else:
            point_features_t_loc_rk4 = point_nn(points_tsfm_t_loc_rk4.permute(0, 2, 1))
        print(f"point_features_t_loc_rk4: {point_features_t_loc_rk4.shape}")
        
        if points_tsfm_t_nor_rk4.shape[0] > 50:
            point_features_t_nor_rk4 = compute_features_in_chunks(points_tsfm_t_nor_rk4, point_nn, chunk_size=50)
        else:
            point_features_t_nor_rk4 = point_nn(points_tsfm_t_nor_rk4.permute(0, 2, 1))
        print(f"point_features_t_nor_rk4: {point_features_t_nor_rk4.shape}")
        
        
        
        
        #== Visualize the pathlines
        multi_lines_vis_fast(vectorfield,[points_ori_rk4.detach().cpu(),
                                          points_tsfm_loc_rk4.detach().cpu(),
                                          points_tsfm_t_loc_rk4.detach().cpu(),
                                     points_tsfm_nor_rk4.detach().cpu(),
                                     points_tsfm_t_nor_rk4.detach().cpu(),
                                     ], \
                                        
                        [labels_test,labels_test,labels_test,labels_test,labels_test],time_index=config.pcds.time_slice,max_step=vis_max_steps,\
                        title=["Original pds RK4",
                               "Local pds wo time(First Point)",
                               "Local pds w time(Center Line)",
                               "normalized pds wo time(First Point)",
                               "Normalized pds w time(Center Line)",
                               ]
                        ,layout=[5,1])
        
        

        
        n_clusters=config.cluster.classes
        
        # 1. convert to  NumPy（scikit-learn needs）
        point_features_ori_rk4 = point_features_ori_rk4.cpu().detach().numpy()
        
        point_features_loc_rk4 = point_features_loc_rk4.cpu().detach().numpy()
        point_features_nor_rk4 = point_features_nor_rk4.cpu().detach().numpy()
        
        point_features_t_loc_rk4 = point_features_t_loc_rk4.cpu().detach().numpy()
        point_features_t_nor_rk4 = point_features_t_nor_rk4.cpu().detach().numpy()
        
        labels_ori_rk4=None
        labels_loc_rk4=None
        labels_nor_rk4=None
        labels_t_loc_rk4=None
        labels_t_nor_rk4=None
        if config.cluster.alg=="kmeans":
            kmeans = KMeans(n_clusters=n_clusters, n_init=50,random_state=0,algorithm="elkan")
            
            kmeans.fit(point_features_ori_rk4)
            # 3. get labels
            labels_ori_rk4 = kmeans.labels_
            
            kmeans.fit(point_features_loc_rk4)
            # 3. get labels
            labels_loc_rk4 = kmeans.labels_
            
            kmeans.fit(point_features_nor_rk4)
            # 3. get labels
            labels_nor_rk4 = kmeans.labels_
            
            
            kmeans.fit(point_features_t_nor_rk4)
            # 3. get labels
            labels_t_nor_rk4 = kmeans.labels_
            
            kmeans.fit(point_features_t_loc_rk4)
            # 3. get labels
            labels_t_loc_rk4 = kmeans.labels_
            
        logger.error(f" We are showing the clustering with latent features")
        
        logger.info(f"config.pcds.time_slice:{config.pcds.time_slice}")
        # multi_lines_vis_fast(vectorfield,[points_ori_rk4.detach().cpu(),
        #                                   points_ori_rk4.detach().cpu(),
        #                              points_ori_rk4.detach().cpu(),
        #                              points_ori_rk4.detach().cpu(),
        #                              points_ori_rk4.detach().cpu(),
        #                              ], \
                                        
        #                 [labels_ori_rk4,labels_loc_rk4,labels_t_loc_rk4,labels_nor_rk4,labels_t_nor_rk4],time_index=config.pcds.time_slice,max_step=vis_max_steps,\
        #                 title=["Original pds features RK4",
        #                        "Local pds features RK4 wo time(First Point)",
        #                        "Local pds features RK4 w time(Center Line)",
        #                        "Normalized pds features wo time(First Point)",
        #                        "Normalized pds features w time(Center Line)"
        #                        ]
        #                 ,layout=[5,1])
        
        multi_points_vis_fast(vectorfield,[points_ori_rk4.detach().cpu(),
                                          points_ori_rk4.detach().cpu(),
                                     points_ori_rk4.detach().cpu(),
                                     points_ori_rk4.detach().cpu(),
                                     points_ori_rk4.detach().cpu(),
                                     ], \
                                        
                        [labels_ori_rk4,labels_loc_rk4,labels_t_loc_rk4,labels_nor_rk4,labels_t_nor_rk4],time_index=config.pcds.time_slice,max_step=vis_max_steps,\
                        title=["Original pds features RK4",
                               "Local pds latent features RK4 wo time(First Point)",
                               "Local pds latent features RK4 w time(Center Line)",
                               "Normalized pds latent features wo time(First Point)",
                               "Normalized pds latent features w time(Center Line)"
                               ]
                        ,layout=[5,1])
         
        # we try no features extraction
        syn_points_no_features_np_ori=points_ori_rk4.reshape(points_ori_rk4.shape[0],-1).detach().cpu().numpy()
        syn_points_no_features_np_loc=points_tsfm_loc_rk4.reshape(points_tsfm_loc_rk4.shape[0],-1).detach().cpu().numpy()
        syn_points_no_features_np_nor=points_tsfm_nor_rk4.reshape(points_tsfm_nor_rk4.shape[0],-1).detach().cpu().numpy()
        syn_points_no_features_np_t_loc=points_tsfm_t_loc_rk4.reshape(points_tsfm_t_loc_rk4.shape[0],-1).detach().cpu().numpy()
        syn_points_no_features_np_t_nor=points_tsfm_t_nor_rk4.reshape(points_tsfm_t_nor_rk4.shape[0],-1).detach().cpu().numpy()
        # print(f"syn_points_no_features:{syn_points_no_features_np.shape}")
        labels_ori=None
        labels_loc=None
        labels_nor=None
        labels_t_loc=None
        labels_t_nor=None
        
        if config.cluster.alg=="kmeans":
            kmeans = KMeans(n_clusters=n_clusters, init="k-means++",random_state=0)
            kmeans.fit(syn_points_no_features_np_ori)
            # 3. get labels
            labels_ori = kmeans.labels_
            
            
            kmeans.fit(syn_points_no_features_np_loc)
            # 3. get labels
            labels_loc = kmeans.labels_
            
            
            kmeans.fit(syn_points_no_features_np_nor)
            # 3. get labels
            labels_nor = kmeans.labels_
            # print(f"labels_nor:{labels_nor}")
            
            kmeans.fit(syn_points_no_features_np_t_nor)
            # 3. get labels
            labels_t_nor = kmeans.labels_
            
            kmeans.fit(syn_points_no_features_np_t_loc)
            # 3. get labels
            labels_t_loc = kmeans.labels_

        logger.error(f" We are showing the clustering without latent features")
        # multi_lines_vis_fast(vectorfield,[points_ori_rk4.detach().cpu(),
        #                              points_ori_rk4.detach().cpu(),
        #                              points_ori_rk4.detach().cpu(),
        #                              points_ori_rk4.detach().cpu(),
        #                              points_ori_rk4.detach().cpu(),
        #                              ], \
        #                 [labels_ori,labels_loc,labels_t_loc,labels_nor,labels_t_nor],time_index=config.pcds.time_slice,max_step=vis_max_steps,\
        #                 title=["Original pds RK4",
        #                        "Local pds RK4 wo time(First Point)",
        #                        "Local pds RK4 w time(Center Line)",
        #                        "Normalized pds  wo time(First Point)",
        #                        "Normalized pds  w time(Center Line)"
        #                        ]
        #                 ,layout=[5,1])
        
        multi_points_vis_fast(vectorfield,[points_ori_rk4.detach().cpu(),
                                     points_ori_rk4.detach().cpu(),
                                     points_ori_rk4.detach().cpu(),
                                     points_ori_rk4.detach().cpu(),
                                     points_ori_rk4.detach().cpu(),
                                     ], \
                        [labels_ori,labels_loc,labels_t_loc,labels_nor,labels_t_nor],time_index=config.pcds.time_slice,max_step=vis_max_steps,\
                        title=["Original pds RK4",
                               "Local pds RK4 wo time(First Point)",
                               "Local pds RK4 w time(Center Line)",
                               "Normalized pds  wo time(First Point)",
                               "Normalized pds  w time(Center Line)"
                               ]
                        ,layout=[5,1])
        
        
        #---- Curvature Clustering  ----#
        # labels_curvature=None
        # if config.cluster.alg=="kmeans":
        #     kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        #     kmeans.fit(curvature.detach().cpu().numpy())
        #     # 3. get labels
        #     labels_curvature = kmeans.labels_
        
        # # multi_lines_vis_fast(vectorfield,[points_ori_rk4.detach().cpu(),
        # #                              points_ori_rk4.detach().cpu(),
        # #                              ], \
        # #                 [labels_t_nor,labels_curvature],time_index=config.pcds.time_slice,max_step=vis_max_steps,\
        # #                 title=[
        # #                         "Normalized pds  w time(Center Line)",
        # #                         "labels_curvature features clustering",
        # #                        ]
        # #                 ,layout=[2,1])
        # multi_points_vis_fast(vectorfield,[points_ori_rk4.detach().cpu(),
        #                              points_ori_rk4.detach().cpu(),
        #                              ], \
        #                 [labels_t_nor,labels_curvature],time_index=config.pcds.time_slice,max_step=vis_max_steps,\
        #                 title=[
        #                         "Normalized pds  w time(Center Line)",
        #                         "labels_curvature features clustering",
        #                        ]
        #                 ,layout=[2,1])
        
        # plt.pause(0.001)        # 让UI有时间渲染