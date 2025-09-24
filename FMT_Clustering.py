import torch,logging,numpy as np
from FLowUtils.VectorField2d import *
from FLowUtils.netCDFLoader import load_UnsteadyVectorFields_netCDFOrAnalytical
from FMT_Utils.FTLE_fitting_utils import generate_FLowMap_SLICE
from FMT_Utils.FlowlinePostProcessing import AngleAwareSampling
from FMT_Utils.FMT_encoder import HierachyFMT_encoder
from sklearn.cluster import KMeans
from DeepUtils.utils import EasyConfig
from FMT_Utils.FlowlinePostProcessing import LocLines,normalizeLines

def visualize_clustering_result(vectorfield, pathlines: torch.Tensor, labels: np.ndarray, save_path: str | None = None, linewidth: float = 0.8, markersize: float = 1.0):
    """
    可视化路径线聚类结果。
    输入:
      - vectorfield: 含有 tmin/tmax 等元数据的对象（仅用于时间信息，可选）
      - pathlines: [B, Y, X, K, L, 3]
      - labels:    [X*Y] 的聚类标签
      - save_path: 结果保存路径（.png/.pdf），为 None 则直接显示
    规则:
      - 取每个 seeding 的主路径线（K 中第 0 条）进行着色绘制
      - 颜色按 labels 映射
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    with torch.no_grad():
        pl = pathlines
        if pl.dim() ==5:  # [ N, K, L, 3]
            Y,X, K, L, D = pl.shape
        else:
            raise ValueError(f"Unsupported pathlines shape: {tuple(pl.shape)}")

        assert D == 3, "pathlines last dim must be 3"
        pl_vis = pl.reshape(Y*X,K,L,3) # 取 batch 0 可视化: [N,K,L,3]
        N0 = pl_vis.shape[0]
        assert N0 == labels.shape[0], f"labels length {labels.shape[0]} != N {N0}"

        # colormap
        classes = int(labels.max()) + 1 if labels.size > 0 else 1
        colors = cm.get_cmap('tab20', classes)

        fig, ax = plt.subplots(figsize=(8, 8))
        for i in range(0,N0,2):
            c = colors(labels[i] % classes)
            # if draw == "curve":
            #     # 取主线 K=0；坐标使用前两维 (x,y)
            #     xy = pl_vis[i, 0, :, :2].cpu().numpy()  # [L,2]
            #     ax.plot(xy[:, 0], xy[:, 1], color=c, linewidth=linewidth, alpha=0.95)
            # else:
            # 仅起点/指定时间步
      
            p = pl_vis[i, 0, 0, :2].cpu().numpy()  # [2]
            ax.scatter(p[0], p[1], color=c, s=markersize, alpha=0.95)

        ax.set_aspect('equal')
        ax.set_title('Pathlines Clustering Result')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.2)

        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', dpi=200)
            plt.close(fig)
        else:
            plt.show()


def buld_FMT_encoder(config):
    encoderConfig=config.encoder
    PathlineLtimesteps = int(config.pathlines.sampled_points_per_line)
    return HierachyFMT_encoder(encoderConfig.receptive_fields, encoderConfig.base_num_stages, encoderConfig.embed_dim, PathlineLtimesteps, encoderConfig.alpha, encoderConfig.beta)


def cluster_pathlines_experiment(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")  # Should return True
    logging.info(f"torch.version.cuda: {torch.version.cuda}")         # Should print the CUDA version PyTorch was built with

    vf_name = config.dataset.names
    vf_objs = load_UnsteadyVectorFields_netCDFOrAnalytical(config.dataset.dat_dir,vf_name)
    timesliceCount=config.dataset.timesliceCount
    low_res_grid_sampling=float(config.dataset.grid_sampling)
    max_steps: int=config.pathlines.max_iterations
    flowline_dt: float=config.pathlines.dt
    offset_dist: float =float(config.pathlines.offset_dist)
    time_window_start_ratio=float(config.dataset.t_start)
    time_window_target_ratio=float(config.dataset.t_target)
    LstepsPerline = int(config.pathlines.sampled_points_per_line)

    Encoder=buld_FMT_encoder(config)
    Encoder.to(device)
    with torch.no_grad():
        for vectorfield in vf_objs:
            time_window_start = float(time_window_start_ratio * (vectorfield.tmax - vectorfield.tmin) + vectorfield.tmin)
            time_window_target = float(time_window_target_ratio * (vectorfield.tmax - vectorfield.tmin) + vectorfield.tmin)
            timeslice=np.linspace(time_window_start, time_window_target, timesliceCount)

            for physcial_time_slice in timeslice:
                true_grid,Pathline_g,nx,ny=generate_FLowMap_SLICE(vectorfield,physcial_time_slice,flowline_dt,max_steps,offset_dist,low_res_grid_sampling)
                Temporal_sampled_Pathline_g=AngleAwareSampling(Pathline_g,LstepsPerline)
                preprocessedPathlines=LocLines(5,Temporal_sampled_Pathline_g)
                preprocessedPathlines=normalizeLines(5,preprocessedPathlines,vectorfield)

                preprocessedPathlines=preprocessedPathlines.reshape(1,ny,nx,5,LstepsPerline,3).to(device)
                point_features_ori_rk4=Encoder(preprocessedPathlines).squeeze(0)
                point_features_ori_rk4=point_features_ori_rk4.permute(1,2,0).cpu().detach().numpy()
                point_features_ori_rk4=point_features_ori_rk4.reshape(nx*ny,-1)
                kmeans = KMeans(n_clusters=config.cluster_classes, n_init=100,random_state=0,algorithm="elkan")

                kmeans.fit(point_features_ori_rk4)

                visualize_clustering_result(vectorfield,Temporal_sampled_Pathline_g.reshape(ny,nx,5,LstepsPerline,3),kmeans.labels_)
    
    print("done")




if __name__ == "__main__":

    config=EasyConfig("config/PathlineFMTclustering.yaml")
    cluster_pathlines_experiment(config)
