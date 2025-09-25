import torch,logging,numpy as np
from FLowUtils.VectorField2d import *
from FLowUtils.netCDFLoader import load_UnsteadyVectorFields_netCDFOrAnalytical
from FMT_Utils.FTLE_fitting_utils import generate_FLowMap_SLICE
from FMT_Utils.FlowlinePostProcessing import AngleAwareSampling
from FMT_Utils.FMT_encoder import HierachyFMT_encoder
from sklearn.cluster import KMeans
from DeepUtils.utils import EasyConfig
from FMT_Utils.FlowlinePostProcessing import LocLines,normalizeLines
from pnn.libs.flows import multi_points_vis_fast

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

                normalizedPathlines=normalizeLines(5,Temporal_sampled_Pathline_g ,vectorfield)
                localPathlines=LocLines(5,Temporal_sampled_Pathline_g)
                normalizedLocalPathlines=normalizeLines(5,localPathlines,vectorfield)

                inputPathlinesList=[Temporal_sampled_Pathline_g,normalizedPathlines,localPathlines,normalizedLocalPathlines]
                titleList=["Original Pathlines","Normalized Pathlines","Local Pathlines","Normalized Local Pathlines"]
                labelsList=[]
                for Pathlines in inputPathlinesList:
                    preprocessedPathlines=Pathlines.reshape(1,ny,nx,5,LstepsPerline,3).to(device)
                    point_features_ori_rk4=Encoder(preprocessedPathlines).squeeze(0)
                    point_features_ori_rk4=point_features_ori_rk4.permute(1,2,0).cpu().detach().numpy()
                    point_features_ori_rk4=point_features_ori_rk4.reshape(nx*ny,-1)
                    kmeans = KMeans(n_clusters=config.cluster_classes, init="k-means++",random_state=0,max_iter=1000)
                    kmeans.fit(point_features_ori_rk4)
                    labels_ori_rk4=kmeans.labels_
                    labelsList.append(labels_ori_rk4)



                renderingPathlinesList=[ Temporal_sampled_Pathline_g ]*len(labelsList)
                multi_points_vis_fast(vectorfield,renderingPathlinesList,labelsList,time=physcial_time_slice,max_step=max_steps-1,title=titleList ,layout=[len(labelsList),1], pick="seed")
    
    print("done")




if __name__ == "__main__":

    config=EasyConfig("config/PathlineFMTclustering.yaml")
    cluster_pathlines_experiment(config)
