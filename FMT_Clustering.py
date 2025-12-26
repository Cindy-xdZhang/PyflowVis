import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import torch,logging,numpy as np
from FLowUtils.VectorField2d import *
from FLowUtils.netCDFLoader import load_UnsteadyVectorFields_netCDFOrAnalytical
from FMT_Utils.FTLE_fitting_utils import generate_FLowMap_SLICE
from FMT_Utils.FlowlinePostProcessing import AngleAwareSampling
from FMT_Utils.FMT_encoder import HierachyFMT_encoder
from sklearn.cluster import KMeans, DBSCAN
from DeepUtils.utils import EasyConfig
from FMT_Utils.FlowlinePostProcessing import LocLines,normalizeLines
from pnn.libs.flows import multi_points_vis_fast
from GuiObjcts.FlowLineRenderObject import FlowLineObject
from main import init_render


def DBSCAN_clustering(point_features:np.ndarray,eps:float,min_samples:int):
    pass


def buld_FMT_encoder(config):
    encoderConfig=config.encoder
    PathlineLtimesteps = int(config.pathlines.sampled_points_per_line)
    return HierachyFMT_encoder(encoderConfig.receptive_fields, encoderConfig.base_num_stages, encoderConfig.embed_dim, PathlineLtimesteps, encoderConfig.alpha, encoderConfig.beta)


def render_pathlines_labels(pathlines:np.ndarray,labels:np.ndarray):
    engine,camera,ObjectNameDict=init_render()
    flowlineObject=ObjectNameDict["Flowline"]
    assert flowlineObject is not None, "Flowline object not found"
    flowlineObject.RendExternalPathline(pathlines,labels)
    engine.MainLoop()
    engine.impl.shutdown()

def DBSCAN_clustering(features, eps=0.5, min_samples=5):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
    labels = clustering.labels_
    return labels

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
                _,Pathline_g,PathlineLength_g,nx,ny=generate_FLowMap_SLICE(vectorfield,physcial_time_slice,flowline_dt,max_steps,offset_dist,low_res_grid_sampling)
                #  Pathline_g shape: (ny*nx, nerbors, max_steps, 3)
                # PathlineLength_b shape: (ny*nx, nerbors)
                #Only pick pathline whose valid length is equal to max_steps
                keep_groups_full = (PathlineLength_g == max_steps).all(dim=1)
                linear_index=np.arange(nx*ny)
                valid_index=linear_index[keep_groups_full]
                ValidPathlinesNum=len(valid_index)
                #ValidPathlinesNum might not be equal to ny*nx, since some pathline may be invalid

                Temporal_sampled_Pathline_g=AngleAwareSampling(Pathline_g,LstepsPerline)

                normalizedPathlines=normalizeLines(5,Temporal_sampled_Pathline_g ,vectorfield)
                localPathlines=LocLines(5,Temporal_sampled_Pathline_g)
                normalizedLocalPathlines=normalizeLines(5,localPathlines,vectorfield)

                # inputPathlinesList=[Temporal_sampled_Pathline_g,normalizedPathlines,localPathlines,normalizedLocalPathlines]
                # titleList=["Original Pathlines","Normalized Pathlines","Local Pathlines","Normalized Local Pathlines"]
                inputPathlinesList=[Temporal_sampled_Pathline_g]
                titleList=["Original Pathlines"]
                labelsList=[]
                for Pathlines in inputPathlinesList:
                    preprocessedPathlines=Pathlines.reshape(1,ny,nx,5,LstepsPerline,3).to(device)
                    point_features_ori_rk4=Encoder(preprocessedPathlines).squeeze(0)
                    point_features_ori_rk4=point_features_ori_rk4.permute(1,2,0).cpu().detach().numpy()
                    point_features_ori_rk4=point_features_ori_rk4.reshape(nx*ny,-1)
                    #now throw away the pathline whose valid length is not equal to max_steps
                    point_features_ori_rk4=point_features_ori_rk4[valid_index]
                    # kmeans = KMeans(n_clusters=config.cluster_classes, init="k-means++",random_state=0,max_iter=1000)
                    # kmeans.fit(point_features_ori_rk4)
                    # labels_ori_rk4=kmeans.labels_
                    labels_ori_rk4 = DBSCAN_clustering(point_features_ori_rk4, eps=0.5, min_samples=5)
                    labelsList.append(labels_ori_rk4)

                #before rendering, we need to throw away the pathline whose valid length is not equal to max_steps
                # Temporal_sampled_Pathline_g=Temporal_sampled_Pathline_g[valid_index]
                # normalizedPathlines=normalizedPathlines[valid_index]
                # localPathlines=localPathlines[valid_index]
                # normalizedLocalPathlines=normalizedLocalPathlines[valid_index]
                # renderingPathlinesList=[Temporal_sampled_Pathline_g,normalizedPathlines,localPathlines,normalizedLocalPathlines]
                # multi_points_vis_fast(vectorfield,renderingPathlinesList,labelsList,time=physcial_time_slice,max_step=max_steps-1,title=titleList ,layout=[len(labelsList),1], pick="seed")


                Temporal_sampled_Pathline_g=Temporal_sampled_Pathline_g[valid_index]
                Temporal_sampled_Pathline_g=Temporal_sampled_Pathline_g.reshape(ValidPathlinesNum, 5, -1,3)
                Pathline_data=Temporal_sampled_Pathline_g[:,0,:,:].reshape(ValidPathlinesNum,-1,3).cpu().detach().numpy()
                Pathline_data_all_time=Pathline_data[:,:,2]
                render_pathlines_labels(Pathline_data,labelsList[0])


    
    print("done")




if __name__ == "__main__":

    config=EasyConfig("config/PathlineFMTclustering.yaml")
    cluster_pathlines_experiment(config)
