import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import torch,logging,numpy as np
from torch.utils.data import Dataset, DataLoader
from FLowUtils.VectorField2d import *
from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import load_UnsteadyVectorFields_general,relocate_flow2d_dataset_folder
from FMT_Utils.FTLE_fitting_utils import generate_FLowMap_SLICE
from FMT_Utils.FlowlinePostProcessing import AngleAwareSampling
from FMT_Utils.FMT_encoder import *
from sklearn.cluster import KMeans, DBSCAN
from DeepUtils.utils import EasyConfig
from FMT_Utils.FlowlinePostProcessing import LocLines,normalizeLines
from pnn.libs.flows import multi_points_vis_fast
# def render_pathlines_labels(pathlines:np.ndarray,labels:np.ndarray):
#     from main import init_render
#     from GuiObjcts.FlowLineRenderObject import FlowLineObject
#     engine,camera,ObjectNameDict=init_render()
#     flowlineObject=ObjectNameDict["Flowline"]
#     assert flowlineObject is not None, "Flowline object not found"
#     flowlineObject.RendExternalPathline(pathlines,labels)
#     engine.MainLoop()
#     engine.impl.shutdown()

class FMTClusteringDataset(Dataset):
    """
    Minimal demonstration Dataset for FMT pathline clustering.

    Pipeline (no caching, no labels, no train/val split):
      1) Load 2D unsteady vector fields (NetCDF / Amira) via
         `load_UnsteadyVectorFields_general` after `relocate_flow2d_dataset_folder`.
      2) For each vector field and each sampled physical time in the
         [t_start, t_target] window:
            a. Seed a regular 2D grid (controlled by `dataset.grid_sampling`)
               and integrate pathline-cross primitives with RK4
               (`generate_FLowMap_SLICE` -> shape (ny*nx, nerbors=5, max_steps, 3)).
            b. Drop primitives whose lines did not reach `max_steps`.
            c. Temporally re-sample each primitive from `max_steps` -> `LstepsPerline`
               points via `AngleAwareSampling` (keeps perceptually salient frames).
      3) Each item is a single pathline-cross primitive:
            shape (nerbors=5, LstepsPerline, 3), with 3 = (x, y, t).

    Apply `LocLines` / `normalizeLines` downstream if you want localized or
    domain-normalized variants (kept out of the Dataset to keep this demo lean).

    Example:
        >>> config = EasyConfig("config/PathlineFMTclustering.yaml")
        >>> ds = FMTClusteringDataset(config)
        >>> loader = DataLoader(ds, batch_size=64, shuffle=True)
        >>> for batch in loader:           # batch: (B, 5, L, 3)
        ...     ...
    """

    def __init__(self, config):
        relocate_flow2d_dataset_folder(config)
        vf_objs = load_UnsteadyVectorFields_general(config.dataset.dat_dir, config.dataset.names)

        timesliceCount          = int(config.dataset.timesliceCount)
        low_res_grid_sampling   = float(config.dataset.grid_sampling)
        max_steps               = int(config.pathlines.max_iterations)
        flowline_dt             = float(config.pathlines.dt)
        offset_dist             = float(config.pathlines.offset_dist)
        time_window_start_ratio = float(config.dataset.t_start)
        time_window_target_ratio= float(config.dataset.t_target)
        LstepsPerline           = int(config.pathlines.sampled_points_per_line)

        primitives_per_slice = []   # list[Tensor (n_valid, 5, L, 3)]
        sample_meta = []            # list[(vf_index, physical_time)] per sample
        groups      = []            # list[(vf_index, physical_time, index_tensor)]
        running     = 0             # running offset into self.pathlines

        for vf_idx, vectorfield in enumerate(vf_objs):
            t0 = float(time_window_start_ratio  * (vectorfield.tmax - vectorfield.tmin) + vectorfield.tmin)
            t1 = float(time_window_target_ratio * (vectorfield.tmax - vectorfield.tmin) + vectorfield.tmin)
            timeslice = np.linspace(t0, t1, timesliceCount)

            for physical_time in timeslice:
                _, Pathline_g, PathlineLength_g, nx, ny = generate_FLowMap_SLICE(
                    vectorfield, float(physical_time),
                    flowline_dt, max_steps, offset_dist, low_res_grid_sampling,
                )
                # Pathline_g       : (ny*nx, 5, max_steps, 3)
                # PathlineLength_g : (ny*nx, 5)

                keep_full   = (PathlineLength_g == max_steps).all(dim=1)
                linear_idx  = np.arange(nx * ny)
                valid_index = linear_idx[keep_full]
                if valid_index.size == 0:
                    logging.warning(f"[FMTClusteringDataset] vf={vf_idx} t={physical_time:.4f}: no valid primitives, skipped.")
                    continue

                # (ny*nx, 5, L, 3); only keep primitives that fully integrated.
                temporal_sampled = AngleAwareSampling(Pathline_g, LstepsPerline)
                valid_primitives = temporal_sampled[valid_index].detach().cpu().float()
                n = valid_primitives.shape[0]

                primitives_per_slice.append(valid_primitives)
                sample_meta.extend([(vf_idx, float(physical_time))] * n)
                groups.append((vf_idx, float(physical_time),
                               torch.arange(running, running + n, dtype=torch.long)))
                running += n

        if not primitives_per_slice:
            raise RuntimeError(
                "[FMTClusteringDataset] no valid pathlines produced; "
                "check dataset.grid_sampling / pathlines.max_iterations / pathlines.offset_dist."
            )

        # (N, nerbors=5, LstepsPerline, 3); 3 = (x, y, t)
        self.pathlines    = torch.cat(primitives_per_slice, dim=0).contiguous()
        self.sample_meta  = sample_meta
        self.groups       = groups       # iterate per (vector field, physical time) batch
        self.vf_objs      = vf_objs      # kept so downstream can render in original field
        self.LstepsPerline = LstepsPerline

        logging.info(f"[FMTClusteringDataset] built {self.pathlines.shape[0]} pathline primitives "
                     f"in {len(self.groups)} (vf, t) groups, "
                     f"shape per item = {tuple(self.pathlines.shape[1:])} (nerbors, L, xyt).")

    def __len__(self):
        return self.pathlines.shape[0]

    def __getitem__(self, idx):
        return self.pathlines[idx]


def DBSCAN_clustering(point_features:np.ndarray,eps:float,min_samples:int):
    pass


def buld_FMT_encoder(config):
    encoderConfig=config.encoder
    PathlineLtimesteps = int(config.pathlines.sampled_points_per_line)
    return HierachyFMT_encoder(encoderConfig.receptive_fields, encoderConfig.base_num_stages, encoderConfig.embed_dim, PathlineLtimesteps, encoderConfig.alpha, encoderConfig.beta)

def DBSCAN_clustering(features, eps=0.5, min_samples=5):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
    labels = clustering.labels_
    return labels

def cluster_pathlines_experiment(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    logging.info(f"torch.version.cuda: {torch.version.cuda}")

    max_steps     = int(config.pathlines.max_iterations)
    LstepsPerline = int(config.pathlines.sampled_points_per_line)
    NUM_STAGES    = int(config.encoder.base_num_stages)
    embed_dim     = int(config.encoder.embed_dim)
    alpha         = float(config.encoder.alpha)
    beta          = float(config.encoder.beta)
    temporal_head = None

    # Build the demo dataset: loads vector fields, integrates pathlines, applies
    # AngleAwareSampling, and groups primitives by (vector field, physical time).
    dataset = FMTClusteringDataset(config)

    Encoder = FMT(LstepsPerline, NUM_STAGES, embed_dim, alpha, beta, temporal_head)
    Encoder.to(device)

    titleList = ["Original Pathlines", "Normalized Pathlines",
                 "Local Pathlines",   "Normalized Local Pathlines"]

    with torch.no_grad():
        for vf_idx, physical_time, idxs in dataset.groups:
            vectorfield = dataset.vf_objs[vf_idx]

            # (n_valid, 5, L, 3) — already filtered to fully-integrated primitives.
            primitives = dataset.pathlines[idxs]
            n_valid    = primitives.shape[0]

            # Build the 4 augmented views used for clustering / comparison.
            normalizedPathlines      = normalizeLines(5, primitives, vectorfield)
            localPathlines           = LocLines(5, primitives)
            normalizedLocalPathlines = normalizeLines(5, localPathlines, vectorfield)
            inputPathlinesList = [primitives, normalizedPathlines,
                                  localPathlines, normalizedLocalPathlines]

            labelsList = []
            for Pathlines in inputPathlinesList:
                # (n_valid, 5*L, 3) -> encoder
                points_N3 = Pathlines.reshape(n_valid, 5 * LstepsPerline, 3).to(device)
                points_3N = points_N3.permute(0, 2, 1).contiguous()
                feats     = Encoder(points_N3, points_3N).squeeze(0)
                feats_np  = feats.reshape(n_valid, -1).cpu().detach().numpy()

                kmeans = KMeans(n_clusters=config.cluster_classes,
                                init="k-means++", random_state=0, max_iter=1000)
                kmeans.fit(feats_np)
                labelsList.append(kmeans.labels_)

            renderingPathlinesList = [primitives] * len(inputPathlinesList)
            multi_points_vis_fast(
                vectorfield, renderingPathlinesList, labelsList,
                time=physical_time, max_step=max_steps - 1,
                title=titleList, layout=[len(labelsList), 1], pick="seed",
            )

    print("done")




if __name__ == "__main__":

    config=EasyConfig("config/PathlineFMTclustering.yaml")
    cluster_pathlines_experiment(config)
