#=====================================================================================================================
# this file is for  loading machine learning vector field 2d dataset
# It depdens on  the AmiraLoader class in FLowUtils/flowDatasetUtils/NetCDF_AmiraLoader.py
#  Works with datasets like: https://www.research-collection.ethz.ch/entities/researchdata/52b7a501-c669-411b-9265-f3f039a28dbe
#Original specification of the dataset:
# """
# Special loader for Amira Mesh 2.1 (*.am) unsteady 2D vector fields.
# Works with datasets like: https://www.research-collection.ethz.ch/entities/researchdata/52b7a501-c669-411b-9265-f3f039a28dbe

# Assumptions:
#   - Header line like: 'define Lattice NX NY NT'
#   - Parameters contain: 'BoundingBox xmin xmax ymin ymax tmin tmax'
#   - Lattice declaration: 'Lattice { float[2] Velocity } @<tag>' (two components interleaved)
#   - Binary block starts after a line '@<tag>' and is little-endian float32 in x-fastest order, then y, then t.

# Returns UnsteadyVectorField2D with field shaped [T, Y, X, 2].
#=====================================================================================================================
# """Resampling the dataset """
#  Due to the large size of the dataset(16TB), we need to resample the dataset, which is controled  by the global resample_ratio_Spatial, resample_ratio_Time
#=====================================================================================================================
#Global variables:
# resample_ratio_Spatial: 0.5
# resample_ratio_Time: 0.25
#=====================================================================================================================

resample_ratio_Spatial=0.5
resample_ratio_Time=0.5

from pathlib import Path
import sys
if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

# from GuiObjcts.ActiveFieldObject import *
from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import AmiraLoader
import os,time,requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry




def download_with_progress(url: str, dest_path: str, resume: bool = True, chunk_size: int = 8 * 1024 * 1024, timeout: int = 60, retries: int = 5) -> None:
    """Download a file with progress bar, retry and resume support.

    Parameters
    ----------
    url : str
        Source URL.
    dest_path : str
        Final destination file path.
    resume : bool
        Whether to enable resume via HTTP Range and .part file.
    chunk_size : int
        Read size per chunk in bytes (default 8 MiB).
    timeout : int
        Per-request timeout in seconds.
    retries : int
        Total retry attempts for transient errors.
    """

    # Lazy import tqdm to avoid hard dependency
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        tqdm = None  # type: ignore

    session = requests.Session()
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET"],
        raise_on_status=False,
    )
    session.mount("http://", HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10))
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10))

    if os.path.exists(dest_path):
        # If file already exists and server provides size, skip if complete
        try:
            head = session.head(url, allow_redirects=True, timeout=timeout)
            if head.ok and head.headers.get("Content-Length"):
                total_size_remote = int(head.headers["Content-Length"])  # remaining for full download
                local_size = os.path.getsize(dest_path)
                if local_size == total_size_remote:
                    return
        except Exception:
            # If HEAD fails, conservatively skip re-download
            return

    temp_path = dest_path + ".part"
    existing = os.path.getsize(temp_path) if resume and os.path.exists(temp_path) else 0

    headers = {}
    accept_ranges = None
    # Try to detect total size and range support
    try:
        head = session.head(url, allow_redirects=True, timeout=timeout)
        if head.ok:
            accept_ranges = head.headers.get("Accept-Ranges")
            # When resuming, only use existing if server supports ranges
            if not (accept_ranges and accept_ranges.lower() == "bytes"):
                existing = 0
            if head.headers.get("Content-Length"):
                total_size_remote = int(head.headers["Content-Length"])  # full size
            else:
                total_size_remote = None
        else:
            total_size_remote = None
    except Exception:
        total_size_remote = None

    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    with session.get(url, headers=headers, stream=True, timeout=timeout) as r:
        r.raise_for_status()

        # Remaining size for this response (may be None)
        remaining_size = r.headers.get("Content-Length")
        remaining_size = int(remaining_size) if remaining_size is not None else None
        total_for_progress = (existing + remaining_size) if remaining_size is not None else total_size_remote

        # Always write into .part file, then move when completed
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        mode = "ab" if existing > 0 else "wb"
        with open(temp_path, mode) as f:
            if tqdm is not None:
                bar = tqdm(total=total_for_progress, initial=existing, unit="B", unit_scale=True, unit_divisor=1024, desc=os.path.basename(dest_path))
            else:
                bar = None
            try:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    if bar is not None:
                        bar.update(len(chunk))
            finally:
                if bar is not None:
                    bar.close()

    # Finalize: rename .part to final path
    os.replace(temp_path, dest_path)



#Gunther_GerrisSolver_ML_FlowMap_Dataset
#A Fluid Flow Data Set for Machine Learning and its Application to Neural Flow Map Interpolation
# Jakob Jakob, Markus Gross and Tobias Günther
# IEEE Transactions on Visualization and Computer Graphics (IEEE Visualization), 2021


def downloadGunther_resample_amira_dataset(Id_start: int = 0, Id_end: int = 1000, Id_step: int = 10):
    temp_amira_folder="C:\\Users\\xingdi\\OneDrive - KAUST\\WorkingInProcess\\FLowVisAssets\\flowData2D\\GerrisFlowSolverDataTemp"
    resampled_amira_folder="C:\\Users\\xingdi\\OneDrive - KAUST\\WorkingInProcess\\FLowVisAssets\\flowData2D\\Gunther_GerrisSolver_ML_FlowMap_Dataset"
    if not os.path.exists(temp_amira_folder):
        os.makedirs(temp_amira_folder)
    if not os.path.exists(resampled_amira_folder):
        os.makedirs(resampled_amira_folder)


    max_retries = 5
    retry_delay = 10  # 秒

    for id in range(Id_start, Id_end,Id_step):
        id_str = f"{id:04d}"
        dest_temp_file = os.path.join(temp_amira_folder, f"{id_str}.am")
        dest_nc_file = os.path.join(resampled_amira_folder, f"{id_str}.am")
        if os.path.exists(dest_nc_file):
            continue

        # 下载部分加重试和异常处理
        if not os.path.exists(dest_temp_file):
            url = f"https://libdrive.ethz.ch/index.php/s/lv7dV40oYlkWJiC/download?path=%2F&files={id_str}.am"
            for attempt in range(max_retries):
                try:
                    download_with_progress(
                        url, dest_temp_file, resume=True, chunk_size=8 * 1024 * 1024, timeout=60, retries=10
                    )
                    break
                except Exception as e:
                    print(f"[{id_str}] 下载失败: {e}，第 {attempt+1}/{max_retries} 次重试")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        print(f"[{id_str}] 多次重试后仍然下载失败，跳过该文件。")
                        continue  # 跳到下一个id
            if not os.path.exists(dest_temp_file):
                print(f"[{id_str}] 文件依然不存在，跳过。")
                continue

        # 加载、重采样、保存部分加异常处理和重试
        for attempt in range(max_retries):
            try:
                vectorField2d = AmiraLoader.load_vector_field2d(dest_temp_file)
                if vectorField2d is None:
                    raise RuntimeError(f"[{id_str}] 加载Amira文件失败: {dest_temp_file}")
                vectorField2d.resample2UnsteadyField((
                    int(vectorField2d.Xdim * resample_ratio_Spatial),
                    int(vectorField2d.Ydim * resample_ratio_Spatial),
                    int(vectorField2d.time_steps * resample_ratio_Time)
                ))
                AmiraLoader.save_vector_field2d(dest_nc_file, vectorField2d)
                # os.remove(os.path.join(temp_amira_folder, f"{id_str}.am"))
                break
            except Exception as e:
                print(f"[{id_str}] 处理/保存失败: {e}，第 {attempt+1}/{max_retries} 次重试")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    print(f"[{id_str}] 多次重试后依然失败，跳过该文件。")

        if os.path.exists(dest_temp_file) and os.path.exists(dest_nc_file):
            os.remove(dest_temp_file)






if __name__ == "__main__":
    downloadGunther_resample_amira_dataset()