import math
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import netCDF4 as nc
from FLowUtils.VectorField2d import UnsteadyVectorField2D
from FLowUtils.VectorField3d import UnsteadyVectorField3D



dataset_meta={
    "channel":{
        "range":{
            "x":(0,8*np.pi),
            "y":(-1,1),
            "z":(0,3*np.pi)
        },
        "resolution":{
            "x":2048 ,
            "y":512 ,
            "z":1536 
        }
    },
    "channel5200":{
        "range":{
            "x":(0,8*np.pi),
            "y":(-1,1),
            "z":(0,3*np.pi)
        }
}
}

class JHTDB_Lodader:

    def __init__(self, auth_token: str = 'edu.jhu.pha.turbulence.testing-201406', output_path: str = './jhtdb_data'):
        """Initialize loader using givernylocal (no libJHTDB).

        Args:
            auth_token: JHTDB authorization token.
            output_path: Output directory used by givernylocal.
        """
        self.auth_token = auth_token
        self.output_path = output_path
        self._dataset_cache: Dict[str, object] = {}

    def _get_or_create_dataset(self, dataset_name: str):
        """Instantiate or fetch cached givernylocal dataset object."""
        if dataset_name in self._dataset_cache:
            return self._dataset_cache[dataset_name]
        from givernylocal.turbulence_dataset import turb_dataset
        dataset = turb_dataset(dataset_title=dataset_name, output_path=self.output_path, auth_token=self.auth_token)
        self._dataset_cache[dataset_name] = dataset
        return dataset



    # ---------- internal query helpers ----------
    def _query_getData(self, dataset_name: str, variable_name: str, times: np.ndarray, temporal_method: str,
                       spatial_method: str, spatial_operator: str, points: np.ndarray,
                       option: Optional[Union[List[float], Dict]] = None,maxQueryOnce:int=4096) -> np.ndarray:
        """按批次调用 givernylocal.getData（每批不超过 maxQueryOnce），并在每个时间点拼接后按时间维堆叠返回。

        返回值：shape 为 (T, N, 3) 的 ndarray，其中 T 为时间步数，N 为点数。
        """
        from givernylocal.turbulence_toolkit import getData
        import time
        dataset = self._get_or_create_dataset(dataset_name)

        num_points = int(points.shape[0])
        resultData = []
        for query_time in times:
            per_time_chunks = []
            for start in range(0, num_points, int(maxQueryOnce)):
                end = min(start + int(maxQueryOnce), num_points)
                print(f"querying time {query_time} from {start} to {end} of {num_points},progress {start/num_points*100}%")
                chunk_points = points[start:end]
                # 重试逻辑：getData 偶发失败（例如 HTTP 500 / result was not filled correctly）时重试
                max_retries = 8
                backoff_seconds = 1.0
                last_err = None
                for attempt in range(max_retries):
                    try:
                        chunk_result = getData(dataset, variable_name, query_time, temporal_method, spatial_method, spatial_operator, chunk_points)
                        chunk_vals = np.array(chunk_result[0]).reshape(-1, 3)
                        per_time_chunks.append(chunk_vals)
                        last_err = None
                        break
                    except Exception as e:
                        print(f"querying failed for error {e}, retrying...")
                        last_err = e
                        if attempt < max_retries - 1:
                            time.sleep(backoff_seconds)
                            backoff_seconds *= 2.0
                        else:
                            raise last_err
                        
            if len(per_time_chunks) == 1:
                result_t = per_time_chunks[0]
            else:
                result_t = np.concatenate(per_time_chunks, axis=0) if per_time_chunks else np.empty((0, 3), dtype=np.float64)
            
            resultData.append(result_t)
         

        resultData = np.stack(resultData, axis=0) if len(resultData) > 0 else np.empty((0, num_points, 3), dtype=np.float64)
        return resultData
   
  


    # ---------- grid/time generation (resolution-based) ----------
    @staticmethod
    def _generate_times(temporal_method: str, time_start: float, time_res: int, time_end: Optional[float],integerTime:bool=False) -> Tuple[np.ndarray, Optional[List[float]]]:
        """Generate time samples and getData option from resolution input.

        For 'pchip': requires time_end and time_res >= 2; returns times linspace and option [time_end, delta_t].
        For others (e.g., 'none'): requires time_res == 1 and returns single time and option None.
        """
        if integerTime:
            return np.arange(time_start, time_end+1, dtype=np.int32), None
        if str(temporal_method).lower() == 'pchip':
            if time_end is None or time_res < 2:
                temporal_method="none"
                return np.array([float(time_start)], dtype=np.float64), None
            times = np.linspace(time_start, time_end, num=int(time_res), dtype=np.float64)
            delta_t = float(times[1] - times[0])
            option = [float(time_end), float(delta_t)]
            return times, option
        # 'none' or others → single time only
        if int(time_res) != 1:
            raise ValueError("For temporal_method != 'pchip', time_res must be 1")
        return np.array([float(time_start)], dtype=np.float64), None

    @staticmethod
    def _generate_grid_3d(x_res: int, y_res: int, z_res: int,
                          x_range: Tuple[float, float], y_range: Tuple[float, float], z_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate 3D grid and flattened points array (N,3)."""
        x_vals = np.linspace(float(x_range[0]), float(x_range[1]), num=int(x_res), dtype=np.float64)
        y_vals = np.linspace(float(y_range[0]), float(y_range[1]), num=int(y_res), dtype=np.float64)
        z_vals = np.linspace(float(z_range[0]), float(z_range[1]), num=int(z_res), dtype=np.float64)
        gx, gy, gz = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')  # (X,Y,Z)
        points = np.stack([gx.ravel(order='C'), gy.ravel(order='C'), gz.ravel(order='C')], axis=1)
        return x_vals, y_vals, z_vals, points

    @staticmethod
    def _generate_grid_2d(plane: str, a_res: int, b_res: int, fixed_value: float,
                          a_range: Tuple[float, float], b_range: Tuple[float, float]) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
        """Generate 2D plane grid points given plane id: 'xy','xz','yz'.

        Returns:
            plane: normalized plane string
            a_vals, b_vals: coordinate arrays along the plane
            points: flattened points array (N,3)
        """
        p = plane.lower()
        a_vals = np.linspace(float(a_range[0]), float(a_range[1]), num=int(a_res), dtype=np.float64)
        b_vals = np.linspace(float(b_range[0]), float(b_range[1]), num=int(b_res), dtype=np.float64)
        ga, gb = np.meshgrid(a_vals, b_vals, indexing='ij')  # (A,B)
        if p == 'xy':
            x_flat = ga.ravel(order='C')
            y_flat = gb.ravel(order='C')
            z_flat = np.full_like(x_flat, float(fixed_value))
        elif p == 'xz':
            x_flat = ga.ravel(order='C')
            z_flat = gb.ravel(order='C')
            y_flat = np.full_like(x_flat, float(fixed_value))
        elif p == 'yz':
            y_flat = ga.ravel(order='C')
            z_flat = gb.ravel(order='C')
            x_flat = np.full_like(y_flat, float(fixed_value))
        else:
            raise ValueError("plane must be one of 'xy','xz','yz'")
        points = np.stack([x_flat, y_flat, z_flat], axis=1)
        return p, a_vals, b_vals, points

    @staticmethod
    def _pack_structured_grid_3d(per_time_values: List[np.ndarray], x_res: int, y_res: int, z_res: int) -> np.ndarray:
        """Pack (N,3) over time into (T, Z, Y, X, 3) using reshape + transpose (no index inference)."""
        T = len(per_time_values)
        data_5d = np.zeros((T, int(z_res), int(y_res), int(x_res), 3), dtype=np.float32)
        for t, vals in enumerate(per_time_values):
            expected = int(x_res) * int(y_res) * int(z_res)
            if vals.shape[0] != expected or vals.shape[1] != 3:
                raise ValueError("Each time slice must be (X*Y*Z, 3)")
            data_5d[t] = vals.reshape(int(x_res), int(y_res), int(z_res), 3).transpose(2, 1, 0, 3)
        return data_5d

    @staticmethod
    def _pack_structured_grid_2d(per_time_values: List[np.ndarray], a_res: int, b_res: int, plane: str) -> Tuple[np.ndarray, Tuple[str, str]]:
        """Pack plane data (N,3) into (T, H, W, 2) by plane selection.

        Plane mapping:
            'xy' → keep (u,v) → axes names ('ydim','xdim') [H=Y, W=X]
            'xz' → keep (u,w) → axes names ('zdim','xdim') [H=Z, W=X]
            'yz' → keep (v,w) → axes names ('zdim','ydim') [H=Z, W=Y]
        """
        T = len(per_time_values)
        H, W = int(b_res), int(a_res)  # map to (H=Wrt second axis, W=Wrt first axis)
        data_4d = np.zeros((T, H, W, 2), dtype=np.float32)
        if plane == 'xy':
            axis_names = ('ydim', 'xdim')
            for t, vals in enumerate(per_time_values):
                arr = vals.reshape(int(a_res), int(b_res), 3)  # (X, Y, 3)
                # transpose to (Y, X, comp)
                arr = arr.transpose(1, 0, 2)
                data_4d[t, :, :, 0] = arr[:, :, 0]
                data_4d[t, :, :, 1] = arr[:, :, 1]
        elif plane == 'xz':
            axis_names = ('zdim', 'xdim')
            for t, vals in enumerate(per_time_values):
                arr = vals.reshape(int(a_res), int(b_res), 3)  # (X, Z, 3)
                arr = arr.transpose(1, 0, 2)  # (Z, X, 3)
                data_4d[t, :, :, 0] = arr[:, :, 0]
                data_4d[t, :, :, 1] = arr[:, :, 2]
        elif plane == 'yz':
            axis_names = ('zdim', 'ydim')
            for t, vals in enumerate(per_time_values):
                arr = vals.reshape(int(a_res), int(b_res), 3)  # (Y, Z, 3) if we feed (A=Y, B=Z)
                arr = arr.transpose(1, 0, 2)  # (Z, Y, 3)
                data_4d[t, :, :, 0] = arr[:, :, 1]
                data_4d[t, :, :, 1] = arr[:, :, 2]
        else:
            raise ValueError("plane must be 'xy','xz','yz'")
        return data_4d, axis_names


    def load_2d_unsteadyFlow(self, dataset_name: str,    
                              plane: str, a_res: int, b_res: int, fixed_value: float,
                              a_range: Tuple[float, float], b_range: Tuple[float, float],
                              time_start: float,  time_end: float,time_res: int,
                              spatial_method: str="lag8",
                              spatial_operator: str="field",
                              variable_name: str="velocity",
                              integerTime:bool=False
                              ) -> UnsteadyVectorField2D:
        """Generate plane grid from resolution, query, and return UnsteadyVectorField2D."""
        p, a_vals, b_vals, points = self._generate_grid_2d(plane, a_res, b_res, fixed_value, a_range, b_range)
        temporal_method="none" if time_res==1 or integerTime else "pchip"
        times, option = self._generate_times(temporal_method, time_start, time_res, time_end,integerTime)
        per_time_values = self._query_getData(dataset_name, variable_name, times, temporal_method,
                                                 spatial_method, spatial_operator, points, option)
        data_4d, axis_names = self._pack_structured_grid_2d(per_time_values, a_res, b_res, p)
        Ydim, Xdim = data_4d.shape[1], data_4d.shape[2]
        xmin, xmax = float(a_vals.min()), float(a_vals.max())
        ymin, ymax = float(b_vals.min()), float(b_vals.max())
        tmin = float(times[0]) if times.size else 0.0
        tmax = float(times[-1]) if times.size else 0.0
        vf2d = UnsteadyVectorField2D(Xdim=Xdim, Ydim=Ydim, time_steps=int(times.size if times.size else 1),
                                     domainMinBoundary=[xmin, ymin, tmin], domainMaxBoundary=[xmax, ymax, tmax])
        vf2d.field = data_4d
        return vf2d













    def load_3d_unsteadyFlow(self, dataset_name: str, variable_name: str,
                              temporal_method: str, spatial_method: str, spatial_operator: str,
                              x_res: int, y_res: int, z_res: int,
                              time_start: float, time_res: int, time_end: Optional[float],
                              x_range: Tuple[float, float], y_range: Tuple[float, float], z_range: Tuple[float, float]) -> UnsteadyVectorField3D:
        """Generate 3D grid from resolution, query, and return UnsteadyVectorField3D."""
        times, option = self._generate_times(temporal_method, time_start, time_res, time_end)
        x_vals, y_vals, z_vals, points = self._generate_grid_3d(x_res, y_res, z_res, x_range, y_range, z_range)
        per_time_values = self._query_getData(dataset_name, variable_name, times, temporal_method,
                                              spatial_method, spatial_operator, points, option)
        data_5d = self._pack_structured_grid_3d(per_time_values, x_res, y_res, z_res)
        tmin = float(times[0]) if times.size else 0.0
        tmax = float(times[-1]) if times.size else 0.0
        vf3d = UnsteadyVectorField3D(Xdim=int(x_res), Ydim=int(y_res), Zdim=int(z_res),
                                     time_steps=int(times.size if times.size else 1),
                                     domainMinBoundary=[float(x_vals.min()), float(y_vals.min()), float(z_vals.min())],
                                     domainMaxBoundary=[float(x_vals.max()), float(y_vals.max()), float(z_vals.max())],
                                     tmin=tmin, tmax=tmax)
        vf3d.field = data_5d
        return vf3d

