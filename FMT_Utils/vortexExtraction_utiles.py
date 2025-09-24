
# ---------------- IVD Vortex Detection Datasets ----------------
class IVDVortexTrainDataset(Dataset):
    """
    训练集：
    - 输入：低分辨率 IVD tile + 低分辨率 pathlines AngleAwareSampling 预处理）
    - 标签：高分辨率 IVD 根据 ratio 百分位阈值二值化的 tile（0/1，float）
    - 采样：与 FTLEUpsamplingTrainDataset 一致，滑窗切 tile
    - 归一化：将低分辨率 IVD 使用全数据的 (ivd_min, ivd_max) 归一化到 [0,1]
    """
    def __init__(self, config, useCacheSystem: bool = True):
        all_vectorfieldsname = [name for name in config.dataset.names]
        timesliceCount = int(getattr(config.dataset, 'timesliceCount', 8))
        UPsampling = int(config.dataset.UPsampling)
        low_res_grid_sampling = float(config.dataset.low_res_grid_sampling)
        max_steps = int(config.pcds.max_iterations)
        flowline_dt = float(config.pcds.dt)
        offset_dist = float(config.pcds.offset_dist)
        LstepsPerline = int(config.pcds.sampled_points_per_line)
        patch_size = int(getattr(config.dataset, 'patchSize', 32))
        patch_stride = 4 * int(getattr(config.dataset, 'patchStride', 4))
        self.ratio = float(getattr(config.dataset, 'IVDThreshold', 0.5))

        def _tiling_starts(length: int, k: int, stride: int) -> list[int]:
            if k >= length:
                return [0]
            s = max(1, int(stride))
            starts = list(range(0, length - k + 1, s))
            last = length - k
            if starts[-1] != last:
                starts.append(last)
            return starts

        LoadCacheSuccess = False
        if useCacheSystem:
            key_obj = {
                "name": "ivd_vortex_train",
                "vectorfields": list(all_vectorfieldsname),
                "timesliceCount": int(timesliceCount),
                "UPsampling": int(UPsampling),
                "lowResGridIntervalScale": float(low_res_grid_sampling),
                "max_steps": int(max_steps),
                "dt": float(flowline_dt),
                "offset_dist": float(offset_dist),
                "LstepsPerline": int(LstepsPerline),
                "patchSize": int(patch_size),
                "patchStride": int(patch_stride),
                "ratio": float(self.ratio),
            }
            tag = stable_hash(key_obj, prefix="IVDVortexTrainingDataset_")
            cache_dir = os.path.join(config.cache_dir, "temp")
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"{tag}.npz")
            if os.path.exists(cache_path):
                try:
                    data = np.load(cache_path)
                    self.lowResIVD = torch.from_numpy(data["Data"]).float()
                    self.lowResPathlines = torch.from_numpy(data["LowResPathlines"]).float()
                    self.labels = torch.from_numpy(data["Labels"]).float()
                    self.ivd_min = float(data["ivd_min"]) if "ivd_min" in data else float(self.lowResIVD.min())
                    self.ivd_max = float(data["ivd_max"]) if "ivd_max" in data else float(self.lowResIVD.max())
                    LoadCacheSuccess = True
                    logging.info(f"[IVDVortexTrainDataset] loaded {self.lowResIVD.shape[0]} samples from cache {cache_path}")
                except Exception as e:
                    print(f"[IVDVortexTrainDataset] cache load failed: {e}. Regenerating...")

        if not LoadCacheSuccess:
            UnsteadyVectorFields = load_UnsteadyVectorFields_netCDFOrAnalytical(config.dataset.dat_dir, config.dataset.names)
            IVDE_fieldsLowRes = []     # list[Tensor patch_yx]
            lowResPathlinesData = []   # list[Tensor (patch_hw groups, nerbors, L, 3)]
            BinaryMaskHighRes = []     # list[Tensor patch_yx (hi) binary]
            ivd_min_global = float('inf')
            ivd_max_global = float('-inf')

            for i, vectorfield in enumerate(UnsteadyVectorFields):
                logging.info(f"[IVDVortexTrainDataset] generate training samples for {i+1}/{len(UnsteadyVectorFields)}...")
                tmin, tmax = float(vectorfield.tmin), float(vectorfield.tmax)
                time_window_start = float(getattr(config.dataset, 't_start', 0.1) * (tmax - tmin) + tmin)
                time_window_target = float(getattr(config.dataset, 't_target', 0.9) * (tmax - tmin) + tmin)
                sample_times = np.linspace(time_window_start, time_window_target, timesliceCount)

                for time_slice in sample_times:
                    # 低分辨率 IVD + pathlines
                    low_resIVD_field, lowResPathlines, low_res_xs, low_res_ys = generate_IVD_SLICE(
                        config, vectorfield, float(time_slice), flowline_dt, max_steps, low_res_grid_sampling
                    )
                    # 高分辨率 IVD（用于生成二值标签）
                    high_res_sampling = UPsampling * low_res_grid_sampling
                    high_resIVD_field, _, high_res_xs, high_res_ys = generate_IVD_SLICE(
                        config, vectorfield, float(time_slice), flowline_dt, max_steps, high_res_sampling
                    )

                    ivd_min_global = min(ivd_min_global, float(np.min(high_resIVD_field)))
                    ivd_max_global = max(ivd_max_global, float(np.max(high_resIVD_field)))

                    # 滑窗切 tile
                    ny_low, nx_low = low_resIVD_field.shape
                    ny_hi, nx_hi = high_resIVD_field.shape
                    ry = float(ny_hi) / float(max(1, ny_low))
                    rx = float(nx_hi) / float(max(1, nx_low))

                    row_starts = _tiling_starts(ny_low, patch_size, patch_stride)
                    col_starts = _tiling_starts(nx_low, patch_size, patch_stride)

                    # 用整幅高分辨率 IVD 的百分位阈值作为该 slice 的二值阈值
                    threshold_value = float(np.nanpercentile(high_resIVD_field, self.ratio * 100.0))

                    # pathlines 预处理（与 FTLE 一致）
                    lowResPathlinesPreprocessed = AngleAwareSampling(lowResPathlines, int(LstepsPerline))

                    for i0 in row_starts:
                        i1 = i0 + patch_size
                        for j0 in col_starts:
                            j1 = j0 + patch_size

                            hi_h = max(1, int(round((i1 - i0) * ry)))
                            hi_w = max(1, int(round((j1 - j0) * rx)))
                            hi_i0 = int(round(i0 * ry))
                            hi_j0 = int(round(j0 * rx))
                            if i0 == row_starts[-1]:
                                hi_i0 = ny_hi - hi_h
                            if j0 == col_starts[-1]:
                                hi_j0 = nx_hi - hi_w
                            hi_i1 = hi_i0 + hi_h
                            hi_j1 = hi_j0 + hi_w

                            lr_patch = torch.from_numpy(low_resIVD_field[i0:i1, j0:j1]).float()
                            hr_patch = torch.from_numpy((high_resIVD_field[hi_i0:hi_i1, hi_j0:hi_j1] >= threshold_value).astype(np.float32)).float()

                            # 选取落在低分辨率窗口内的 pathline groups
                            idx_list = []
                            for rr in range(i0, i1):
                                base = rr * nx_low
                                for cc in range(j0, j1):
                                    idx_list.append(base + cc)
                            if len(idx_list) == 0:
                                continue
                            idx_tensor = torch.as_tensor(idx_list, dtype=torch.long)
                            pl_patch = lowResPathlinesPreprocessed[idx_tensor]

                            IVDE_fieldsLowRes.append(lr_patch)
                            BinaryMaskHighRes.append(hr_patch)
                            lowResPathlinesData.append(pl_patch)

            self.lowResIVD = torch.stack(IVDE_fieldsLowRes)
            self.lowResPathlines = torch.stack(lowResPathlinesData)
            self.labels = torch.stack(BinaryMaskHighRes)
            self.ivd_min = float(ivd_min_global if np.isfinite(ivd_min_global) else float(self.lowResIVD.min()))
            self.ivd_max = float(ivd_max_global if np.isfinite(ivd_max_global) else float(self.lowResIVD.max()))

            if useCacheSystem:
                try:
                    key_obj = {
                        "name": "ivd_vortex_train",
                        "vectorfields": list(all_vectorfieldsname),
                        "timesliceCount": int(timesliceCount),
                        "UPsampling": int(UPsampling),
                        "lowResGridIntervalScale": float(low_res_grid_sampling),
                        "max_steps": int(max_steps),
                        "dt": float(flowline_dt),
                        "offset_dist": float(offset_dist),
                        "LstepsPerline": int(LstepsPerline),
                        "patchSize": int(patch_size),
                        "patchStride": int(patch_stride),
                        "ratio": float(self.ratio),
                    }
                    tag = stable_hash(key_obj, prefix="IVDVortexTrainingDataset_")
                    cache_dir = os.path.join(config.cache_dir, "temp")
                    os.makedirs(cache_dir, exist_ok=True)
                    cache_path = os.path.join(cache_dir, f"{tag}.npz")
                    np.savez(cache_path,
                             Data=self.lowResIVD.detach().cpu().numpy().astype(np.float32),
                             Labels=self.labels.detach().cpu().numpy().astype(np.float32),
                             LowResPathlines=self.lowResPathlines.detach().cpu().numpy().astype(np.float32),
                             ivd_min=np.array([self.ivd_min], dtype=np.float32),
                             ivd_max=np.array([self.ivd_max], dtype=np.float32))
                    logging.info(f"[IVDVortexTrainDataset] saved {self.lowResIVD.shape[0]} samples to cache {cache_path}")
                except Exception as e:
                    print(f"[IVDVortexTrainDataset] cache save failed: {e}")

        # 归一化低分辨率 IVD 到 [0,1]
        self.lowResIVD = self.lowResIVD.clamp(self.ivd_min, self.ivd_max)
        denom = max(1e-12, (self.ivd_max - self.ivd_min))
        self.lowResIVD = (self.lowResIVD - self.ivd_min) / denom

    def __len__(self):
        return len(self.lowResIVD)

    def __getitem__(self, idx):
        return (self.lowResIVD[idx], self.lowResPathlines[idx]), self.labels[idx]


class IVDVortexTestDataset(Dataset):
    """
    测试集：
    - 针对每个流场的若干时间切片，生成：
        低分辨率 IVD 切片、对应的低分辨率 AngleAwareSampling 预处理）、高分辨率 IVD 的二值化切片
    - 模型测试阶段可按 tile 滑动预测并合并（外部使用与 FTLE 测试相同的滑窗逻辑）
    """
    def __init__(self, config, useCacheSystem: bool = True):
        names_cfg = config['test_vectorfield'] if 'test_vectorfield' in config else [config.dataset.names[0]]
        test_vectorfield_names = names_cfg if isinstance(names_cfg, (list, tuple)) else [names_cfg]
        timesliceCount = int(getattr(config.dataset, 'timesliceCount', 8)) // 2
        UPsampling = int(config.dataset.UPsampling)
        low_res_grid_sampling = float(config.dataset.low_res_grid_sampling)
        max_steps = int(config.pcds.max_iterations)
        flowline_dt = float(config.pcds.dt)
        LstepsPerline = int(config.pcds.sampled_points_per_line)
        self.ratio = float(getattr(config.dataset, 'IVDThreshold', 0.5))

        # 容器
        self.lowResIVD_list = []
        self.lowResPathlines_list = []
        self.binaryMask_list = []
        self.ivd_min = float('inf')
        self.ivd_max = float('-inf')

        # cache
        LoadCacheSuccess = False
        if useCacheSystem:
            key_obj = {
                "name": "ivd_vortex_test",
                "vectorfields": list(map(str, test_vectorfield_names)),
                "timesliceCount": int(timesliceCount),
                "UPsampling": int(UPsampling),
                "lowResGridIntervalScale": float(low_res_grid_sampling),
                "max_steps": int(max_steps),
                "dt": float(flowline_dt),
                "LstepsPerline": int(LstepsPerline),
                "ratio": float(self.ratio),
            }
            tag = stable_hash(key_obj, prefix="IVDVortexTestDataset_")
            cache_dir = os.path.join(config.cache_dir, "temp")
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"{tag}.npz")
            if os.path.exists(cache_path):
                try:
                    data = np.load(cache_path, allow_pickle=True)
                    self.lowResIVD_list = list(data["lowResIVD_list"])  # object array -> list
                    self.lowResPathlines_list = list(data["lowResPathlines_list"])  # stored as object
                    self.binaryMask_list = list(data["binaryMask_list"])  # object array -> list
                    self.ivd_min = float(data["ivd_min"]) if "ivd_min" in data else 0.0
                    self.ivd_max = float(data["ivd_max"]) if "ivd_max" in data else 1.0
                    LoadCacheSuccess = True
                    logging.info(f"[IVDVortexTestDataset] loaded {len(self.lowResIVD_list)} samples from cache {cache_path}")
                except Exception as e:
                    logging.info(f"[IVDVortexTestDataset] cache load failed: {e}. Regenerating...")

        if not LoadCacheSuccess:
            for vf_name in test_vectorfield_names:
                vf_obj = load_UnsteadyVectorFields_netCDFOrAnalytical(config.dataset.dat_dir, vf_name)[0]
                if vf_obj is None:
                    logging.info(f"[IVDVortexTestDataset] load {vf_name} failed. Skip.")
                    continue

                tmin, tmax = float(vf_obj.tmin), float(vf_obj.tmax)
                time_window_start = float(getattr(config.dataset, 't_start', 0.1) * (tmax - tmin) + tmin)
                time_window_target = float(getattr(config.dataset, 't_target', 0.9) * (tmax - tmin) + tmin)
                sample_times = np.linspace(time_window_start, time_window_target, num=timesliceCount)

                for time_slice in sample_times:
                    low_resIVD_field, lowResPathlines, _, _ = generate_IVD_SLICE(
                        config, vf_obj, float(time_slice), flowline_dt, max_steps, low_res_grid_sampling
                    )
                    high_res_sampling = UPsampling * low_res_grid_sampling
                    high_resIVD_field, _, _, _ = generate_IVD_SLICE(
                        config, vf_obj, float(time_slice), flowline_dt, max_steps, high_res_sampling
                    )

                    self.ivd_min = min(self.ivd_min, float(np.min(high_resIVD_field)))
                    self.ivd_max = max(self.ivd_max, float(np.max(high_resIVD_field)))

                    threshold_value = float(np.nanpercentile(high_resIVD_field, self.ratio * 100.0))
                    binary_hi = (high_resIVD_field >= threshold_value).astype(np.float32)

                    # pathlines 预处理
                    lowResPathlinesPreprocessed = AngleAwareSampling(lowResPathlines, int(LstepsPerline))

                    self.lowResIVD_list.append(low_resIVD_field.astype(np.float32))
                    self.lowResPathlines_list.append(lowResPathlinesPreprocessed)
                    self.binaryMask_list.append(binary_hi.astype(np.float32))

            if useCacheSystem:
                try:
                    tag = stable_hash({
                        "name": "ivd_vortex_test",
                        "vectorfields": list(map(str, test_vectorfield_names)),
                        "timesliceCount": int(timesliceCount),
                        "UPsampling": int(UPsampling),
                        "lowResGridIntervalScale": float(low_res_grid_sampling),
                        "max_steps": int(max_steps),
                        "dt": float(flowline_dt),
                        "LstepsPerline": int(LstepsPerline),
                        "ratio": float(self.ratio),
                    }, prefix="IVDVortexTestDataset_")
                    cache_dir = os.path.join(config.cache_dir, "temp")
                    os.makedirs(cache_dir, exist_ok=True)
                    cache_path = os.path.join(cache_dir, f"{tag}.npz")

                    # 将 list 保存为 object 数组
                    np.savez(cache_path,
                             lowResIVD_list=np.array(self.lowResIVD_list, dtype=object),
                             lowResPathlines_list=np.array(self.lowResPathlines_list, dtype=object),
                             binaryMask_list=np.array(self.binaryMask_list, dtype=object),
                             ivd_min=np.array([self.ivd_min], dtype=np.float32),
                             ivd_max=np.array([self.ivd_max], dtype=np.float32))
                    logging.info(f"[IVDVortexTestDataset] saved {len(self.lowResIVD_list)} samples to cache {cache_path}")
                except Exception as e:
                    logging.info(f"[IVDVortexTestDataset] cache save failed: {e}")

        # 将低分辨率 IVD 归一化参数暴露，外部测试时可按需使用
        if not np.isfinite(self.ivd_min) or not np.isfinite(self.ivd_max) or self.ivd_max <= self.ivd_min:
            self.ivd_min, self.ivd_max = 0.0, 1.0

    def __len__(self):
        return len(self.lowResIVD_list)

    def __getitem__(self, idx):
        # 返回原始低分辨率 IVD（未归一化，供外部使用 train-set 统计或本数据集统计归一化）
        lr_ivd = torch.from_numpy(self.lowResIVD_list[idx]).float()
        pl = self.lowResPathlines_list[idx].float() if isinstance(self.lowResPathlines_list[idx], torch.Tensor) else torch.from_numpy(self.lowResPathlines_list[idx]).float()
        mask = torch.from_numpy(self.binaryMask_list[idx]).float()
        return (lr_ivd, pl), mask
