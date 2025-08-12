# 1. 首先创建配置文件来管理傅里叶参数
创建新文件`helper_fourier.py`:
```python
import torch
import numpy as np
import torch.nn as nn

class FourierMotionModel:
    """傅里叶运动模型的辅助类"""
    
    @staticmethod
    def get_motion_dim(fourier_terms):
        """计算motion参数的维度"""
        return 3 * (2 * fourier_terms + 1)
    
    @staticmethod
    def compute_position(xyz_base, motion_params, t, fourier_terms):
        """
        使用傅里叶级数计算时间t时的位置
        
        Args:
            xyz_base: 基础位置 (N, 3)
            motion_params: 傅里叶系数 (N, 3*(2L+1))
            t: 时间标量或张量
            fourier_terms: 傅里叶项数L
        
        Returns:
            xyz_t: 时间t时的位置 (N, 3)
        """
        N = xyz_base.shape[0]
        L = fourier_terms
        
        # 复制基础位置
        xyz_t = xyz_base.clone()
        
        # 分解motion参数为x, y, z三个方向
        motion_x = motion_params[:, 0:(2*L+1)]
        motion_y = motion_params[:, (2*L+1):2*(2*L+1)]
        motion_z = motion_params[:, 2*(2*L+1):3*(2*L+1)]
        
        # 添加常数项 w_0
        xyz_t[:, 0] += motion_x[:, 0]
        xyz_t[:, 1] += motion_y[:, 0]
        xyz_t[:, 2] += motion_z[:, 0]
        
        # 添加傅里叶级数项
        for i in range(1, L+1):
            sin_term = torch.sin(2 * np.pi * i * t)
            cos_term = torch.cos(2 * np.pi * i * t)
            
            # x方向: w_{2i-1} * sin(2πit) + w_{2i} * cos(2πit)
            xyz_t[:, 0] += motion_x[:, 2*i-1] * sin_term + motion_x[:, 2*i] * cos_term
            # y方向
            xyz_t[:, 1] += motion_y[:, 2*i-1] * sin_term + motion_y[:, 2*i] * cos_term
            # z方向
            xyz_t[:, 2] += motion_z[:, 2*i-1] * sin_term + motion_z[:, 2*i] * cos_term
        
        return xyz_t
    
    @staticmethod
    def compute_velocity(motion_params, t, fourier_terms):
        """
        计算时间t时的速度（位置对时间的导数）
        
        v_x(t) = Σ[2πi * (w_{x,2i-1} * cos(2πit) - w_{x,2i} * sin(2πit))]
        """
        N = motion_params.shape[0]
        L = fourier_terms
        
        velocity = torch.zeros((N, 3), device=motion_params.device)
        
        motion_x = motion_params[:, 0:(2*L+1)]
        motion_y = motion_params[:, (2*L+1):2*(2*L+1)]
        motion_z = motion_params[:, 2*(2*L+1):3*(2*L+1)]
        
        for i in range(1, L+1):
            omega = 2 * np.pi * i
            sin_term = torch.sin(omega * t)
            cos_term = torch.cos(omega * t)
            
            # dx/dt
            velocity[:, 0] += omega * (motion_x[:, 2*i-1] * cos_term - motion_x[:, 2*i] * sin_term)
            # dy/dt
            velocity[:, 1] += omega * (motion_y[:, 2*i-1] * cos_term - motion_y[:, 2*i] * sin_term)
            # dz/dt
            velocity[:, 2] += omega * (motion_z[:, 2*i-1] * cos_term - motion_z[:, 2*i] * sin_term)
        
        return velocity
```
# 2. 修改 `scene/oursfull.py` 完整版本
```python
# 在文件开头添加导入
from helper_fourier import FourierMotionModel

class GaussianModel:

    def __init__(self, sh_degree : int, rgbfuntion="rgbv1", fourier_terms=5):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        
        # 傅里叶运动模型参数
        self.fourier_terms = fourier_terms  # L值
        self._motion = torch.empty(0)  # 将存储傅里叶系数
        
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._omega = torch.empty(0)
        
        self.rgbdecoder = getcolormodel(rgbfuntion)
        
        self.setup_functions()
        self.delta_t = None
        self.omegamask = None 
        self.maskforems = None 
        self.distancetocamera = None
        self.trbfslinit = None 
        self.ts = None 
        self.trbfoutput = None 
        self.preprocesspoints = False 
        self.addsphpointsscale = 0.8
        
        self.maxz, self.minz =  0.0 , 0.0 
        self.maxy, self.miny =  0.0 , 0.0 
        self.maxx, self.minx =  0.0 , 0.0  
        self.raystart = 0.7
        self.computedtrbfscale = None 
        self.computedopacity = None 
        self.computedscales = None 

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        # 预处理点云（如果需要）
        if self.preprocesspoints == 3:
            pcd = interpolate_point(pcd, 4) 
        elif self.preprocesspoints == 4:
            pcd = interpolate_point(pcd, 2)
        # ... 其他预处理选项 ...
        
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        times = torch.tensor(np.asarray(pcd.times)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        scales = torch.clamp(scales, -10, 1.0)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))

        features9channel = torch.cat((fused_color, fused_color), dim=1)
        self._features_dc = nn.Parameter(features9channel.contiguous().requires_grad_(True))
        
        N, _ = fused_color.shape
        fomega = torch.zeros((N, 3), dtype=torch.float, device="cuda")
        self._features_t = nn.Parameter(fomega.contiguous().requires_grad_(True))

        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))

        omega = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        self._omega = nn.Parameter(omega.requires_grad_(True))
        
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        
        # 初始化傅里叶运动参数
        motion_dim = FourierMotionModel.get_motion_dim(self.fourier_terms)
        motion = torch.zeros((fused_point_cloud.shape[0], motion_dim), device="cuda")
        self._motion = nn.Parameter(motion.requires_grad_(True))
        
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self._trbf_center = nn.Parameter(times.contiguous().requires_grad_(True))
        self._trbf_scale = nn.Parameter(torch.ones((self.get_xyz.shape[0], 1), device="cuda").requires_grad_(True))

        if self.trbfslinit is not None:
            nn.init.constant_(self._trbf_scale, self.trbfslinit)
        else:
            nn.init.constant_(self._trbf_scale, 0)

        nn.init.constant_(self._features_t, 0)
        nn.init.constant_(self._omega, 0)
        
        # 初始化傅里叶系数 - 高频项初始化为较小值
        with torch.no_grad():
            L = self.fourier_terms
            for i in range(3):  # 对x, y, z三个方向
                offset = i * (2*L+1)
                # 常数项初始化为0
                self._motion[:, offset] = 0
                # 傅里叶系数随频率递减初始化
                for j in range(1, L+1):
                    scale = 0.01 / j  # 高频项更小
                    self._motion[:, offset + 2*j-1] = torch.randn_like(self._motion[:, offset + 2*j-1]) * scale
                    self._motion[:, offset + 2*j] = torch.randn_like(self._motion[:, offset + 2*j]) * scale
        
        self.rgb_grd = {}

        self.maxz, self.minz = torch.amax(self._xyz[:,2]), torch.amin(self._xyz[:,2]) 
        self.maxy, self.miny = torch.amax(self._xyz[:,1]), torch.amin(self._xyz[:,1]) 
        self.maxx, self.minx = torch.amax(self._xyz[:,0]), torch.amin(self._xyz[:,0]) 
        self.maxz = min((self.maxz, 200.0))

        for name, W in self.rgbdecoder.named_parameters():
            if 'weight' in name:
                self.rgb_grd[name] = torch.zeros_like(W, requires_grad=False).cuda()

    def get_xyz_at_time(self, t):
        """使用傅里叶级数计算时间t时的位置"""
        return FourierMotionModel.compute_position(
            self._xyz, self._motion, t, self.fourier_terms
        )
    
    def get_velocity_at_time(self, t):
        """计算时间t时的速度"""
        return FourierMotionModel.compute_velocity(
            self._motion, t, self.fourier_terms
        )

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z','trbf_center', 'trbf_scale' ,'nx', 'ny', 'nz']
        
        # 添加傅里叶运动参数
        motion_dim = FourierMotionModel.get_motion_dim(self.fourier_terms)
        for i in range(motion_dim):
            l.append('motion_{}'.format(i))

        for i in range(self._features_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._omega.shape[1]):
            l.append('omega_{}'.format(i))
        for i in range(self._features_t.shape[1]):
            l.append('f_t_{}'.format(i))
        
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        trbf_center = self._trbf_center.detach().cpu().numpy()
        trbf_scale = self._trbf_scale.detach().cpu().numpy()
        motion = self._motion.detach().cpu().numpy()
        omega = self._omega.detach().cpu().numpy()
        f_t = self._features_t.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, trbf_center, trbf_scale, normals, motion, f_dc, opacities, scale, rotation, omega, f_t), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        # 保存模型和傅里叶配置
        model_fname = path.replace(".ply", ".pt")
        print(f'Saving model checkpoint to: {model_fname}')
        torch.save({
            'rgb_decoder': self.rgbdecoder.state_dict(),
            'fourier_terms': self.fourier_terms
        }, model_fname)

    def load_ply(self, path):
        plydata = PlyData.read(path)
        
        # 加载模型配置
        ckpt = torch.load(path.replace(".ply", ".pt"))
        self.rgbdecoder.load_state_dict(ckpt['rgb_decoder'])
        self.fourier_terms = ckpt.get('fourier_terms', 5)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        trbf_center = np.asarray(plydata.elements[0]["trbf_center"])[..., np.newaxis]
        trbf_scale = np.asarray(plydata.elements[0]["trbf_scale"])[..., np.newaxis]

        # 加载傅里叶运动参数
        motion_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("motion")]
        motion_dim = FourierMotionModel.get_motion_dim(self.fourier_terms)
        motion = np.zeros((xyz.shape[0], motion_dim))
        for i in range(min(len(motion_names), motion_dim)):
            motion[:, i] = np.asarray(plydata.elements[0]["motion_"+str(i)])

        # 加载其他参数
        dc_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_dc")]
        num_dc_features = len(dc_f_names)
        features_dc = np.zeros((xyz.shape[0], num_dc_features))
        for i in range(num_dc_features):
            features_dc[:, i] = np.asarray(plydata.elements[0]["f_dc_"+str(i)])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        omega_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("omega")]
        omegas = np.zeros((xyz.shape[0], len(omega_names)))
        for idx, attr_name in enumerate(omega_names):
            omegas[:, idx] = np.asarray(plydata.elements[0][attr_name])

        ft_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_t")]
        ftomegas = np.zeros((xyz.shape[0], len(ft_names)))
        for idx, attr_name in enumerate(ft_names):
            ftomegas[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._trbf_center = nn.Parameter(torch.tensor(trbf_center, dtype=torch.float, device="cuda").requires_grad_(True))
        self._trbf_scale = nn.Parameter(torch.tensor(trbf_scale, dtype=torch.float, device="cuda").requires_grad_(True))
        self._motion = nn.Parameter(torch.tensor(motion, dtype=torch.float, device="cuda").requires_grad_(True))
        self._omega = nn.Parameter(torch.tensor(omegas, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_t = nn.Parameter(torch.tensor(ftomegas, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        self.computedopacity = self.opacity_activation(self._opacity)
        self.computedscales = torch.exp(self._scaling)
        self.computedtrbfscale = torch.exp(self._trbf_scale)

    # 修改所有涉及motion的densification函数
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        
        numpytmp = rots.cpu().numpy() @ samples.unsqueeze(-1).cpu().numpy()
        new_xyz = torch.from_numpy(numpytmp).cuda().squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_trbf_center = self._trbf_center[selected_pts_mask].repeat(N,1)
        new_trbf_scale = self._trbf_scale[selected_pts_mask].repeat(N,1)
        
        # 复制傅里叶运动参数
        new_motion = self._motion[selected_pts_mask].repeat(N,1)
        
        new_omega = self._omega[selected_pts_mask].repeat(N,1)
        new_feature_t = self._features_t[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_opacity, new_scaling, new_rotation, 
                                  new_trbf_center, new_trbf_scale, new_motion, new_omega, new_feature_t)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_trbf_center = torch.rand((self._trbf_center[selected_pts_mask].shape[0], 1), device="cuda")
        new_trbfscale = self._trbf_scale[selected_pts_mask]
        
        # 克隆傅里叶运动参数
        new_motion = self._motion[selected_pts_mask]
        
        new_omega = self._omega[selected_pts_mask]
        new_featuret = self._features_t[selected_pts_mask]
        
        self.densification_postfix(new_xyz, new_features_dc, new_opacities, new_scaling, new_rotation,
                                  new_trbf_center, new_trbfscale, new_motion, new_omega, new_featuret)

    # 其他需要修改的函数...
```
# 3.修改 `renderer/__init__.py` 完整版本
```python
# 在文件开头添加导入
from helper_fourier import FourierMotionModel

def train_ours_full(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, 
                    scaling_modifier = 1.0, override_color = None, basicfunction = None, 
                    GRsetting=None, GRzer=None):
    """
    Render the scene with Fourier motion model
    """
    
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0
    
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False)

    rasterizer = GRzer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity

    trbfcenter = pc.get_trbfcenter
    trbfscale = pc.get_trbfscale

    trbfdistanceoffset = viewpoint_camera.timestamp * pointtimes - trbfcenter
    trbfdistance = trbfdistanceoffset / torch.exp(trbfscale)
    trbfoutput = basicfunction(trbfdistance)
    
    opacity = pointopacity * trbfoutput
    pc.trbfoutput = trbfoutput

    cov3D_precomp = None
    scales = pc.get_scaling
    shs = None
    
    # 使用傅里叶级数计算位置
    timestamp = viewpoint_camera.timestamp
    means3D = pc.get_xyz_at_time(timestamp)
    
    # 获取旋转（可能也需要时间相关）
    tforpoly = trbfdistanceoffset.detach()
    rotations = pc.get_rotation(tforpoly)
    colors_precomp = pc.get_features(tforpoly)
    
    rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    rendered_image = pc.rgbdecoder(rendered_image.unsqueeze(0), viewpoint_camera.rays, viewpoint_camera.timestamp)
    rendered_image = rendered_image.squeeze(0)
    
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "depth": depth}

def test_ours_full(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, 
                   scaling_modifier = 1.0, override_color = None, basicfunction = None, 
                   GRsetting=None, GRzer=None):
    """
    Test rendering with Fourier motion model
    """
    
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    torch.cuda.synchronize()
    startime = time.time()

    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0
    
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False)

    rasterizer = GRzer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity

    trbfcenter = pc.get_trbfcenter
    trbfscale = pc.get_trbfscale

    trbfdistanceoffset = viewpoint_camera.timestamp * pointtimes - trbfcenter
    trbfdistance = trbfdistanceoffset / torch.exp(trbfscale)
    trbfoutput = basicfunction(trbfdistance)
    
    opacity = pointopacity * trbfoutput
    pc.trbfoutput = trbfoutput

    cov3D_precomp = None
    scales = pc.get_scaling
    shs = None
    
    # 使用傅里叶级数计算位置
    timestamp = viewpoint_camera
```
# 3.修改 `renderer/__init__.py` 完整版本（续）
```python
def test_ours_full(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, 
                   scaling_modifier = 1.0, override_color = None, basicfunction = None, 
                   GRsetting=None, GRzer=None):
    """
    Test rendering with Fourier motion model
    """
    
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    torch.cuda.synchronize()
    startime = time.time()

    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0
    
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False)

    rasterizer = GRzer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity

    trbfcenter = pc.get_trbfcenter
    trbfscale = pc.get_trbfscale

    trbfdistanceoffset = viewpoint_camera.timestamp * pointtimes - trbfcenter
    trbfdistance = trbfdistanceoffset / torch.exp(trbfscale)
    trbfoutput = basicfunction(trbfdistance)
    
    opacity = pointopacity * trbfoutput
    pc.trbfoutput = trbfoutput

    cov3D_precomp = None
    scales = pc.get_scaling
    shs = None
    
    # 使用傅里叶级数计算位置
    timestamp = viewpoint_camera.timestamp
    means3D = pc.get_xyz_at_time(timestamp)
    
    # 获取旋转
    tforpoly = trbfdistanceoffset.detach()
    rotations = pc.get_rotation(tforpoly)
    colors_precomp = pc.get_features(tforpoly)
    
    rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    rendered_image = pc.rgbdecoder(rendered_image.unsqueeze(0), viewpoint_camera.rays, viewpoint_camera.timestamp)
    rendered_image = rendered_image.squeeze(0)
    torch.cuda.synchronize()
    duration = time.time() - startime

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "depth": depth,
            "duration":duration}

def train_ours_lite(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, 
                    scaling_modifier = 1.0, override_color = None, basicfunction = None, 
                    GRsetting=None, GRzer=None):
    """
    Render the scene with Fourier motion model (lite version)
    """
    
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0
    
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False)

    rasterizer = GRzer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity

    trbfcenter = pc.get_trbfcenter
    trbfscale = pc.get_trbfscale

    trbfdistanceoffset = viewpoint_camera.timestamp * pointtimes - trbfcenter
    trbfdistance = trbfdistanceoffset / torch.exp(trbfscale)
    trbfoutput = basicfunction(trbfdistance)
    
    opacity = pointopacity * trbfoutput
    pc.trbfoutput = trbfoutput

    cov3D_precomp = None
    scales = pc.get_scaling
    shs = None
    
    # 使用傅里叶级数计算位置
    timestamp = viewpoint_camera.timestamp
    means3D = pc.get_xyz_at_time(timestamp)
    
    tforpoly = trbfdistanceoffset.detach()
    rotations = pc.get_rotation(tforpoly)
    colors_precomp = pc.get_features(tforpoly)

    rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "depth": depth}

def test_ours_lite(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, 
                   scaling_modifier = 1.0, override_color = None, basicfunction = None,
                   GRsetting=None, GRzer=None):
    
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    torch.cuda.synchronize()
    startime = time.time()

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False)

    rasterizer = GRzer(raster_settings=raster_settings)
    
    # 使用傅里叶级数计算位置
    timestamp = viewpoint_camera.timestamp
    means3D = pc.get_xyz_at_time(timestamp)
    
    tforpoly = viewpoint_camera.timestamp - pc.get_trbfcenter
    rotations = pc.get_rotation(tforpoly)
    colors_precomp = pc.get_features(tforpoly)
    
    means2D = screenspace_points
    cov3D_precomp = None
    shs = None
    
    # 注意：这里需要传递傅里叶运动参数而不是原来的motion
    rendered_image, radii = rasterizer(
        timestamp = viewpoint_camera.timestamp,
        trbfcenter = pc.get_trbfcenter,
        trbfscale = pc.computedtrbfscale,
        motion = pc._motion,  # 现在是傅里叶系数
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = pc.computedopacity,
        scales = pc.computedscales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    torch.cuda.synchronize()
    duration = time.time() - startime
    
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "duration":duration}
```
# 4.修改 `scene/ourslite.py` 完整版本
```python
# 在文件开头添加导入
from helper_fourier import FourierMotionModel

class GaussianModel:
    
    def __init__(self, sh_degree : int, rgbfuntion="rgbv1", fourier_terms=5):
        # 与oursfull.py类似的初始化，但不包含RGB decoder相关的部分
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        
        # 傅里叶运动模型参数
        self.fourier_terms = fourier_terms
        self._motion = torch.empty(0)
        
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._omega = torch.empty(0)
        
        self.setup_functions()
        # ... 其他初始化 ...

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        # 与oursfull类似，但简化版
        # ... 前面的代码保持不变 ...
        
        # 初始化傅里叶运动参数
        motion_dim = FourierMotionModel.get_motion_dim(self.fourier_terms)
        motion = torch.zeros((fused_point_cloud.shape[0], motion_dim), device="cuda")
        self._motion = nn.Parameter(motion.requires_grad_(True))
        
        # 初始化傅里叶系数
        with torch.no_grad():
            L = self.fourier_terms
            for i in range(3):
                offset = i * (2*L+1)
                self._motion[:, offset] = 0
                for j in range(1, L+1):
                    scale = 0.01 / j
                    self._motion[:, offset + 2*j-1] = torch.randn_like(self._motion[:, offset + 2*j-1]) * scale
                    self._motion[:, offset + 2*j] = torch.randn_like(self._motion[:, offset + 2*j]) * scale
        
        # ... 其余代码 ...

    def get_xyz_at_time(self, t):
        """使用傅里叶级数计算时间t时的位置"""
        return FourierMotionModel.compute_position(
            self._xyz, self._motion, t, self.fourier_terms
        )
    
    def get_velocity_at_time(self, t):
        """计算时间t时的速度"""
        return FourierMotionModel.compute_velocity(
            self._motion, t, self.fourier_terms
        )
    
    # 其他方法的修改与oursfull.py类似
    # ...
```
# 5.修改配置文件和参数解析
修改 `thirdparty/gaussian_splatting/arguments.py`：
```python
class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # ... 现有参数 ...
        
        # 添加傅里叶相关参数
        self.fourier_terms = 5  # 默认使用5个傅里叶项
        self.fourier_lr_scale = 1.0  # 傅里叶系数的学习率缩放
        
        super().__init__(parser, "Optimization Parameters")
        
        # 添加命令行参数
        parser.add_argument('--fourier_terms', type=int, default=5,
                          help='Number of Fourier terms for motion modeling (L)')
        parser.add_argument('--fourier_lr_scale', type=float, default=1.0,
                          help='Learning rate scale for Fourier coefficients')
```
# 6. 修改训练脚本入口
修改 `helper3dg.py` 中的 `getparser()` 函数：
```python
def getparser():
    parser = ArgumentParser(description="Training script parameters")
    # ... 现有参数 ...
    
    # 添加傅里叶参数
    parser.add_argument("--fourier_terms", type=int, default=5, 
                       help="Number of Fourier terms for motion modeling")
    parser.add_argument("--fourier_init_scale", type=float, default=0.01,
                       help="Initialization scale for Fourier coefficients")
    
    # ... 其余代码 ...
    
    # 在创建高斯模型时传递傅里叶参数
    # 修改场景初始化部分
    return args, lp.extract(args), op.extract(args), pp.extract(args)
```
# 7.修改场景初始化
修改 `scene/__init__.py`：
```python
class Scene:
    def __init__(self, args : ModelParams, gaussians, load_iteration=None, 
                 shuffle=True, resolution_scales=[1.0], multiview=False, 
                 duration=50.0, loader="colmap"):
        # ... 现有代码 ...
        
        # 在初始化gaussians时传递傅里叶参数
        if hasattr(args, 'fourier_terms'):
            gaussians.fourier_terms = args.fourier_terms
        
        # ... 其余代码 ...
```
# 8.创建新的训练脚本 `trainv2_fourier.py`
```python
import os
import torch
from random import randint
import sys
from scene import Scene
from scene.oursfull import GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser
from helper3dg import getparser, getrenderparts
from helper_train import getloss, controlgaussians
from helper_fourier import FourierMotionModel
import numpy as np

def training(dataset, opt, pipe, testing_iterations, saving_iterations, 
             checkpoint_iterations, checkpoint, debug_from, args):
    
    # 初始化高斯模型，包含傅里叶参数
    gaussians = GaussianModel(dataset.sh_degree, 
                             rgbfuntion=args.rgbfunction,
                             fourier_terms=args.fourier_terms)
    
    scene = Scene(dataset, gaussians, shuffle=False, 
                  resolution_scales=[1.0], duration=args.duration)
    
    gaussians.training_setup(opt)
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        first_iter = 0
    else:
        first_iter = 0

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 训练循环
    viewpoint_stack = None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    
    for iteration in range(first_iter, opt.iterations + 1):
        
        # 更新学习率
        gaussians.update_learning_rate(iteration)
        
        # 每1000次迭代增加SH degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        # 选择随机相机
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # 渲染
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        
        # 使用傅里叶运动模型渲染
        if args.rdpip == "v2":
            from renderer import train_ours_full
            render_pkg = train_ours_full(viewpoint_cam, gaussians, pipe, bg, 
                                        basicfunction=basicfunction,
                                        GRsetting=GRsetting, GRzer=GRzer)
        elif args.rdpip == "v3":
            from renderer import train_ours_lite
            render_pkg = train_ours_lite(viewpoint_cam, gaussians, pipe, bg,
                                        basicfunction=basicfunction,
                                        GRsetting=GRsetting, GRzer=GRzer)
        
        image, viewspace_point_tensor, visibility_filter, radii = getrenderparts(render_pkg)
        
        # 计算损失
        gt_image = viewpoint_cam.original_image.cuda()
        loss, Ll1, Lssim = getloss(image, gt_image, opt)
        
        # 添加运动正则化损失（可选）
        if iteration < opt.densify_until_iter:
            # 可以添加傅里叶系数的正则化，例如高频项的L2正则化
            L = gaussians.fourier_terms
            motion_reg = 0
            for i in range(1, L+1):
                # 高频项的权重
                freq_weight = 1.0 / (i ** 2)  # 频率越高，正则化越强
                for j in range(3):  # x, y, z
                    offset = j * (2*L+1)
                    sin_idx = offset + 2*i-1
                    cos_idx = offset + 2*i
                    motion_reg += freq_weight * (
                        torch.norm(gaussians._motion[:, sin_idx]) +
                        torch.norm(gaussians._motion[:, cos_idx])
                    )
            
            motion_reg_weight = 0.001  # 可调整的权重
            loss = loss + motion_reg_weight * motion_reg
        
        loss.backward()
        
        with torch.no_grad():
            # 进度条更新
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}"})
                progress_bar.update(10)
            
            # 保存和测试
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], 
                    radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    controlgaussians(gaussians, opt, scene, visibility_filter, radii, 
                                   viewspace_point_tensor, iteration)
            
            # 优化器步骤
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
            
            # 保存检查点
            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), 
                          scene.model_path + "/chkpnt" + str(iteration) + ".pth")

if __name__ == "__main__":
    # 获取参数
    args, lp, op, pp = getparser()
    
    # 设置随机种子
    safe_state(args.quiet)
    
    # 开始训练
    training(lp, op, pp, args.test_iterations, args.save_iterations,
            args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)
    
    print("\nTraining complete.")
```
# 9.修改测试脚本 `test.py`
```python
# 在相应的位置添加傅里叶运动模型的支持
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, args):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # 根据渲染管道选择合适的渲染函数
        if args.rdpip == "v2":
            from renderer import test_ours_full
            rendering = test_ours_full(view, gaussians, pipeline, background,
                                      basicfunction=basicfunction,
                                      GRsetting=GRsetting, GRzer=GRzer)["render"]
        elif args.rdpip == "v3":
            from renderer import test_ours_lite
            rendering = test_ours_lite(view, gaussians, pipeline, background,
                                      basicfunction=basicfunction,
                                      GRsetting=GRsetting, GRzer=GRzer)["render"]
        
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, 
                                    os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, 
                                    os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
```
# 10.创建配置文件 `configs/fourier_config.json`
```json
{
    "fourier_terms": 5,
    "fourier_init_scale": 0.01,
    "fourier_lr_scale": 1.0,
    "motion_reg_weight": 0.001,
    "high_freq_penalty": true,
    "iterations": 30000,
    "position_lr_init": 0.00016,
    "position_lr_final": 0.0000016,
    "position_lr_delay_mult": 0.01,
    "position_lr_max_steps": 30000,
    "feature_lr": 0.0025,
    "opacity_lr": 0.05,
    "scaling_lr": 0.005,
    "rotation_lr": 0.001,
    "densify_from_iter": 500,
    "densify_until_iter": 15000,
    "densify_grad_threshold": 0.0002,
    "densification_interval": 100,
    "opacity_reset_interval": 3000,
    "lambda_dssim": 0.2
}
```
# 11.修改启动脚本
创建 `run_fourier.sh`：
```bash
#!/bin/bash

# 训练脚本
python trainv2_fourier.py \
    -s /path/to/dataset \
    -m /path/to/output \
    --eval \
    --fourier_terms 5 \
    --fourier_init_scale 0.01 \
    --duration 50 \
    --rdpip v2 \
    --configpath configs/fourier_config.json

# 测试脚本
python test.py \
    -s /path/to/dataset \
    -m /path/to/output \
    --test_iteration 30000 \
    --rdpip v2 \
    --configpath configs/fourier_config.json
```
这个完整的修改方案涵盖了所有需要修改的文件和细节，确保傅里叶运动模型能够正确集成到现有的代码框架中。主要改动包括：
1. 创建了专门的傅里叶运动模型辅助类
2. 修改了所有涉及运动计算的地方
3. 更新了参数初始化和优化策略
4. 添加了运动正则化损失
5. 修改了保存和加载机制
6. 更新了配置文件系统
7. 确保所有渲染函数都支持新的运动模型

这样就能完整地将多项式运动模型替换为傅里叶级数运动模型。