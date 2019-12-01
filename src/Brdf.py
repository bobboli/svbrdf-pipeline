import exifread
import numpy as np
import cv2
from Clusterer import Clusterer
import os
from time import time
import datetime
from scipy import optimize
from sklearn.cluster import MeanShift
from utilities import ReadImg, ReadImgAsUint8, WriteImg, WriteImgAsUint8, WritePfm, PackNormalTangentMap, \
    UnpackNormalTangentMap, plt, OriField, Rgb2Lum, Rgb2Pca, \
    _delta
from hdri import FitResponseCurve, Img2Radiance, Radiance2Img, FetchWeights, zmax
import json
from matplotlib import pyplot as plt


# Disney BRDF fitter via L-BFGS-B optimizer
class Brdf:
    def __init__(self):
        pass

    # Use a point light illuminated greycard to calibrate point light's position and intensity
    def Calibrate(self, rootpath, colorcard_list, greycard, greycard_dist, align_colorcard=False):
        ####################################################################################################
        # Fit camera response curve
        ####################################################################################################
        samples_list = []
        exposure_time_list = []
        for i in range(len(colorcard_list)):
            sample_path = os.path.join(rootpath, colorcard_list[i])
            samples_list.append(ReadImgAsUint8(sample_path))
            with open(sample_path, 'rb') as img:
                exif_info_sample = exifread.process_file(img)
            exposure_time_list.append(exif_info_sample['EXIF ExposureTime'].values[0].num / \
                                      exif_info_sample['EXIF ExposureTime'].values[0].den
                                      )

        #self.response_curve = FitResponseCurve(samples_list, exposure_time_list, align_colorcard)
        self.response_curve = np.zeros((256, 3), dtype=np.float)
        xx = np.arange(256)
        xx[0] = 1
        self.response_curve[:, 0] = np.log((xx / 255) ** 2.2)
        self.response_curve[:, 1] = np.log((xx / 255) ** 2.2)
        self.response_curve[:, 2] = np.log((xx / 255) ** 2.2)
        del xx

        del samples_list
        del exposure_time_list

        ####################################################################################################
        # Determine point light's intensity using greycard photo
        ####################################################################################################
        img_greycard_path = os.path.join(rootpath, greycard)
        img_greycard = self.ReadImagesAsRadiance(img_greycard_path)
        # WritePfm('grey.pfm', img_greycard/10)
        img_greycard_flattened = img_greycard.reshape(-1, 3)

        self.size_y, self.size_x = img_greycard.shape[0:2]

        with open(img_greycard_path, 'rb') as img:
            exifinfo_greycard = exifread.process_file(img)
        # 35 mm equivalent focal length
        self.efl_35mm = exifinfo_greycard['EXIF FocalLengthIn35mmFilm'].values[0]

        # print(self.efl_35mm)

        # https://en.wikipedia.org/wiki/Angle_of_view
        # https://en.wikipedia.org/wiki/35_mm_equivalent_focal_length, diagonal-based
        # diagonal aov = 2 arctan(d/2f).  d = **sensor** diagonal length. For 35mm film: 36mm * 24mm, diagonal = 43.3mm
        # In our coordinate system camera is set to be at (0, 0, 1),so half diagonal length is d/2f
        half_diag_size = np.sqrt(36 ** 2 + 24 ** 2) / (2 * self.efl_35mm)
        # Store the difference in world space that corresponds to a pixel's offset.
        self.delta_xy = 2 * half_diag_size / np.sqrt(self.size_x ** 2 + self.size_y ** 2)

        # Level adjust
        # shadow, midtone, highlight = level_adjust
        # shadow /= 255
        # highlight /= 255
        # img_greycard_enhanced = np.clip((ReadImg(img_greycard_path, gammaDecoding=False) - shadow) / (highlight - shadow), 0, 1) ** (1 / midtone)

        img_greycard_grey = Rgb2Lum(img_greycard)
        thres = np.percentile(img_greycard_grey, 90)
        iy, ix = np.where(img_greycard_grey > thres)
        grey_flattened = img_greycard_grey[(iy, ix)]

        iy_light = np.round((iy * grey_flattened).sum() / grey_flattened.sum()).astype(np.int)
        ix_light = np.round((ix * grey_flattened).sum() / grey_flattened.sum()).astype(np.int)
        greycard_light_position = self.Idx2Pos(iy_light, ix_light) + (1.0,)

        # Show the point light position as a red cross
        # img_greycard_show = np.copy(img_greycard)
        # img_greycard_show[img_greycard_grey<=thres] = 0.0
        # plt.imshow(img_greycard_show)
        # plt.scatter(ix_light, iy_light, s=100, c='red', marker='+')
        # plt.show()

        iy, ix = np.meshgrid(np.arange(0, self.size_y), np.arange(0, self.size_x), indexing='ij')
        iy = iy.flatten()
        ix = ix.flatten()
        xs, ys = self.Idx2Pos(iy, ix)

        # l = sample / albedp * pi * r^2 / costheta
        inv_lambert_brdf = np.pi / 0.18
        light_dist = np.sqrt((greycard_light_position[0] - xs) * (greycard_light_position[0] - xs) +
                             (greycard_light_position[1] - ys) * (greycard_light_position[1] - ys) + 1.0)
        # Show light intensity solution
        # plt.imshow((img_greycard_flattened * inv_lambert_brdf * (light_dist ** 3)[:, np.newaxis]).reshape(self.size_y, self.size_x, 3))
        # plt.scatter(ix_light, iy_light, s=100, c='red', marker='+')
        # plt.show()
        self.light_intensity_greycard = tuple(
            (img_greycard_flattened * inv_lambert_brdf * (light_dist ** 3)[:, np.newaxis]).mean(axis=0))

        print('Point light intensity in greycard image: ', self.light_intensity_greycard)

        self.exposure_time_greycard = exifinfo_greycard['EXIF ExposureTime'].values[0].num / \
                                      exifinfo_greycard['EXIF ExposureTime'].values[0].den
        self.greycard_dist = greycard_dist

        self.calibrated = True

        return self

    def ManualCalibrate(self, light_position, light_intensity_greycard, greycard_dist):
        self.light_position = light_position
        self.light_intensity_greycard = light_intensity_greycard
        self.greycard_dist = greycard_dist

        self.calibrated = True
        return self

    def Img2Radiance(self, list_img, list_exposure_time):
        return Img2Radiance(list_img, list_exposure_time, self.response_curve)

    def Radiance2Img(self, radiance, exposure_time):
        return Radiance2Img(radiance, exposure_time, self.response_curve)

    def ReadImagesAsRadiance(self, image_path_list, exposure_time_list=None):
        if type(image_path_list) is not list:
            image_path_list = [image_path_list]
        images_list = []
        if exposure_time_list is None:
            exposure_time_list = []
            for i in range(len(image_path_list)):
                img_path = image_path_list[i]
                images_list.append(ReadImgAsUint8(img_path))
                with open(img_path, 'rb') as img:
                    exif_info_img = exifread.process_file(img)
                exposure_time_list.append(exif_info_img['EXIF ExposureTime'].values[0].num / \
                                          exif_info_img['EXIF ExposureTime'].values[0].den
                                          )
        return self.Img2Radiance(images_list, exposure_time_list)

    def WriteRadianceAsImage(self, file_path, radiance, exposure_time):
        WriteImgAsUint8(file_path, self.Radiance2Img(radiance, exposure_time))

    def Fit(self, img_input_config, n_clusters=500, n_samples=20000):
        assert self.calibrated, 'Point light not calibrated yet. Call Calibrate or ManualCalibrate.'

        try:
            with open(img_input_config, 'r') as f:
                input_config = json.load(f)
        except Exception:
            raise Exception('Invalid input config file:',img_input_config)

        dirPath = os.path.dirname(img_input_config)



        self.default_output_path = os.path.join(dirPath, 'out')
        self.default_config_file = 'config.json'

        self.img_dir = os.path.realpath(dirPath)

        self.img_ambient = self.ReadImagesAsRadiance(os.path.join(self.img_dir, input_config['ambient']))
        with open(os.path.join(self.img_dir, input_config['ambient']), 'rb') as img:
            exif_info = exifread.process_file(img)
        self.ambient_exposure_time = exif_info['EXIF ExposureTime'].values[0].num / \
                                     exif_info['EXIF ExposureTime'].values[
                                         0].den

        if type(input_config['point']) is list:
            self.img_point = self.ReadImagesAsRadiance(
                [os.path.join(self.img_dir, item) for item in input_config['point']])
            self.exposure_time = self.exposure_time_greycard
        else:
            self.img_point = self.ReadImagesAsRadiance(os.path.join(self.img_dir, input_config['point']))
            with open(os.path.join(self.img_dir, input_config['point']), 'rb') as img:
                exif_info = exifread.process_file(img)
            self.exposure_time = exif_info['EXIF ExposureTime'].values[0].num / exif_info['EXIF ExposureTime'].values[
                0].den
        # plt.figure(1)
        # plt.imshow(brdf.img_point)
        # plt.figure(2)
        # plt.imshow(brdf.img_ambient)
        # plt.show()

        # fig = plt.figure()
        # plt.axis('off')
        # img = plt.imshow(Rgb2Lum(self.img_point), cmap=plt.get_cmap('jet'))
        # plt.clim(0, 20)
        # cbaxes = fig.add_axes([0.90, 0.2, 0.03, 0.6])
        # cb = plt.colorbar(img, cax=cbaxes)
        # # plt.colorbar()
        # #plt.show()
        # plt.savefig('radiance.png')

        self.dist = input_config['distance']

        self.light_intensity = tuple(np.copy(self.light_intensity_greycard) * (self.greycard_dist / self.dist) ** 2)
        # self.light_position = tuple(np.copy(self.greycard_light_position[0:2]) * (self.greycard_dist / self.dist)) + (1.0, )
        img_point_grey = Rgb2Lum(self.img_point)
        thres = np.percentile(img_point_grey, 90)
        iy, ix = np.where(img_point_grey > thres)
        grey_flattened = img_point_grey[(iy, ix)]

        iy_light = np.round((iy * grey_flattened).sum() / grey_flattened.sum()).astype(np.int)
        ix_light = np.round((ix * grey_flattened).sum() / grey_flattened.sum()).astype(np.int)
        self.light_position = self.Idx2Pos(iy_light, ix_light) + (1.0,)

        # Show the point light position as a red cross
        # img_point_show = np.copy(self.img_point)
        # img_point_show[img_point_grey<=thres] = 0.0
        # plt.imshow(img_point_show)
        # plt.scatter(ix_light, iy_light, s=100, c='red', marker='+')
        # plt.show()

        print('Light position: ', self.light_position)

        # Clusterer
        self.clusterer = Clusterer(self.img_ambient * self.ambient_exposure_time, self.size_x, self.size_y, n_clusters)
        ## TODO: Load clsuter info
        if self.clusterer.LoadClusterInfo(self.default_output_path):
            print('Loaded from existing cluster info.')
        else:
            self.clusterer.Cluster(n_samples)
        self.clusterer.SaveClusterInfo(self.default_output_path)
        self.clusterer.SaveClusterMap(self.default_output_path)

        # Used when bump info is loaded from maps other than computed from bump params (n1, n2, t)
        self.normal_map = None
        self.tangent_map = None

        # Parameters initialization
        try:
            self.is_metal = input_config['is_metal']
        except KeyError:
            self.is_metal = False

        self.InitParams()

        return self

    @classmethod
    def FromMaps(cls, map_path, configFile):
        brdf = Brdf()
        brdf.LoadMap(map_path, configFile)

        return brdf

    # Initialize brdf and bump parameters
    def InitParams(self):
        # Disney's principled BRDF
        # Initial values:
        # basecolor: average color of each cluster in ambient image
        # metallic: 0
        # specular: 0.5
        # speculartint: 0
        # roughness: 0.5
        # anisotropic: 0
        # spatially varying BRDF parameters
        # [0:3] ---  basecolor, in linear space (gamma-decoded)
        # [3]   ---  specular
        # [4]   ---  specularTint
        # [5]   ---  anisotropic
        print('Initializing BRDF params from ambient image.')
        self.brdf_cluster = np.zeros((self.clusterer.n_clusters, 6))
        self.brdf_cluster[:, 0:3] = Brdf.saturate(self.clusterer.cluster_centers_numerical[:, 0:3])
        self.brdf_cluster[:, 4] = 0.5
        self.brdf_map = np.zeros((self.size_y, self.size_x, 6))
        self.brdf_map[..., 0:3] = Brdf.saturate(self.img_ambient * self.ambient_exposure_time)
        self.brdf_map[..., 4] = 0.5
        # global BRDF parameters
        # [0]   ---  metallic
        # [1]   ---  roughness
        self.brdf_global = np.zeros(2)
        self.brdf_global[1] = 0.5
        if self.is_metal:
            self.brdf_global[0] = 1.0

        # Two bump parameters to determine tangent, bitangent and normal
        # 0:depth
        # 1:anisoaxis: [0, 1]

        self.bump_map = np.zeros((self.size_y, self.size_x, 2), dtype=np.float)
        self.bump_map[..., 1] = 0.5  # OriField(self.img_ambient) / np.pi

        self.bump_cluster = self.Map2Cluster(self.bump_map)

        self.SaveMap()
        self.isProcessed = False

    # Some utilities used by rendering routines

    @staticmethod
    def ComputeNormal(height_map):
        size_y, size_x = height_map.shape[0:2]
        # To normalize Sobel operator, dividing by 8
        sobelx = cv2.Sobel(height_map, cv2.CV_64F, 1, 0, ksize=3) / 8
        sobely = cv2.Sobel(height_map, cv2.CV_64F, 0, 1, ksize=3) / 8
        z = np.ones((size_y, size_x))

        return Brdf.normalize(np.stack((-sobelx, -sobely, z), axis=-1))

    @staticmethod
    def ComputeBump(anisoaxis, normal):
        normal = np.copy(normal)
        phi_t = anisoaxis * (np.pi * 2)
        cos_phi_t = np.cos(phi_t)
        sin_phi_t = np.sin(phi_t)
        # theta_t lies in [0, pi]
        sin_theta_t = normal[..., 2]
        cos_theta_t = -(normal[..., 0] * cos_phi_t + normal[..., 1] * sin_phi_t)
        norm_factr = 1 / np.sqrt(sin_theta_t * sin_theta_t + cos_theta_t * cos_theta_t + _delta)
        sin_theta_t *= norm_factr
        cos_theta_t *= norm_factr

        tangent = np.stack((sin_theta_t * cos_phi_t, sin_theta_t * sin_phi_t, cos_theta_t), axis=-1)
        bitangent = Brdf.cross(normal, tangent)

        return tangent, bitangent, normal

    @staticmethod
    def sqr(u):
        return np.square(u)

    @staticmethod
    def normalize(a):
        return a / np.sqrt((a * a).sum(axis=-1))[..., np.newaxis]

    @staticmethod
    def cross(a, b):
        return np.stack((a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1], a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                         a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]), axis=-1)

    @staticmethod
    def saturate(a):
        return np.clip(a, 0, 1)

    @staticmethod
    def dot(a, b):
        return (a * b).sum(axis=-1)

    @staticmethod
    def SchlickFresnel(u):
        m = np.clip(1 - u, 0, 1)
        m2 = m * m
        return m * m2 * m2

    @staticmethod
    def lerp(a, b, t):
        return a * (1.0 - t) + b * t

    @staticmethod
    def PseudoHuberLoss(residual, delta=0.1):
        delta2 = delta * delta
        # Pseudo Huber loss with delta = 0.1
        # L = delta^2 * (sqrt(1+a^2/delta^2)-1)
        return delta2 * (np.sqrt(1 + Brdf.sqr(residual) / delta2) - 1)

    # Loss for fitting parameters on a cluster
    # param1 and param2: t or BRDF parameters depending on boolean fittingBrdf
    @staticmethod
    def ClusterRenderLossKernel(brdf_cluster_and_anisoaxis, brdf_global, normals, positions, sample_radiances, light_position,
                                light_intensity, lossScale=10.0):
        xs, ys = positions
        n_samples = sample_radiances.shape[0]
        brdf = brdf_cluster_and_anisoaxis[np.newaxis, :-1]  # For convenient operand broadcasting
        ansioaxis = brdf_cluster_and_anisoaxis[-1]


        light_intensity = np.copy(light_intensity).reshape(1, 3)

        T, B, N = Brdf.ComputeBump(ansioaxis, normals)


        sqr = Brdf.sqr
        lerp = Brdf.lerp
        SchlickFresnel = Brdf.SchlickFresnel
        PseudoHuberLoss = Brdf.PseudoHuberLoss
        normalize = Brdf.normalize
        dot = Brdf.dot

        L_unnormalized = np.stack((light_position[0] - xs, light_position[1] - ys,
                                   np.repeat(light_position[2], n_samples)), axis=-1)

        V = np.stack((- xs, - ys, np.ones(n_samples, dtype=np.float)), axis=-1)
        L = normalize(L_unnormalized)
        V = normalize(V)
        H = normalize(L + V)

        light_dist2 = sqr(L_unnormalized).sum(axis=-1)

        LdotH = dot(L, H)
        LdotN = dot(L, N)
        LdotX = dot(L, T)
        LdotY = dot(L, B)

        VdotN = dot(V, N)
        VdotX = dot(V, T)
        VdotY = dot(V, B)

        HdotX = dot(H, T)
        HdotY = dot(H, B)
        HdotN = dot(N, H)

        Fd90 = 0.5 + 2 * sqr(LdotH) * brdf_global[1]
        FL = SchlickFresnel(LdotN)
        FV = SchlickFresnel(VdotN)
        fDiff = (lerp(1.0, Fd90, FL) * lerp(1.0, Fd90, FV) * (1.0 - brdf_global[0]))[..., np.newaxis] \
                * brdf[:, 0:3] / np.pi

        colTint = brdf[:, 0:3] / (Rgb2Lum(brdf[:, 0:3])[:, np.newaxis] + _delta)
        colNonMetal = 0.08 * brdf[:, 3, np.newaxis] * (
                (1 - brdf[:, 4, np.newaxis]) + colTint * brdf[:, 4, np.newaxis])
        colSpec = colNonMetal * (1 - brdf_global[0]) + brdf[:, 0:3] * brdf_global[0]

        aspect = np.sqrt(1.0 - 0.9 * brdf[:, 5])
        roughness2 = sqr(brdf_global[1])
        alpha_x = np.clip(roughness2 / aspect, 0.001, None)
        alpha_y = np.clip(roughness2 * aspect, 0.001, None)

        Ds = 1.0 / (np.pi * alpha_x * alpha_y * sqr(sqr(HdotX / alpha_x) + sqr(HdotY / alpha_y) + sqr(HdotN)))
        FH = SchlickFresnel(LdotH)
        Fs = lerp(colSpec, 1.0, FH[..., np.newaxis])
        Gs = 1.0 / ((LdotN + np.sqrt(sqr(LdotX * alpha_x) + sqr(LdotY * alpha_y) + sqr(LdotN)))
                    * (VdotN + np.sqrt(sqr(VdotX * alpha_x) + sqr(VdotY * alpha_y) + sqr(VdotN))))

        fSpec = Ds[..., np.newaxis] * Fs * Gs[..., np.newaxis]

        f = fDiff + fSpec

        # When fitting BRDF parameters, minimize fitting residual
        loss = Rgb2Lum(PseudoHuberLoss(
            light_intensity * f * (LdotN / light_dist2).reshape(-1, 1) - sample_radiances)).mean()

        # When fitting main anisotropic axis, make the rendered and reference samples have the same average luminance, then
        # minimize residual
        # else:
        #     samples_grey = Rgb2Lum(sample_radiances)
        #     norm_factor = samples_grey.sum()
        #     samples_rendered_grey = np.clip(Rgb2Lum(light_intensity * f * (LdotN / light_dist2).reshape(-1, 1)), 0, None)
        #     # Align sum(mean)
        #     samples_rendered_grey *= samples_grey.mean() / (samples_rendered_grey.mean() + _delta)
        #     # Cross-entropy
        #     loss = (samples_grey * np.log(1/(samples_rendered_grey+_delta))).sum() / (norm_factor + _delta)

        return loss * lossScale

    # Render the image with parameters.
    # New light position and light intensity is optionally supplied. If they are None, use default values.
    # Render error can also be returned under default settings.
    @staticmethod
    def ImageRenderKernel(brdf_global, light_position, light_intensity, delta_xy, brdf_map, tangent_map, bitangent_map,
                          normal_map,
                          tilesize_y=500, tilesize_x=500):

        size_y, size_x = brdf_map.shape[0:2]

        # Turn potential tuple into numpy array
        light_position = np.copy(light_position)
        light_intensity = np.copy(light_intensity)
        # View position is default
        view_position = np.copy((0.0, 0.0, 1.0))

        cross = Brdf.cross
        sqr = Brdf.sqr
        lerp = Brdf.lerp
        dot = Brdf.dot
        SchlickFresnel = Brdf.SchlickFresnel
        normalize = Brdf.normalize

        # Splitting into tiles to save memory
        img_rendered = np.zeros((size_y, size_x, 3), dtype=np.float)
        for iy_beg in range(0, size_y, tilesize_y):
            for ix_beg in range(0, size_x, tilesize_x):
                iy_end = iy_beg + tilesize_y
                ix_end = ix_beg + tilesize_x
                iy_end = min(iy_end, size_y)
                ix_end = min(ix_end, size_x)

                cur_tilesize_y = iy_end - iy_beg
                cur_tilesize_x = ix_end - ix_beg

                iy, ix = np.meshgrid(np.arange(iy_beg, iy_end), np.arange(ix_beg, ix_end), indexing='ij')
                iy = iy.flatten()
                ix = ix.flatten()

                xs = (ix + 0.5 - size_x / 2) * delta_xy
                ys = (iy + 0.5 - size_y / 2) * -delta_xy

                brdf_flattened = brdf_map[iy_beg:iy_end, ix_beg:ix_end].reshape((-1, brdf_map.shape[2]))

                T = tangent_map[iy_beg:iy_end, ix_beg:ix_end, :].reshape(-1, 3)
                B = bitangent_map[iy_beg:iy_end, ix_beg:ix_end, :].reshape(-1, 3)
                N = normal_map[iy_beg:iy_end, ix_beg:ix_end, :].reshape(-1, 3)

                L_unnormalized = np.stack((light_position[0] - xs, light_position[1] - ys,
                                           np.repeat(light_position[2], cur_tilesize_x * cur_tilesize_y)), axis=-1)

                V = np.stack((view_position[0] - xs, view_position[1] - ys,
                              np.repeat(view_position[2], cur_tilesize_x * cur_tilesize_y)), axis=-1)
                L = normalize(L_unnormalized)
                V = normalize(V)
                H = L + V
                H = normalize(H)

                LdotH = dot(L, H)
                LdotN = dot(L, N)
                LdotX = dot(L, T)
                LdotY = dot(L, B)

                VdotN = dot(V, N)
                VdotX = dot(V, T)
                VdotY = dot(V, B)

                HdotX = dot(H, T)
                HdotY = dot(H, B)
                HdotN = dot(N, H)

                Fd90 = 0.5 + 2 * sqr(LdotH) * brdf_global[1]
                FL = SchlickFresnel(LdotN)
                FV = SchlickFresnel(VdotN)
                fDiff = (lerp(1.0, Fd90, FL) * lerp(1.0, Fd90, FV) * (1.0 - brdf_global[0]))[..., np.newaxis] \
                        * brdf_flattened[:, 0:3] / np.pi

                colTint = brdf_flattened[:, 0:3] / (Rgb2Lum(brdf_flattened[:, 0:3])[:, np.newaxis] + _delta)
                colNonMetal = 0.08 * brdf_flattened[:, 3, np.newaxis] * (
                        (1.0 - brdf_flattened[:, 4, np.newaxis]) + colTint * brdf_flattened[:, 4, np.newaxis])
                colSpec = colNonMetal * (1 - brdf_global[0]) + brdf_flattened[:, 0:3] * brdf_global[0]

                aspect = np.sqrt(1.0 - 0.9 * brdf_flattened[:, 5])
                roughness2 = sqr(brdf_global[1])
                alpha_x = np.clip(roughness2 / aspect, 0.001, None)
                alpha_y = np.clip(roughness2 * aspect, 0.001, None)

                Ds = 1.0 / (np.pi * alpha_x * alpha_y * sqr(sqr(HdotX / alpha_x) + sqr(HdotY / alpha_y) + sqr(HdotN)))
                FH = SchlickFresnel(LdotH)
                Fs = lerp(colSpec, 1.0, FH[..., np.newaxis])
                Gs = 1.0 / ((LdotN + np.sqrt(sqr(LdotX * alpha_x) + sqr(LdotY * alpha_y) + sqr(LdotN)))
                            * (VdotN + np.sqrt(sqr(VdotX * alpha_x) + sqr(VdotY * alpha_y) + sqr(VdotN))))

                fSpec = Ds[..., np.newaxis] * Fs * Gs[..., np.newaxis]

                f = fDiff + fSpec

                light_dist2 = sqr(L_unnormalized).sum(axis=-1)

                img_rendered[iy_beg:iy_end, ix_beg:ix_end] = (
                        light_intensity.reshape(1, 3) * f * (LdotN / light_dist2)[..., np.newaxis]).reshape(
                    cur_tilesize_y, cur_tilesize_x, 3)

        return np.clip(img_rendered, 0, None)

    @staticmethod
    def RoughnessFitLossKernel(roughness, metallic, light_position, light_intensity, delta_xy, brdf_map, bump_map, img_point,
                               tilesize_y=500, tilesize_x=500, lossScale=10.0):
        brdf_global = np.copy([metallic, roughness])
        normal_map = Brdf.ComputeNormal(bump_map[..., 0])
        tangent_map, bitangent_map, normal_map = Brdf.ComputeBump(bump_map[..., 1], normal_map)

        img_rendered_grey = Rgb2Lum( Brdf.ImageRenderKernel(brdf_global, light_position, light_intensity, delta_xy, brdf_map,
                                              tangent_map, bitangent_map, normal_map,
                                              tilesize_y, tilesize_x))

        img_grey = Rgb2Lum(img_point)
        norm_factor = img_grey.sum()
        # Align sum(mean)
        img_rendered_grey *= img_grey.mean() / (img_rendered_grey.mean() + _delta)
        # Cross-entropy
        loss = (img_grey * np.log(1 / (img_rendered_grey + _delta))).sum() / (norm_factor + _delta)

        return loss * lossScale

    # Convert pixel indices to positions
    # (iy, ix) -> (xs, ys)
    def Idx2Pos(self, iy, ix):
        xs = (ix + 0.5 - self.size_x / 2) * self.delta_xy
        ys = (iy + 0.5 - self.size_y / 2) * -self.delta_xy

        return xs, ys

    # [n_cluster] -> [ix, iy] data spread
    def Cluster2Map(self, params_cluster):
        n_params = params_cluster.shape[1]
        params_map = np.zeros((self.size_y, self.size_x, n_params))
        for n in range(self.clusterer.n_clusters):
            iy, ix = self.clusterer.Indices(n)
            params_map[(iy, ix)] = params_cluster[n]
        return params_map

    # [ix, iy] -> [n_cluster] data reduce
    def Map2Cluster(self, params_map):
        n_params = params_map.shape[2]
        params_cluster = np.zeros((self.clusterer.n_clusters, n_params))
        for n in range(self.clusterer.n_clusters):
            iy, ix = self.clusterer.Indices(n)
            params_cluster[n] = params_map[(iy, ix)].mean(axis=0)  # Arithmetic mean as cluster's value
        return params_cluster

    # Fit BRDF parameters and bumps (normals and tangents)
    def Process(self, n_iter=5, height_scale=0.5, fracPrint=0.5):
        # Transform positions to our coordinate system
        # Note that the first pixel's center has an offset of 0.5 pixel size away from the border

        if self.isProcessed:
            self.InitParams()

        self.isProcessed = True

        t_process_begin = time()

        # sigma drops exponentially from 27.0 to 0.1 after n_iter times
        # factr: 1e12 -> 1e7
        sigma = 27.0
        sigma_final = 0.1
        sigma_atten = np.exp(-np.log(sigma / sigma_final) / n_iter)
        factr = 1e12
        factr_final = 1e4
        factr_atten = np.exp(-np.log(factr / factr_final) / n_iter)


        print('Number of clusters: {:d}'.format(self.clusterer.n_clusters))
        # Loop n_iter times
        for i in range(n_iter):
            print('{:d} in {:d} iterations.'.format(i + 1, n_iter))

            iprint = np.round(self.clusterer.n_clusters * fracPrint).astype(np.int)

            ####################################################################################################
            # Compute height field, find optimal scale, detail factor
            ####################################################################################################
            # In case during the first iteration, normal_map is not computed yet
            if i == 0:
                self.bump_map[..., 0] = 0
                self.bump_cluster = self.Map2Cluster(self.bump_map)
                normal_map = np.zeros((self.size_y, self.size_x, 3), dtype=np.float)
                normal_map[..., 2] = 1.0

            else:
                shading = Rgb2Lum(
                    self.img_ambient * self.ambient_exposure_time / np.clip(self.brdf_map[..., 0:3], 0.001, 1))
                # Normalized so that shading map's average value is 0.5

                shading = shading * 0.5 * self.size_y * self.size_x / shading.sum()
                shading = Brdf.saturate(shading)

                kernels = [1, 2, 4, 8]
                shading_blurred = np.zeros((self.size_y, self.size_x, 4))
                # Gaussian blur
                for i in range(4):
                    shading_blurred[..., i] = cv2.GaussianBlur(shading, (0, 0), kernels[i])

                for i in range(3):
                    shading_blurred[..., i] = (shading_blurred[..., i] * 0.5 / shading_blurred[..., i + 1] + _delta)

                height_map = np.zeros((self.size_y, self.size_x), dtype=np.float)
                for i in range(3, -1, -1):
                    # Convert shading increment to depth **in place**
                    shading_inc = shading_blurred[..., i]
                    mask = shading_inc >= 0.5
                    shading_inc[mask] = 2 * (1 - shading_inc[mask])
                    mask = np.logical_not(mask)
                    shading_inc[mask] = np.sqrt(1 / (shading_inc[mask] + _delta) - 1)

                    shading_inc -= 1  # Mapped to have mean depth of 0

                    height_map -= shading_inc * kernels[i]  # Note: depth and height have inverse meaning

                # plt.imshow(height_map, cmap=plt.get_cmap('Greys'))
                # plt.show()

                # Give a global scale for height map
                self.bump_map[..., 0] = height_map * height_scale

                # plt.imshow(height, cmap=plt.get_cmap('Greys'))
                # plt.show()

                # Calculate a normal map to be used by optimization steps of next iteration
                normal_map = Brdf.ComputeNormal(self.bump_map[..., 0])
                # plt.imshow(PackNormalTangentMap(normal_map))
                # plt.show()

                self.bump_cluster = self.Map2Cluster(self.bump_map)

            ####################################################################################################
            # Optimize roughness
            ####################################################################################################
            t0 = time()
            self.brdf_global[1], func_val, dic = optimize.fmin_l_bfgs_b(func=self.RoughnessFitLossKernel,
                                                                     x0=self.brdf_global[1],
                                                                     args=(
                                                                         self.brdf_global[0],
                                                                         self.light_position,
                                                                         self.light_intensity,
                                                                         self.delta_xy,
                                                                         self.brdf_map,
                                                                         self.bump_map,
                                                                         self.img_point
                                                                     ),
                                                                     factr=factr, approx_grad=True,
                                                                     bounds=[(0, 1)],
                                                                     iprint=-1, maxiter=20)

            print(
                "global BRDF params fitting done in {:.3f}s. Roughness: {:.3f}; Loss: {:.3f}.".format(
                    time() - t0, self.brdf_global[1], func_val), dic)


            ####################################################################################################
            # Optimize spatially varying BRDF parameters on each cluster
            ####################################################################################################
            for n in range(self.clusterer.n_clusters):
                t0 = time()
                iy, ix = self.clusterer.Indices(n)
                xs, ys = self.Idx2Pos(iy, ix)
                # The iteration stops when (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr * eps, where eps is the machine precision
                # numpy.finfo(float).eps == 2.2e-16
                brdf_anisoaxis_cluster, func_val, dic = optimize.fmin_l_bfgs_b(func=self.ClusterRenderLossKernel,
                                                                             x0=np.concatenate((self.brdf_cluster[n],
                                                                                                self.bump_cluster[n][1:2])),
                                                                             args=(
                                                                                 self.brdf_global,
                                                                                 normal_map[(iy, ix)],
                                                                                 (xs, ys),
                                                                                 self.img_point[(iy, ix)].reshape(-1,
                                                                                                                  3),
                                                                                 self.light_position,
                                                                                 self.light_intensity
                                                                             ),
                                                                             factr=factr, approx_grad=True,
                                                                             bounds=[(0, 1)] * (self.brdf_cluster.shape[
                                                                                 1]+1), iprint=-1, maxiter=5000)
                self.brdf_cluster[n] = brdf_anisoaxis_cluster[:-1]
                self.bump_cluster[n][1] = brdf_anisoaxis_cluster[-1]
                if iprint == 0 or n % iprint == 0:
                    print(
                        "BRDF params and aniso axis fitting for cluster {:d} done in {:.3f}s. Function calls: {:d}, iterations: {:d}.".format(
                            n, time() - t0, dic['funcalls'], dic['nit']))

            self.bump_map[..., 1:2] = self.Cluster2Map(self.bump_cluster[..., 1:2])
            self.brdf_map = self.Cluster2Map(self.brdf_cluster)

            ####################################################################################################
            # Gaussian filtering
            ####################################################################################################

            # In OpenCV, if sigma is not specified, then it is computed as: sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
            # If ksize is not specified, it is computed as: ksize = cvRound(sigma*(depth == CV_8U ? 3 : 4)*2 + 1)|1
            # We use the second method.
            self.brdf_map[..., 0:3] = cv2.GaussianBlur(self.brdf_map[..., 0:3], (0, 0), sigma)
            self.brdf_map[..., 3] = cv2.GaussianBlur(self.brdf_map[..., 3], (0, 0), sigma)
            self.brdf_map[..., 4] = cv2.GaussianBlur(self.brdf_map[..., 4], (0, 0), sigma)
            self.brdf_map[..., 5] = cv2.GaussianBlur(self.brdf_map[..., 5], (0, 0), sigma)
            # Note that shape of brdf_cluster is changed
            self.brdf_cluster = self.Map2Cluster(self.brdf_map)

            # Height map need not to be blurred
            self.bump_map[..., 1] = cv2.GaussianBlur(self.bump_map[..., 1], (0, 0), sigma)
            self.bump_cluster = self.Map2Cluster(self.bump_map)

            ####################################################################################################
            # Save
            ####################################################################################################
            self.SaveMap()

            ####################################################################################################
            # Update parameters
            ####################################################################################################
            sigma *= sigma_atten
            factr *= factr_atten


        print('Fitting done. Time consumption: {}'.format(datetime.timedelta(seconds=time() - t_process_begin)))
        img_rendered, error = self.RenderImage(returnError=True)
        WriteImgAsUint8(self.default_output_path + '/rerendered.jpg', img_rendered)
        WriteImg(self.default_output_path + '/error.jpg', Brdf.saturate((error * error).sum(axis=2) / 3.0))
        return

    def RenderImage(self, light_position=None, light_intensity=None, exposure=None, hdr=False, returnError=False, hasNormalTangentMap=False):
        if light_position is None:
            light_position = self.light_position
        if light_intensity is None:
            light_intensity = self.light_intensity
        if exposure is None:
            exposure = self.exposure_time

        if hasNormalTangentMap:
            normal_map = self.normal_map
            tangent_map = self.tangent_map
            bitangent_map = Brdf.normalize(self.cross(normal_map, tangent_map))
        else:
            normal_map = Brdf.ComputeNormal(self.bump_map[..., 0])
            tangent_map, bitangent_map, normal_map = Brdf.ComputeBump(self.bump_map[..., 1], normal_map)

        img_rendered = Brdf.ImageRenderKernel(self.brdf_global, light_position, light_intensity, self.delta_xy,
                                              self.brdf_map,
                                              tangent_map, bitangent_map, normal_map)
        if hdr:
            if not returnError:
                return img_rendered
            else:
                return img_rendered, img_rendered - self.img_point
        else:
            img_rendered = Radiance2Img(img_rendered, exposure, self.response_curve)
            if not returnError:
                return img_rendered
            else:
                return img_rendered, img_rendered / zmax - Radiance2Img(self.img_point, exposure,
                                                                        self.response_curve) / zmax

    # Save fitted BRDF parameters and bump info as images
    def SaveMap(self, path=None, fromCluster=False, format='jpg'):
        if path is None:
            path = self.default_output_path

        if not os.path.isdir(path):
            os.makedirs(path)

        if not 'pfm' == format.lower():
            gammaEncoding = True
        else:
            gammaEncoding = False

        map_files = {
            'basecolor': 'basecolor.{}'.format(format),
            'specular': 'specular.{}'.format(format),
            'speculartint': 'speculartint.{}'.format(format),
            'anisotropic': 'anisotropic.{}'.format(format),
            'normal': 'normal.{}'.format(format),
            'tangent': 'tangent.{}'.format(format)
        }
        brdf_global = {
            'metallic': self.brdf_global[0],
            'roughness': self.brdf_global[1]
        }

        basecolor_path = path + '/' + map_files['basecolor']
        specular_path = path + '/' + map_files['specular']
        speculartint_path = path + '/' + map_files['speculartint']
        anisotropic_path = path + '/' + map_files['anisotropic']
        normal_path = path + '/' + map_files['normal']
        tangent_path = path + '/' + map_files['tangent']

        if fromCluster:
            brdf_map = np.zeros((self.size_y, self.size_x, self.brdf_cluster.shape[1]))
            bump_map = np.zeros((self.size_x, self.size_y, self.bump_cluster.shape[1]))
            for n in range(self.clusterer.n_clusters):
                iy, ix = self.clusterer.Indices(n)
                brdf_map[(iy, ix)] = self.brdf_cluster[n]
                bump_map[(iy, ix)] = self.bump_cluster[n]
        else:
            brdf_map = self.brdf_map
            bump_map = self.bump_map

        normal = Brdf.ComputeNormal(bump_map[..., 0])
        tangent, bitangent, normal = Brdf.ComputeBump(bump_map[..., 1], normal)

        tangent = PackNormalTangentMap(tangent)
        normal = PackNormalTangentMap(normal)

        # Implement gamma encoding only when storing basecolor
        WriteImg(basecolor_path, brdf_map[:, :, 0:3], gammaEncoding)
        WriteImg(specular_path, brdf_map[:, :, 3], False)
        WriteImg(speculartint_path, brdf_map[:, :, 4], False)
        WriteImg(anisotropic_path, brdf_map[:, :, 5], False)

        WriteImg(tangent_path, tangent, False)
        WriteImg(normal_path, normal, False)

        # Save json config file
        config = {
            'size_x': self.size_x,
            'size_y': self.size_y,
            'efl_35mm': self.efl_35mm,
            'delta_xy': self.delta_xy,
            'light_position': self.light_position,
            'light_intensity': self.light_intensity,
            'map_files': map_files,
            'brdf_global': brdf_global,
            'exposure_time': self.exposure_time,
            'response_curve': self.response_curve.tolist()
        }

        with open(path + '/' + self.default_config_file, 'w') as f:
            json.dump(config, f, indent=4)

        return

    # TODO: Config`已经改了
    def LoadMap(self, path=None, configFile=None):
        if path is None:
            path = self.default_output_path
        if configFile is None:
            configFile = self.default_config_file

        with open(path + '/' + configFile, 'r') as f:
            config = json.load(f)

        self.size_x = config['size_x']
        self.size_y = config['size_y']

        self.efl_35mm = config['efl_35mm']
        self.delta_xy = config['delta_xy']
        self.light_position = np.copy(config['light_position'])
        self.light_intensity = config['light_intensity']

        map_files = config['map_files']


        self.brdf_global = np.zeros(2)
        self.brdf_global[0] = config['brdf_global']['metallic']
        self.brdf_global[1] = config['brdf_global']['roughness']

        self.brdf_map = np.zeros((self.size_y, self.size_x, 6))
        if map_files['basecolor'] is not None:
            basecolor_path = path + '/' + map_files['basecolor']
            self.brdf_map[:, :, 0:3] = ReadImg(basecolor_path, True)
        if map_files['specular'] is not None:
            specular_path = path + '/' + map_files['specular']
            self.brdf_map[:, :, 3] = ReadImg(specular_path, False, True)
        if map_files['speculartint'] is not None:
            speculartint_path = path + '/' + map_files['speculartint']
            self.brdf_map[:, :, 4] = ReadImg(speculartint_path, False, True)
        if map_files['anisotropic'] is not None:
            anisotropic_path = path + '/' + map_files['anisotropic']
            self.brdf_map[:, :, 5] = ReadImg(anisotropic_path, False, True)
        if map_files['normal'] is not None:
            normal_path = path + '/' + map_files['normal']
            self.normal_map = UnpackNormalTangentMap( ReadImg(normal_path, False))
        if map_files['tangent'] is not None:
            tangent_path = path + '/' + map_files['tangent']
            self.tangent_map = UnpackNormalTangentMap(ReadImg(tangent_path, False))

        self.exposure_time = config['exposure_time']
        self.response_curve = np.copy(config['response_curve'])

        return
