import os
import matplotlib.pyplot as plt
import pylab
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import leastsq, minimize
from scipy import interpolate
from utils import imread
from utils import normxcorr2
from utils import rectify_image


class AutoRect:
    '''
        Params:
            img_bg = Background image with circles
            img = image to be rectified
            nnum = pixels behind each microlens
            pixel_steps = steps between each lineprofile [in pixels]
            average_range = amount of pixels over which lineprofile is averaged
            pixel_diff = absolute cutoff for dropping column due to misalignment (in function align_swipe_peaks)
            min_seperation = minimum seperation in pixels of detected minima (assures that only one minimum is found for each microlens)
    '''
    def __init__(self, img_bg, img=None, nnum=19, pitch=None, center_x_rect_bg=None, center_y_rect_bg=None, rotation=None, plot_grid_fit=False, use_rotation=False):
        # TODO: implement use_rotation = True. have to adapt shift due to rotation in convert_fit_params_to_rect_params
        self.img_bg = img_bg
        self.img = img
        self.nnum = nnum
        self.peaks_x = None
        self.peaks_y = None
        self.df_swipe_peaks_x = None
        self.df_swipe_peaks_y = None
        self.df_swipe_peaks_aligned_x = None
        self.df_swipe_peaks_aligned_y = None
        self.df_peak_ranges_x = None
        self.df_peak_ranges_y = None
        self.df_fitted_peaks_x = None
        self.df_fitted_peaks_y = None
        self.df_fitted_lines_x = None
        self.df_fitted_lines_y = None
        self.df_fitted_lines_y_vert = None
        self.inters_reshape_x = None
        self.inters_reshape_y = None
        self.pitch = pitch
        self.center_x_rect_bg = center_x_rect_bg
        self.center_y_rect_bg = center_y_rect_bg
        self.rotation = rotation
        self.center_x_rect_img = None
        self.center_y_rect_img = None
        self.img_rect_bg = None
        self.img_rect = None
        self.plot_grid_fit = plot_grid_fit
        self.use_rotation = use_rotation
        self._set_params()

    def _set_params(self):
        self._start_pos_x = None
        self._start_pos_y = None
        self._end_pos_x = None
        self._end_pos_y = None
        self._average_range = 7
        self._min_seperation = 2
        self._cut_lenselets = 3
        self._pixel_steps = 3
        self._pixel_diff = 3
        self._pixel_tolerance = 1
                                    
    def _get_start_end_pos_x(self):
        self._start_pos_x = 1 + self._average_range
        self._end_pos_x = self.img_bg.shape[0] - self._average_range
        return self._start_pos_x, self._end_pos_x 

    def _get_start_end_pos_y(self):
        self._start_pos_y = 1 + self._average_range
        self._end_pos_y = self.img_bg.shape[1] - self._average_range
        return self._start_pos_y, self._end_pos_y

    def _get_mean_lineprofile_x(self, pos):
        x_range = range(pos - self._average_range, pos + self._average_range)
        lineprofile_x = self.img_bg[x_range, :].mean(axis=0)   
        return lineprofile_x

    def _get_mean_lineprofile_y(self, pos):
        y_range = range(pos - self._average_range, pos + self._average_range)
        lineprofile_y = self.img_bg[:, y_range].mean(axis=1)
        return lineprofile_y

    def _find_local_minima(self, lineprofile):
        if self._min_seperation == 0:
            peaks, _ = find_peaks(lineprofile * -1, 
                            height=lineprofile.mean() * -1)
        else:
            peaks, _ = find_peaks(lineprofile * -1, 
                                height=lineprofile.mean() * -1,
                                distance=self.nnum - self._min_seperation)
        return peaks

    def swipe_lineprofiles_x(self):
        self._start_pos_x, self._end_pos_x = self._get_start_end_pos_x()  
        swipe_lineprofiles = dict()
        swipe_peaks = dict()
        for pos in np.arange(self._start_pos_x, self._end_pos_x, self._pixel_steps):
            lineprofile = self._get_mean_lineprofile_x(pos)
            swipe_lineprofiles[f"x__{pos}"] = lineprofile
            swipe_peaks[f"x__{pos}"] = self._find_local_minima(lineprofile)
        self.df_swipe_lineprofiles_x = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in swipe_lineprofiles.items() ]))
        self.df_swipe_peaks_x = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in swipe_peaks.items() ]))
        return self.df_swipe_lineprofiles_x, self.df_swipe_peaks_x

    def swipe_lineprofiles_y(self):
        self._start_pos_y, self._end_pos_y = self._get_start_end_pos_y()  
        swipe_lineprofiles = dict()
        swipe_peaks = dict()
        for pos in np.arange(self._start_pos_y, self._end_pos_y, self._pixel_steps):
            lineprofile = self._get_mean_lineprofile_y(pos)
            swipe_lineprofiles[f"y__{pos}"] = lineprofile
            swipe_peaks[f"y__{pos}"] = self._find_local_minima(lineprofile)
        self.df_swipe_lineprofiles_y = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in swipe_lineprofiles.items() ]))
        self.df_swipe_peaks_y = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in swipe_peaks.items() ]))
        return self.df_swipe_lineprofiles_y, self.df_swipe_peaks_y

    def align_swipe_peaks_x(self):
        df = self.df_swipe_peaks_x.copy()
        df['median_swipe'] = df.median(axis=1)
        df = df.apply(lambda x : np.where(np.isclose(df.median_swipe, x, atol=self._pixel_diff),
                                                x, np.NAN))
        df = df.drop('median_swipe', axis=1)
        self.df_swipe_peaks_aligned_x = df.copy()
        return self.df_swipe_peaks_aligned_x

    def align_swipe_peaks_y(self):
        df = self.df_swipe_peaks_y.copy()
        df['median_swipe'] = df.median(axis=1)
        df = df.apply(lambda x : np.where(np.isclose(df.median_swipe, x, atol=self._pixel_diff),
                                                x, np.NAN))
        df = df.drop('median_swipe', axis=1)
        self.df_swipe_peaks_aligned_y = df.copy()
        return self.df_swipe_peaks_aligned_y

    def get_peak_ranges_x(self):
        df = self.df_swipe_peaks_aligned_x.copy()
        range_size = int(self.nnum/2)
        for j in range(df.shape[1]):
            try:
                df.iloc[:,j] = df.iloc[:,j].apply(lambda x: list(range(int(x)-range_size, int(x)+range_size)))
            except ValueError:
                pass
        self.df_peak_ranges_x = df.copy()
        return self.df_peak_ranges_x

    def get_peak_ranges_y(self):
        df = self.df_swipe_peaks_aligned_y.copy()
        range_size = int(self.nnum/2)
        for j in range(df.shape[1]):
            try:
                df.iloc[:,j] = df.iloc[:,j].apply(lambda x: list(range(int(x)-range_size, int(x)+range_size)))
            except ValueError:
                pass
        self.df_peak_ranges_y = df.copy()
        return self.df_peak_ranges_y

    def _find_fitted_positions_x(self):
        df = self.df_swipe_peaks_aligned_x.copy()
        df = df[df.isna()]
        for i in range(self.df_peak_ranges_x.shape[0]):
            for j in range(self.df_peak_ranges_x.shape[1]):
                try:
                    x = self.df_peak_ranges_x.iloc[i,j]
                    y = self.df_swipe_lineprofiles_x.iloc[x,j]
                    f = interp1d(x,y)
                    xnew = np.linspace(x[0], x[-1], 100)
                    ynew = f(xnew)
                    p1,p2,p3 = np.polyfit(xnew, ynew, deg=2)
                    min_pos = -p2/(2*p1)
                    df.iloc[i,j] = min_pos
                except ValueError:
                    pass
                except IndexError:
                    pass        
        return df

    def _find_fitted_positions_y(self):
        df = self.df_swipe_peaks_aligned_y.copy()
        df = df[df.isna()]
        for i in range(self.df_peak_ranges_y.shape[0]):
            for j in range(self.df_peak_ranges_y.shape[1]):
                try:
                    x = self.df_peak_ranges_y.iloc[i,j]
                    y = self.df_swipe_lineprofiles_y.iloc[x,j]
                    f = interp1d(x,y)
                    xnew = np.linspace(x[0], x[-1], 100)
                    ynew = f(xnew)
                    p1,p2,p3 = np.polyfit(xnew, ynew, deg=2)
                    min_pos = -p2/(2*p1)
                    df.iloc[i,j] = min_pos
                except ValueError:
                    pass
                except IndexError:
                    pass        
        return df

    def find_minima_positions_averaged_x(self):
        self.df_swipe_lineprofiles_x, self.df_swipe_peaks_x = self.swipe_lineprofiles_x()
        self.df_swipe_peaks_aligned_x = self.align_swipe_peaks_x()
        self.df_peak_ranges_x = self.get_peak_ranges_x() 
        df_fitted_peaks_x = self._find_fitted_positions_x()
        df_fitted_peaks_x[df_fitted_peaks_x < 0] = np.nan
        df_fitted_peaks_x = df_fitted_peaks_x.dropna(axis=0, thresh=10)
        df_fitted_peaks_x = df_fitted_peaks_x.dropna(axis=1, thresh=10)
        if self._cut_lenselets is not None:
            range_after_cut = range(self._cut_lenselets, df_fitted_peaks_x.shape[0] - self._cut_lenselets)
            df_fitted_peaks_x = df_fitted_peaks_x.iloc[range_after_cut,:]
        self.df_fitted_peaks_x = df_fitted_peaks_x.copy()
        return self.df_fitted_peaks_x

    def find_minima_positions_averaged_y(self):
        self.df_swipe_lineprofiles_y, self.df_swipe_peaks_y = self.swipe_lineprofiles_y()
        self.df_swipe_peaks_aligned_y = self.align_swipe_peaks_y()
        self.df_peak_ranges_y = self.get_peak_ranges_y() 
        self.df_fitted_peaks_y = self._find_fitted_positions_y()
        df_fitted_peaks_y = self._find_fitted_positions_y()
        df_fitted_peaks_y[df_fitted_peaks_y < 0] = np.nan
        df_fitted_peaks_y = df_fitted_peaks_y.dropna(axis=0, thresh=10)
        df_fitted_peaks_y = df_fitted_peaks_y.dropna(axis=1, thresh=10)
        if self._cut_lenselets is not None:
            range_after_cut = range(self._cut_lenselets, df_fitted_peaks_y.shape[0] - self._cut_lenselets)
            df_fitted_peaks_y = df_fitted_peaks_y.iloc[range_after_cut,:]
        self.df_fitted_peaks_y = df_fitted_peaks_y.copy()
        return self.df_fitted_peaks_y

    def linear_fit_of_minima_to_borderline_x(self):
        self.df_fitted_peaks_x = self.find_minima_positions_averaged_x()
        df = self.df_fitted_peaks_x.copy()
        fitted_lines = list()
        for row in range(df.shape[0]):
            coord_pos_values = list(df.columns.str.split(f'x__').str[1].astype(int))
            coord_pos_values = np.array(coord_pos_values)
            fitted_peaks = df.iloc[row, :].values
            filter_na = ~np.isnan(fitted_peaks)
            fitted_peaks = fitted_peaks[filter_na]  
            coord_pos_values = coord_pos_values[filter_na]
            try:
                fitted_lines.append(list(np.polyfit(coord_pos_values, fitted_peaks, deg=1)))
            except TypeError:
                pass
            del coord_pos_values
        df_fitted_lines = pd.DataFrame.from_records(fitted_lines)
        self.df_fitted_lines_x = df_fitted_lines.rename(columns={0: 'slope', 1: 'intercept'})
        return self.df_fitted_lines_x

    def linear_fit_of_minima_to_borderline_y(self):
        self.df_fitted_peaks_y = self.find_minima_positions_averaged_y()
        df = self.df_fitted_peaks_y.copy()
        fitted_lines = list()
        for row in range(df.shape[0]):
            coord_pos_values = list(df.columns.str.split(f'y__').str[1].astype(int))
            coord_pos_values = np.array(coord_pos_values)
            fitted_peaks = df.iloc[row, :].values
            filter_na = ~np.isnan(fitted_peaks)
            fitted_peaks = fitted_peaks[filter_na]  
            coord_pos_values = coord_pos_values[filter_na]
            try:
                fitted_lines.append(list(np.polyfit(coord_pos_values, fitted_peaks, deg=1)))
            except TypeError:
                pass
            del coord_pos_values
        df_fitted_lines = pd.DataFrame.from_records(fitted_lines)
        self.df_fitted_lines_y = df_fitted_lines.rename(columns={0: 'slope', 1: 'intercept'})
        return self.df_fitted_lines_y

    def get_intersections_of_fitted_lines(self):
        inters_x_list = list()
        inters_y_list = list()
        self.df_fitted_lines_y_vert = self.df_fitted_lines_y.copy()
        self.df_fitted_lines_y_vert['slope'] = 1/self.df_fitted_lines_y['slope']
        self.df_fitted_lines_y_vert['intercept'] = -self.df_fitted_lines_y['intercept']/self.df_fitted_lines_y['slope']
        for i in range(self.df_fitted_lines_x.shape[0]):
            for j in range(self.df_fitted_lines_y_vert.shape[0]):
                inters_x = (self.df_fitted_lines_x.iloc[i,1] - self.df_fitted_lines_y_vert.iloc[j,1]) / (self.df_fitted_lines_y_vert.iloc[j,0] - self.df_fitted_lines_x.iloc[i,0])
                inters_y = self.df_fitted_lines_x.iloc[i,0] * inters_x + self.df_fitted_lines_x.iloc[i,1] 
                inters_x_list.append(inters_x)
                inters_y_list.append(inters_y)
        self.inters_reshape_x = np.reshape(inters_x_list, (self.df_fitted_lines_x.shape[0], self.df_fitted_lines_y_vert.shape[0]))
        self.inters_reshape_y = np.reshape(inters_y_list, (self.df_fitted_lines_x.shape[0], self.df_fitted_lines_y_vert.shape[0]))
        return self.inters_reshape_x, self.inters_reshape_y

    def plot_grid_points(self, x_grid, y_grid, xe, ye, pitch, center_x, center_y):
        plt.figure(figsize=(30,30))
        plt.plot(x_grid, y_grid, "xr")
        plt.plot(xe, ye, "xb")
        plt.plot(center_x, center_y, 'xy', ms=20,linewidth=20)
        plt.title(f'{pitch:.2f}, {center_x:.2f}, {center_y:.2f}')
        plt.show()

    def _get_spot_grid(self, shape, pitch, center_x, center_y, rotation=0):
        # TODO: only correctly works for rotation == 0, otherwise have to adapt shift due to rotation in convert_fit_params_to_rect_params
        x_spots, y_spots = np.meshgrid(
                (np.arange(shape[1]) - (shape[1]-1)/2.)*pitch, 
                (np.arange(shape[0]) - (shape[0]-1)/2.)*pitch)
        if self.use_rotation == False:
            theta = 0
        else:                            
            theta = rotation/180.*np.pi
        x_spots = x_spots*np.cos(theta) - y_spots*np.sin(theta) + center_x
        y_spots = x_spots*np.sin(theta) + y_spots*np.cos(theta) + center_y
        return x_spots, y_spots

    def _get_mean_distance(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2).mean()

    def err_func(self, params, xe, ye):
        pitch, center_x, center_y, rotation = params
        x_grid, y_grid = self._get_spot_grid(xe.shape, pitch, center_x, center_y, rotation)
        if self.plot_grid_fit == True:
            self.plot_grid_points(x_grid, y_grid, xe, ye, pitch, center_x, center_y)
        return self._get_mean_distance(x_grid, y_grid, xe, ye)
        
    def convert_fit_params_to_rect_params(self):  
        # TODO: put some asserts here
        min_distance_x = np.abs((pd.DataFrame(self.inters_reshape_x) - self.center_x_rect_bg)).min().min()
        min_distance_y = np.abs((pd.DataFrame(self.inters_reshape_y) - self.center_y_rect_bg)).min().min()
        if min_distance_x > self.nnum/4:
            shift = np.abs((np.arange(self.inters_reshape_x.shape[1]) - (self.inters_reshape_x.shape[1]-1)/2.)*self.pitch).min()
            self.center_x_rect_bg = self.center_x_rect_bg + shift
        if min_distance_y > self.nnum/4:
            shift = np.abs((np.arange(self.inters_reshape_x.shape[0]) - (self.inters_reshape_x.shape[0]-1)/2.)*self.pitch).min()
            self.center_y_rect_bg = self.center_y_rect_bg + shift
        self.center_x_rect_bg = self.center_x_rect_bg + self.pitch/2
        self.center_y_rect_bg = self.center_y_rect_bg + self.pitch/2
        return self.pitch, self.center_x_rect_bg, self.center_y_rect_bg, self.rotation

    def rect_params_img_bg(self):
        self.df_fitted_lines_x = self.linear_fit_of_minima_to_borderline_x()
        self.df_fitted_lines_y = self.linear_fit_of_minima_to_borderline_y()
        self.inters_reshape_x, self.inters_reshape_y = self.get_intersections_of_fitted_lines()
        # TODO: somewhere is a weird flip with x and y. This fixes it.
        inter_reshape_x_updated = self.inters_reshape_y.transpose()
        inter_reshape_y_updated = self.inters_reshape_x.transpose()
        self.inters_reshape_x = None
        self.inters_reshape_y = None
        self.inters_reshape_x = inter_reshape_x_updated
        self.inters_reshape_y = inter_reshape_y_updated

        results = minimize(self.err_func, x0=(self.nnum, 0, 0, 0), args=(self.inters_reshape_x, self.inters_reshape_y))
        self.pitch, self.center_x_rect_bg, self.center_y_rect_bg, self.rotation = results.x
        self.pitch, self.center_x_rect_bg, self.center_y_rect_bg, self.rotation = self.convert_fit_params_to_rect_params()
        return self.pitch, self.center_x_rect_bg, self.center_y_rect_bg, self.rotation

    def register_img_to_bg(self):
        if None in [self.pitch, self.center_x_rect_bg, self.center_y_rect_bg]:
            self.pitch, self.center_x_rect_bg, self.center_y_rect_bg, self.rotation = self.rect_params_img_bg()
        c = normxcorr2(self.img_bg, self.img)
        index_max = np.argmax(c)
        x_peak, y_peak = np.unravel_index(index_max, c.shape)
        corr_offset_x = (x_peak + 1) - self.img_bg.shape[0]
        corr_offset_y = (y_peak + 1) - self.img_bg.shape[1]
        self.center_x_rect_img = self.center_x_rect_bg + corr_offset_x
        self.center_y_rect_img = self.center_y_rect_bg + corr_offset_y
        return self.pitch, self.center_x_rect_img, self.center_y_rect_img

    def rectify_img_bg(self):
        if None in [self.pitch, self.center_x_rect_bg, self.center_y_rect_bg]:
            self.pitch, self.center_x_rect_bg, self.center_y_rect_bg, self.rotation = self.rect_params_img_bg() 
        self.img_rect_bg = rectify_image(img=self.img_bg, pitch=self.pitch, center_x=self.center_x_rect_bg, center_y=self.center_y_rect_bg, nnum=self.nnum)
        self.check_rectification_quality()
        return self.img_rect_bg

    def rectify_img(self):
        if None in [self.pitch, self.center_x_rect_bg, self.center_y_rect_bg]:
            self.pitch, self.center_x_rect_bg, self.center_y_rect_bg, self.rotation = self.rect_params_img_bg()
        if None in [self.pitch, self.center_x_rect_img, self.center_y_rect_img]:
            self.pitch, self.center_x_rect_img, self.center_y_rect_img = self.register_img_to_bg()
        self.img_rect = rectify_image(img=self.img, pitch=self.pitch, center_x=self.center_x_rect_img, center_y=self.center_y_rect_img, nnum=self.nnum)
        self.check_rectification_quality(img=self.img_rect)
        return self.img_rect

    def check_rectification_quality(self, img=None):
        if img is None:
            img = self.img_rect_bg
        lineprofile_x = img.mean(axis=0)
        lineprofile_y = img.mean(axis=1)
        peaks_x, _  = find_peaks(lineprofile_x * -1, 
                                height=lineprofile_x.mean() * -1,
                                distance=self.nnum - self._min_seperation)
        peaks_y, _  = find_peaks(lineprofile_y * -1, 
                                height=lineprofile_y.mean() * -1,
                                distance=self.nnum - self._min_seperation)
        try:
            assert np.isclose(peaks_x[0], self.nnum, atol=self._pixel_tolerance)
        except AssertionError: 
            print('center_x_rect_bg was not correctly determined.')
            raise
        try:
            assert np.isclose(peaks_y[0], self.nnum, atol=self._pixel_tolerance)
        except AssertionError: 
            print('center_y_rect_bg was not correctly determined.')
            raise

    def plot_lineprofile_x(self, pos, range_plot=None):
        # TODO: fix bug, for range_plot not None
        if range_plot == None:
            range_plot = range(self.img_bg.shape[1])
        lineprofile_x = self._get_mean_lineprofile_x(pos)
        print(lineprofile_x.shape)
        plt.plot(range_plot, lineprofile_x[range_plot])
        peaks, _ = find_peaks(lineprofile_x[range_plot] * -1, 
                        height=lineprofile_x[range_plot].mean() * -1,
                        distance=self.nnum - self._min_seperation)
        plt.plot(peaks + range_plot[0], lineprofile_x[peaks], "x")
        plt.show()
    
    def plot_lineprofile_y(self, pos, range_plot=None):
        # TODO: fix bug, for range_plot not None
        if range_plot == None:
            range_plot = range(self.img_bg.shape[0])
        lineprofile_x = self._get_mean_lineprofile_x(pos)
        print(lineprofile_x.shape)
        plt.plot(range_plot, lineprofile_x[range_plot])
        peaks, _ = find_peaks(lineprofile_x[range_plot] * -1, 
                        height=lineprofile_x[range_plot].mean() * -1,
                        distance=self.nnum - self._min_seperation)
        plt.plot(peaks + range_plot[0], lineprofile_x[peaks], "x")
        plt.show()

    def plot_line_of_fitted_peaks_x(self, range_plot=None):
        if self.df_fitted_peaks_x is None:
            self.df_fitted_peaks_x = self.find_minima_positions_averaged_x()
        df = self.df_fitted_peaks_x
        if range_plot == None:
            range_plot = range(df.shape[0])
        coord_pos_values = list(df.columns.str.split(f'x__').str[1].astype(int))
        for row in range_plot:
            fitted_peaks = df.iloc[row, :].values
            y_mean = fitted_peaks.mean()
            plt.plot(coord_pos_values, fitted_peaks, 'b')

    def plot_line_of_fitted_peaks_y(self, range_plot=None):
        if self.df_fitted_peaks_y is None:
            self.df_fitted_peaks_y = self.find_minima_positions_averaged_y()
        df = self.df_fitted_peaks_y
        if range_plot == None:
            range_plot = range(df.shape[0])
        coord_pos_values = list(df.columns.str.split(f'y__').str[1].astype(int))
        for row in range_plot:
            fitted_peaks = df.iloc[row, :].values
            y_mean = fitted_peaks.mean()
            plt.plot(fitted_peaks, coord_pos_values, 'b')

    def plot_intersection_on_img_bg(self, i=10, j=10, img_size_show=30):
        if self.df_fitted_lines_x is None:
            self.df_fitted_lines_x = self.linear_fit_of_minima_to_borderline_x()
            self.df_fitted_lines_y = self.linear_fit_of_minima_to_borderline_y()
            self.df_fitted_lines_y_vert = self.df_fitted_lines_y.copy()
            self.df_fitted_lines_y_vert['slope'] = 1/self.df_fitted_lines_y['slope']
            self.df_fitted_lines_y_vert['intercept'] = -self.df_fitted_lines_y['intercept']/self.df_fitted_lines_y['slope']
        inters_x = (self.df_fitted_lines_x.iloc[i,1] - self.df_fitted_lines_y_vert.iloc[j,1]) / (self.df_fitted_lines_y_vert.iloc[j,0] - self.df_fitted_lines_x.iloc[i,0])
        inters_y = self.df_fitted_lines_x.iloc[i,0] * inters_x + self.df_fitted_lines_x.iloc[i,1] 
        plt.imshow(self.img_bg[round(inters_x)-img_size_show : round(inters_x)+img_size_show, 
                        round(inters_y)-img_size_show : round(inters_y)+img_size_show])
        plt.plot(inters_y-round(inters_y)+img_size_show, inters_x-round(inters_x)+img_size_show, 'rx')

    def plot_fitted_positions_x(self, i=10, j=10):
        if self.df_peak_ranges_x is None:
            self.df_swipe_lineprofiles_x, self.df_swipe_peaks_x = self.swipe_lineprofiles_x()
            self.df_swipe_peaks_aligned_x = self.align_swipe_peaks_x()
            self.df_peak_ranges_x = self.get_peak_ranges_x()
        x = self.df_peak_ranges_x.iloc[i,j]
        y = self.df_swipe_lineprofiles_x.iloc[x,j]
        f = interp1d(x,y)
        xnew = np.linspace(x[0], x[-1], 100)
        ynew = f(xnew)
        p1,p2,p3 = np.polyfit(xnew, ynew, deg=2)
        plt.plot(x, y, 'o', xnew, ynew, '-')
        min_pos = -p2/(2*p1)
        plt.plot(min_pos, f(min_pos), 'rx')

    def plot_fitted_positions_y(self, i=10, j=10):
        if self.df_peak_ranges_y is None:
            self.df_swipe_lineprofiles_y, self.df_swipe_peaks_y = self.swipe_lineprofiles_y()
            self.df_swipe_peaks_aligned_y = self.align_swipe_peaks_y()
            self.df_peak_ranges_y = self.get_peak_ranges_y()
        x = self.df_peak_ranges_y.iloc[i,j]
        y = self.df_swipe_lineprofiles_y.iloc[x,j]
        f = interp1d(x,y)
        xnew = np.linspace(x[0], x[-1], 100)
        ynew = f(xnew)
        p1,p2,p3 = np.polyfit(xnew, ynew, deg=2)
        plt.plot(x, y, 'o', xnew, ynew, '-')
        min_pos = -p2/(2*p1)
        plt.plot(min_pos, f(min_pos), 'rx')

    def plot_rect_quality_check(self, img=None, plot_range = 100, vmin=None, vmax=None):
        if img is None:
            img = self.img_rect_bg
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(15,40))
        ax1.imshow(img[:plot_range,:plot_range], vmin=vmin, vmax=vmax)
        ax1.set_title('top left')
        ax2.imshow(img[:plot_range:,-plot_range:], vmin=vmin, vmax=vmax)
        ax2.set_title('top right')
        ax3.imshow(img[-plot_range:,:plot_range], vmin=vmin, vmax=vmax)
        ax3.set_title('bottom left')
        ax4.imshow(img[-plot_range:,-plot_range:], vmin=vmin, vmax=vmax)
        ax4.set_title('bottom right')

    def plot_results_of_fit(self, x_range_lenselets=None, y_range_lenselets=None):
        x_resample = np.concatenate((np.flip(np.arange(self.center_x_rect_bg, 0, step=-self.pitch)), 
                                    np.arange(self.center_x_rect_bg, self.img_bg.shape[1], step=self.pitch)))
        y_resample = np.concatenate((np.flip(np.arange(self.center_y_rect_bg, 0, step=-self.pitch)), 
                                    np.arange(self.center_y_rect_bg, self.img_bg.shape[0], step=self.pitch)))
        x_mesh, y_mesh = np.meshgrid(x_resample, y_resample)  
        plt.imshow(self.img_bg)
        x_points = self.inters_reshape_x.flatten()
        y_points = self.inters_reshape_y.flatten()
        for i in range(x_points.shape[0]):
            if x_points[i] > 0 and y_points[i] > 0:
                plt.plot(x_points[i], y_points[i], 'yx')

        plt.plot(x_mesh, y_mesh, "xr")
        plt.plot(self.center_x_rect_bg, self.center_y_rect_bg, 'xb', ms=30,linewidth=30)

        if x_range_lenselets is not None:
            plt.xlim(self.center_x_rect_bg - self.nnum*x_range_lenselets, self.center_x_rect_bg + self.nnum*x_range_lenselets)
        if y_range_lenselets is not None:
            plt.ylim(self.center_y_rect_bg - self.nnum*y_range_lenselets, self.center_y_rect_bg + self.nnum*y_range_lenselets)
        plt.show()