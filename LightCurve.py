import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SequenceError(Exception):
    pass

class LightCurve():
    '''
    A class for deal with astronomical light curves

    A light curve should include at least 3 columns 
    which represent times,measurements,errors,respectively.
    
    data should be a pandas Dataframe with integer index

    the first column should always be times
    '''

    def __init__(self, sorted=True, time_fmt='mjd', measurement='mag', error='err'):
        '''
        Parameters
        ----------
        '''
        self.data = None
        self.folded = False
        self.sorted = sorted
        self.time_span = None
        self.phase_span = None
        self.period = None
        self.amplitude = None
        self.smoothed = False
        self.time_fmt = time_fmt
        self.measurement = measurement
        self.error = error
        self.GP_model = None        

    def copy(self):
        '''
        没写完，怎么复制字符串是个问题，GP_model这样的对象是不是没法复制是个问题
        ……是不是不要这个方法比较好
        '''
        copy = LightCurve(self.sorted, self.time_fmt, self.measurement, self.error)
        pass

    def __len__(self):
        return len(self.data)

    def add_column(self, column, name):
        '''
        Parameters
        ----------
        column : a 1-D numpy array
        name : 
        '''
        index = self.data.index.copy()
        self.data = pd.concat([self.data, pd.Series(column, index=index, name=name)], axis=1)

    def fold(self, period, normalize_phase=False, normalize_section=[-0.5,0.5],
            time='time'):
        '''
        Parameters
        ----------
        period : 
        normalize : if True, the phase will be set within [-0.5, 0.5]
        time : name of time column

        note that the light curve should be sorted by time before fold
        '''
        if self.folded==True:
            pass
        phase = self.data[time].copy()
        phase = (phase - phase.iloc[0]) % period
        self.period = period
        self.phase_span = period
        if normalize_phase==True:
            self.phase_span = normalize_section[1] - normalize_section[0]
            phase = (phase / max(phase)) * self.phase_span + normalize_section[0]
        self.add_column(phase, name='phase')
        self.data = self.data.sort_values(by='phase', ignore_index=True)
        self.folded = True

    def supersmoother_fit(self):
        if self.smoothed == True:
            pass
        x_label = 'phase' if self.folded else self.time_fmt
        from supersmoother import SuperSmoother
        model = SuperSmoother(period=self.period) #使用周期参数可以避免报重复点错

        x = np.around(self.data[x_label], decimals=4)
        y = np.around(self.data[self.measurement], decimals=4)
        err = np.around(self.data[self.error], decimals=4)
        temp_data = pd.DataFrame(np.column_stack((x, y, err)), 
                                    columns=[x_label, self.measurement, self.error])
        temp_data.drop_duplicates(subset=x_label, inplace=True)    
        x_dup_removed = np.array(temp_data[x_label])
        y_dup_removed = np.array(temp_data[self.measurement])
        err_dup_removed = np.array(temp_data[self.error])

        model.fit(x_dup_removed, y_dup_removed, err_dup_removed)
        smoothed_measurement = model.predict(x)
        self.add_column(smoothed_measurement, name='smoothed_'+self.measurement)
        self.amplitude = np.max(smoothed_measurement) - np.min(smoothed_measurement)
        self.smoothed = True

    def GP_smooth(self):
        if self.GP_model == None:
            self.fit_GP_model()
        x = np.array(self.data['phase'])
        original_y = np.array(self.data[self.measurement])
        pred, pred_var = self.GP_model.predict(original_y, x, return_var=True)
        self.add_column(pred, name='smoothed_'+self.measurement)


    def clean(self):
        if self.folded == False:
            print('warnning:clean operation should be after folding')
        if self.smoothed == False:
            self.supersmoother_fit()
            # self.GP_smooth()
        mean_err = np.mean(self.data[self.error])
        bogus_list=[]
        for index,measurement,smoothed_measure,err in zip(self.data.index, self.data['smoothed_'+self.measurement],
                                                self.data[self.measurement], self.data[self.error]):
            if abs(measurement-smoothed_measure) >= 3*err or err >= 2*mean_err:
                bogus_list.append(index)
        self.data = self.data.drop(index=bogus_list)
        pass
    
    def show(self):
        x_label = 'phase' if self.folded else self.data.columns[0]
        x = np.array(self.data[x_label])
        y = np.array(self.data[self.measurement])
        err = np.array(self.data[self.error])
        plt.errorbar(x,y,err, fmt='o',ms=4, mfc='r', elinewidth=1, capsize=2)
        plt.show()

    def show_smoothed(self):
        self.measurement = 'smoothed_'+self.measurement
        self.show()
        self.measurement = self.measurement[9:]
   
    def fit_GP_model_one_attempt(self, kernel=None):
        if self.folded==False:
            raise SequenceError('GP model only for phase folded curve')
        if self.GP_model != None:
            return self.GP_model
        import george
        from george import kernels
        from scipy.optimize import minimize
        x = self.data['phase']
        y = self.data[self.measurement]
        err = self.data[self.error]
        if kernel==None:
            kernel = np.var(y) * kernels.Matern52Kernel(metric=0.1, ndim=1)
        gp = george.GP(kernel)
        gp.compute(x, err)

        # print("Initial ln-likelihood: {0:.2f}".format(gp.log_likelihood(y)))
        def neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.log_likelihood(y)
        def grad_neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(y)
    
        result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like, method="L-BFGS-B")
        # print(result)
        gp.set_parameter_vector(result.x)
        # print("\nFinal ln-likelihood: {0:.2f}".format(gp.log_likelihood(y)))
        self.GP_model = gp
        return gp

    def fit_GP_model(self, kernel=None):
        '''
        try to fix error when fit GP model
        don't know why but it works
        '''
        import random
        success = False
        count_limit = 200
        count = 0
        while not success:
            try:
                self.fit_GP_model_one_attempt(kernel=kernel)
                success = True
            except:
                self.data = self.data.drop(random.sample(list(self.data.index),1))
                count += 1
                if count >= count_limit:
                    raise SequenceError('GP_fitting failed')

    def compute_time_delta(self):
        '''
        note that this operation can't be applied both before and after fold,
        otherwise the time_delta will be meaningless
        '''
        if self.folded==True:
            phase_delta = np.concatenate(([0],np.diff(self.data['phase'])))
            self.add_column(phase_delta, name='phase_delta')
        else:
            time_delta = np.concatenate(([0],np.diff(self.data[self.time_fmt])))
            self.add_column(time_delta, name='time_delta')
    
    def generate_GP_simulation(self, x, phase_shift_ratio=0, scale_std=True):
        import random
        if self.GP_model == None:
            self.fit_GP_model()
        simu_lc = LightCurve(self.sorted, self.time_fmt, self.measurement, self.error)
        simu_lc.folded = True
        simu_lc.time_span = self.time_span
        simu_lc.phase_span = self.phase_span
        simu_lc.period = self.period

        original_y = np.array(self.data[self.measurement])
        pred, pred_var = self.GP_model.predict(original_y, x, return_var=True)
        pred_std = np.sqrt(pred_var)

        if scale_std == True:
            mean_error =  np.mean(self.data[self.error])
            mean_pred_std = np.mean(pred_std)
            scaled_std = pred_std * mean_error / mean_pred_std
            simu_std = scaled_std
        else:
            simu_std = pred_std

        noise = [random.normalvariate(mu, sigma)%(3*sigma)for mu, sigma in zip(pred, simu_std)]
        # 对3倍sigma取模为了防止生成极端值，少量取模应该不影响数据
        noised_pred = pred + np.array(noise)
        phase = (np.array(x) + phase_shift_ratio*self.phase_span)%(self.phase_span)
        time = phase.copy()
        
        simu_lc.data = pd.DataFrame(np.column_stack((time, noised_pred, simu_std, phase)), 
                                    columns=['mjd', 'mag', 'err', 'phase'])
        simu_lc.data = simu_lc.data.sort_values(by='phase', ignore_index=True)
        return simu_lc

    def to_image(self, figsize=(4,4),dpi=32):
        from PIL import Image
        fig = plt.figure(figsize=figsize,dpi=dpi)
        fig.patch.set_facecolor('black')
        x_label = 'phase' if self.folded else self.data.columns[0]
        x = np.array(self.data[x_label])
        y = np.array(self.data[self.measurement])
        plt.scatter(x,y, s=2,c='w')
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0)
        plt.margins(0,0)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        buf = np.roll(buf, 3, axis=2)
        image = Image.frombytes("RGBA", (w, h), buf.tostring())
        image = image.convert('L')
        plt.close('all')
        return image
    
    def to_uncertainty_map(self, figsize=128, scale_std=True):
        if self.GP_model == None:
            self.fit_GP_model()
        x = np.linspace(0, self.phase_span, 128)
        original_y = np.array(self.data[self.measurement])
        pred, pred_var = self.GP_model.predict(original_y, x, return_var=True)
        pred_std = np.sqrt(pred_var)

        if scale_std == True:
            mean_error =  np.mean(self.data[self.error])
            mean_pred_std = np.mean(pred_std)
            scaled_std = pred_std * mean_error / mean_pred_std
            simu_std = scaled_std
        else:
            simu_std = pred_std
                
        phase = x
        time = phase.copy()
        tmp_data = pd.DataFrame(np.column_stack((time, pred, simu_std, phase)), 
                                columns=['mjd', 'mag', 'err', 'phase'])
        
        from PIL import Image
        fig = plt.figure(figsize=(4,4),dpi=32)
        fig.patch.set_facecolor('black')
        x_label = 'phase' if self.folded else self.data.columns[0]
        x = np.array(self.data[x_label])
        y = np.array(self.data[self.measurement])
        plt.scatter(x,y, s=2,c='b')
        plt.scatter(tmp_data['phase'], tmp_data['mag'],s=10,c='r')
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0)
        plt.margins(0,0)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        buf = np.roll(buf, 3, axis=2)
        image = Image.frombytes("RGBA", (w, h), buf.tostring())
        plt.close('all')

        pred_line_image = np.array(image)[:,:,0]
        line_coordinate = np.array(range(0,128))
        column_centres = []
        for column in range(0,128):
            column_weight = pred_line_image[:,column]
            centre = np.matmul(line_coordinate, column_weight) / np.sum(column_weight)
            column_centres.append(centre)
        column_centres = np.array(column_centres)

        pred_max_mag = np.max(tmp_data['mag'])
        pred_min_mag = np.min(tmp_data['mag'])
        pred_max_pixel = np.argmax(tmp_data['mag'])
        pred_min_pixel = np.argmin(tmp_data['mag'])

        mag_pixel_ratio = (pred_max_mag - pred_min_mag) / abs(column_centres[pred_max_pixel] - column_centres[pred_min_pixel])
        
        from scipy import stats
        uncertainty_map = np.zeros((128,128))
        normlization_ratio = 255/stats.norm.pdf(0)
        for column in range(0,128):
            for line in range(0,128):
                distance = (line - column_centres[column]) * mag_pixel_ratio
                #这里可能会报出除零警告，导致无效值，如果对学习结果有影响，应想办法剔除
                uncertainty_map[line][column] = stats.norm.pdf(distance/tmp_data['err'][column]) * normlization_ratio
        return uncertainty_map


    
class CRTS_VS_LightCurve(LightCurve):

    def read_CRTS_dat(self, file_path, id):
        df = pd.read_csv(file_path, delim_whitespace=True,
                        names=['mjd','mag','err'])
        self.data = df
        self.id = id
        self.time_span = df['mjd'].iloc[-1] - df['mjd'].iloc[0]

    def fold(self, period, normalize_phase=False, normalize_section=[-0.5,0.5]):
        return super().fold(period, normalize_phase, normalize_section, 'mjd')

    def supersmoother_fit(self):
        if self.folded==False:
            raise SequenceError('light curve should be folded before smoothing')
        return super().supersmoother_fit()



    
    



