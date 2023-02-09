import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
def gait_identify(peaks):
    counter = 0
    peaks_temp = []
    start = []
    end = []
    for i in range(len(peaks)-1):
        distance = peaks[i+1] - peaks[i]
        if distance > 120:
            if counter >= 1:
                peaks_temp = np.append(peaks_temp,peaks[(i-counter):i])
                end = np.append(end,peaks[i])
                start = np.append(start, peaks[i-counter])
                counter = 0
            elif counter < 1:
                counter = 0
                continue
        elif distance < 120:
                counter += 1
                if i == len(peaks)-1:
                    peaks_temp = np.append(peaks_temp,peaks[(i-counter):i])
    peaks_temp = [int(peak) for peak in peaks_temp]
    return peaks_temp, start, end


def extract_gait_params(gait_cycle_mat,freq):
    '''
    Outputs gait cycle time, swing time, and swing percent from gait cycles
    '''
    # Determine the gait cycle time (GCT)



    gct =( gait_cycle_mat[:, 4] - gait_cycle_mat[:, 0])/freq
    step_time =(gait_cycle_mat[:, 2]- gait_cycle_mat[:, 0])/freq
    adj_step_time = (gait_cycle_mat[:, 4]- gait_cycle_mat[:, 2])/freq
    step_time_percent= ((gait_cycle_mat[:, 2]- gait_cycle_mat[:, 0])/freq/gct)  *100
    stance_percent = ((gait_cycle_mat[:, 2]- gait_cycle_mat[:, 1])/freq/gct)*100
    stance_time =(gait_cycle_mat[:, 2]- gait_cycle_mat[:, 1])/freq
    adj_stance_time = (gait_cycle_mat[:, 4]- gait_cycle_mat[:, 3])/freq

    # get duty factor
    duty_factor = (gait_cycle_mat[:, 3]- gait_cycle_mat[:,0])/freq
    duty_factor_percent = (gait_cycle_mat[:, 3]- gait_cycle_mat[:,0])/freq/gct * 100;

    # get duty factor
    
    cadence = 60/(step_time)
    # Calculate swing times and percentages
    swing_times = (gait_cycle_mat[:, 4] - gait_cycle_mat[:, 3])/freq
    adj_swing_times = (gait_cycle_mat[:, 2] - gait_cycle_mat[:, 1])/freq
    swing_percents = (gait_cycle_mat[:, 4] - gait_cycle_mat[:, 3])/freq / gct * 100

    # Calculate double support (DS) with initial (IDS) and terminal (TDS)
    ids = (gait_cycle_mat[:, 1] - gait_cycle_mat[:, 0])/freq / gct * 100   
    tds = (gait_cycle_mat[:, 3] - gait_cycle_mat[:, 2])/freq / gct * 100


    ds = ids + tds
    
    

    # Calculate limp
    limp = np.abs(ids-tds)



    return gct, swing_times, adj_swing_times, swing_percents, ds, limp,gait_cycle_mat,step_time, cadence,duty_factor,step_time_percent,duty_factor_percent, adj_step_time,stance_percent, stance_time,adj_stance_time
def extract_peak_values(gait_cycle_height):
    toe_off_val = gait_cycle_height[:,3]
    heel_contact_val = gait_cycle_height[:,0]
    adj_toe_off_val = gait_cycle_height[:,1]
    adj_heel_cont_val = gait_cycle_height[:,2]
    result = heel_contact_val/adj_heel_cont_val
    return result
    
def mean_std(data):
    return np.mean(data), np.std(data)


def get_gait_param_means(leg, method, start_time, gct, swt, swp, ds, limp,asym,ang_vel, cadence, step_time,duty_factor):
    '''
    Gets mean and standard deviations of gait params for specific leg
    '''

    length = len(gct)
    leg_list = [leg for i in range(length)]
    
    
    method_list = [method for i in range(length)]

    
    param_df = pd.DataFrame(
        {'Leg': leg_list, 'Method': method_list, 'Cycle Start': start_time, 'GCT': gct/1e3, 'Swing Time': swt/1e3, 'Swing Percent': swp, 'Double Support': ds, 'Limp': limp, 'Asymmetry': asym,
                 'Heel Contact Angular Velocity':ang_vel,'Cadence':cadence,'Step Time':step_time/1e3,'Duty Factor': duty_factor})

    gct_mean, gct_std = mean_std(gct/1e3)
    swt_mean, swt_std = mean_std(swt/1e3)
    swp_mean, swp_std = mean_std(swp)
    ds_mean, ds_std = mean_std(ds)
    limp_mean, limp_std = mean_std(limp)
    asym_mean, asym_std = mean_std(asym)
    ang_vel_mean, ang_vel_std = mean_std(ang_vel)
    cadence_mean, cadence_std = mean_std(cadence)
    step_time_mean, step_time_std = mean_std(step_time/1e3)
    duty_factor_mean, duty_factor_std = mean_std(duty_factor)

    mean_std_list = [leg, method, gct_mean, gct_std, swp_mean, swp_std, ds_mean, ds_std, limp_mean, limp_std,asym_mean,asym_std,ang_vel_mean, ang_vel_std,cadence_mean, cadence_std,step_time_mean, step_time_std,duty_factor_mean, duty_factor_std]
    mean_std_labels = ['Leg', 'Method', 'Gait cycle time avg. [s]', 'Gait cycle time std. [s]',
                       'Swing time avg. [% GCT]', 'Swing time std. [% GCT]',
                       'Double support avg. [% GCT]', 'Double support std. [% GCT]',
                       'Limp avg. [% GCT]', 'Limp std. [% GCT]','Asym avg.', 'Asym std.','Heel Contact Angular Velocity avg.', 'Heel Contact Angular Velocity std.', 'Cadence avg.','Cadence std.','Step Time avg.','Step Time std.','Duty Factor avg.','Duty Factor std.']
    mean_std_df = pd.DataFrame([mean_std_list], columns=mean_std_labels)
    

    return param_df, mean_std_df
def get_asymmetry(swt_left, swt_right):
    '''
    Calculates limb asymmetry
    '''
    # Calculating using averages of swt (paper method)

    swtl_mean, swtl_std = mean_std(swt_left/1e3)
    swtr_mean, swtr_std = mean_std(swt_right/1e3)

    if swtl_mean > swtr_mean:
        lswt_mean = swtl_mean
        sswt_mean = swtr_mean
    else:
        lswt_mean = swtr_mean
        sswt_mean = swtl_mean

    asym_paper = abs(np.log(sswt_mean/lswt_mean))


    return asym_paper

def get_duty_factor_asymmetry(inj,healthy):
    inj_mean,_ = mean_std(inj/1000)
    healthy_mean,_ = mean_std(healthy/1000)
    asym = np.abs((inj_mean-healthy_mean)/healthy_mean)
    return asym

def find_walking(peaks,thresh):
    m = [[peaks[0]]]

    for i,x in enumerate(peaks[1:]):
        if x - peaks[i] < thresh:
            m[-1].append(x)
        else:
            m.append([x])
    lens = [len(i) for i in m]
    walk_loc = lens.index(max(lens))     
    return m[walk_loc]

def gait_cycle(peak_dict,side,freq):
    gct = []
    if side == 'left':
        for val,i in enumerate(peak_dict['left hc'][0:-1]):
            if any(np.logical_and(peak_dict['right to']>i,peak_dict['right to']<peak_dict['left hc'][val+1])) and any(np.logical_and(peak_dict['right hc']>i,peak_dict['right hc']<peak_dict['left hc'][val+1])) and any(np.logical_and(peak_dict['left to']>i,peak_dict['left to']<peak_dict['left hc'][val+1])) and any(peak_dict['left hc']>i):
            

                adj_to = peak_dict['right to'][np.where(np.logical_and(peak_dict['right to']>peak_dict['left hc'][val],peak_dict['right to']<peak_dict['left hc'][val+1])==True)[0][0]]
                adj_hc = peak_dict['right hc'][np.where(np.logical_and(peak_dict['right hc']>peak_dict['left hc'][val],peak_dict['right hc']<peak_dict['left hc'][val+1])==True)[0][0]]
                tar_to = peak_dict['left to'][np.where(np.logical_and(peak_dict['left to']>peak_dict['left hc'][val],peak_dict['left to']<peak_dict['left hc'][val+1])==True)[0][0]]
                #next_hc = peak_dict['right hc'][peak_dict['right hc']>i][0]
                next_hc = peak_dict['left hc'][val+1]

  
                if  adj_to < adj_hc and adj_hc<= tar_to and tar_to <= next_hc and np.abs(i-next_hc) < 3000 and adj_to-i < 200 and tar_to-adj_hc <200:
                    temp = [i, adj_to, adj_hc, tar_to, next_hc]
                    gct.append(temp)
    if side == 'right':
        for val,i in enumerate(peak_dict['right hc'][0:-1]):
            if any(np.logical_and(peak_dict['right to']>i,peak_dict['right to']<peak_dict['right hc'][val+1])) and any(np.logical_and(peak_dict['left hc']>i,peak_dict['left hc']<peak_dict['right hc'][val+1])) and any(np.logical_and(peak_dict['left to']>i,peak_dict['left to']<peak_dict['right hc'][val+1])) and any(peak_dict['right hc']>i):

                
                adj_to = peak_dict['left to'][np.where(np.logical_and(peak_dict['left to']>peak_dict['right hc'][val],peak_dict['left to']<peak_dict['right hc'][val+1])==True)[0][0]]
                adj_hc = peak_dict['left hc'][np.where(np.logical_and(peak_dict['left hc']>peak_dict['right hc'][val],peak_dict['left hc']<peak_dict['right hc'][val+1])==True)[0][0]]
                tar_to = peak_dict['right to'][np.where(np.logical_and(peak_dict['right to']>peak_dict['right hc'][val],peak_dict['right to']<peak_dict['right hc'][val+1])==True)[0][0]]
                next_hc = peak_dict['right hc'][val+1]
                if  adj_to < adj_hc and adj_hc<= tar_to and tar_to <= next_hc and np.abs(i-next_hc) < 3000 and adj_to-i < 200 and tar_to-adj_hc <200:
                    temp = [i, adj_to, adj_hc, tar_to, next_hc]
                    gct.append(temp)
    gct = np.array(gct)
    return gct

    


def heel_contact_toe_off(shank_z, analysis_time, output_fs):
    min_peak_dist = output_fs*0.4
    # min_height = 0.174

    
    # Finds the top peaks and the minimum peaks
    swing_peak,swing_height = find_peaks(shank_z, height=[1.5,10], distance=min_peak_dist)

    swing_peak,swing_height = find_peaks(shank_z, height=[np.mean(swing_height['peak_heights'])-np.std(swing_height['peak_heights']),10], distance=min_peak_dist)

    
    heel_contact = []  # Initial (after peak)
    toe_off = []
    neg_peaks_temp, min_peak_height= find_peaks(-1*shank_z,height=0)

    
    min_peak_height = -min_peak_height['peak_heights']
    peaks_temp = swing_peak
    heel_contact = [list(filter(lambda j: j >= i, neg_peaks_temp))[0] for i in peaks_temp if any(np.array(neg_peaks_temp)>=i)]

    heel_contact = sorted(set(heel_contact))
    toe_off = [neg_peaks_temp[np.where(min_peak_height == np.min(min_peak_height[neg_peaks_temp<peaks_temp[i]][-2:]))][0] if len(min_peak_height[neg_peaks_temp<peaks_temp[i]]) > 1 else neg_peaks_temp[neg_peaks_temp<peaks_temp[i]][-1] for i in range(len(peaks_temp)) if any(neg_peaks_temp<peaks_temp[i])]
    #heel_contact= analysis_time[np.array(heel_contact)]

    #toe_off = analysis_time[np.array(toe_off)]
    return heel_contact, toe_off
def find_closest(x,L2):
    l2_diffs = [abs(x - y) for y in L2]
    return l2_diffs.index(min(l2_diffs)),min(l2_diffs)

def calculate_RMSE(imu, kp):
    return np.sqrt(((imu - kp)**2).mean())
def Sync_Left_Right(L_sens,R_sens):
    

     
     r_peaks,_ = find_peaks(R_sens.values.flatten(),height=1)
     l_peaks,_ = find_peaks(L_sens.values.flatten(),height=1)


     res = [(i, find_closest(x,l_peaks)[0],find_closest(x,l_peaks)[1]) for i, x in enumerate(r_peaks)]
     hops = [i[0:2] for i in res if i[2] <10]

     
     start_end = [[r_peaks[hops[i][0]],r_peaks[hops[i+1][0]]] for i in range(len(hops)-1) if hops[i+1][0]-hops[i][0] > 15]

     if start_end == []:
         R_sens = R_sens
         L_sens = L_sens
     else:
         R_sens = R_sens.iloc[start_end[0][0]+10:start_end[0][1]-10]
         L_sens = L_sens.iloc[start_end[0][0]+10:start_end[0][1]-10]

     return L_sens,R_sens
 
def find_hops(L_sens,R_sens):
        

         r_peaks,_ = find_peaks(R_sens.values.flatten(),height=1)
         l_peaks,_ = find_peaks(L_sens.values.flatten(),height=1)
         # print(r_peaks)


         res = [(i, find_closest(x,l_peaks)[0],find_closest(x,l_peaks)[1]) for i, x in enumerate(r_peaks)]

         hops = np.array([i[0:2] for i in res if i[2] <10])


         if len(hops) != 0:
             r_hops = r_peaks[hops[:,0]]
             l_hops = l_peaks[hops[:,1]]
         else:
            r_hops = []
            l_hops = []
         
         return l_hops,r_hops
     
def step_counter(x):
    counter = 1

    list_comp = []
    for val, i in enumerate(x):
        if i <2300:
            counter = counter+1
            if val == len(x)-1:
                list_comp = np.append(list_comp,counter)
        elif i > 2300:
            list_comp = np.append(list_comp,counter)
            counter = 1
    return list_comp
     
def get_cycle_asymmetry(swt_left, swt_right):
     '''
     Calculates limb asymmetry
     '''
     # Calculating using averages of swt (paper method)


     # Calculate using individual swing times

     lswt = []
     sswt = []
     asym = []

     for i in range(len(swt_left)):
         if swt_left[i] > swt_right[i]:
             lswt.append(swt_left[i])
             sswt.append(swt_right[i])
         else:
             lswt.append(swt_right[i])
             sswt.append(swt_left[i])
             

         asym.append(abs(np.log(sswt[i]/lswt[i])))

     return asym
def Sensor_Gait_Param_Analysis(lshank_z_df, rshank_z_df,analysis_time,dominant_limb, freq=100): 
    '''
    Takes left and right shank Z-gyroscope data and outputs gait parameters
    '''
    



    l_hops,r_hops  = find_hops(lshank_z_df,rshank_z_df)


    lshank_z = lshank_z_df.values
    rshank_z = rshank_z_df.values
    lshank_z_temp = lshank_z_df
    rshank_z_temp = rshank_z_df

    min_peak_dist = freq*0.4
    lheel_contact_time, ltoe_contact_time= heel_contact_toe_off(lshank_z.flatten(), analysis_time, output_fs=freq)
    rheel_contact_time, rtoe_contact_time= heel_contact_toe_off(rshank_z.flatten(), analysis_time, output_fs=freq)


    peak_dict = {'left hc':lheel_contact_time,'right hc': rheel_contact_time,'left to': ltoe_contact_time,'right to':rtoe_contact_time}

    
    gait_cycle_mat_left = gait_cycle(peak_dict,'left',freq)

    l_step_per_direction = step_counter(np.diff(gait_cycle_mat_left[:,0]))


    gait_cycle_mat_right = gait_cycle(peak_dict,'right',freq)
    
    r_step_per_direction = step_counter(np.diff(gait_cycle_mat_right[:,0]))
    


    r_gct, r_swing_times, r_adj_swing_times, r_swing_percents, r_ds, r_limp,r_gait_cycle_mat,r_step_time, r_cadence,r_duty_factor,r_st_percent, r_df_percent,r_adj_step_time,r_stance_percent, r_stance_time,r_adj_stance_time = extract_gait_params(gait_cycle_mat_right,freq)
    l_gct, l_swing_times, l_adj_swing_times, l_swing_percents, l_ds, l_limp,l_gait_cycle_mat,l_step_time, l_cadence,l_duty_factor,l_st_percent, l_df_percent,l_adj_step_time,l_stance_percent, l_stance_time,l_adj_stance_time = extract_gait_params(gait_cycle_mat_left,freq)

    if dominant_limb == 'Left':
                        dom_swt =l_swing_times
                        rec_swt = l_adj_swing_times     
                        dom_st = l_step_time
                        rec_st =l_adj_step_time
                        dom_df = l_duty_factor
                        rec_df = r_duty_factor
                        dom_gct = l_gct
                        dom_swp = l_swing_percents
                        dom_limp = l_limp
                        dom_cadence = l_cadence
                        dom_ds = l_ds
                        main_st = l_st_percent
                        main_df = l_df_percent
                        main_stance = l_stance_percent
                        dom_stance = l_stance_time
                        rec_stance = l_adj_stance_time
                        dom_step_count = np.mean(l_step_per_direction)
    if dominant_limb == 'Right':
                        dom_swt =r_swing_times
                        rec_swt = r_adj_swing_times      
                        dom_st = r_step_time
                        rec_st = r_adj_step_time
                        dom_df = r_duty_factor
                        rec_df = l_duty_factor
                        dom_gct = r_gct
                        dom_swp = r_swing_percents
                        dom_limp = r_limp
                        dom_cadence = r_cadence
                        dom_ds = r_ds
                        main_st = r_st_percent
                        main_df = r_df_percent
                        main_stance = r_stance_percent
                        dom_stance = r_stance_time
                        rec_stance = r_adj_stance_time
                        dom_step_count = np.mean(r_step_per_direction)
    swt_asym = get_cycle_asymmetry(dom_swt, rec_swt)
    st_asym = get_cycle_asymmetry(dom_st, rec_st)
    df_asym = get_duty_factor_asymmetry(dom_df, rec_df)
    stance_asym =   get_cycle_asymmetry(dom_stance, rec_stance)            
    temp_dict = {}

    temp_dict['swing time asym'] = swt_asym
    temp_dict['Double Support'] = dom_ds
    temp_dict['stance time asym'] = stance_asym
    temp_dict['step time asym'] = st_asym
    temp_dict['duty factor asym'] = df_asym
    temp_dict['Gait Cycle Time (s)'] = dom_gct
    temp_dict['Double Support (% GCT)'] = dom_ds
    temp_dict['Limp (% GCT)'] = dom_limp
    temp_dict['Cadence (steps/min)']  = dom_cadence
    temp_dict['Duty Factor (% GCT)']  = main_df
    temp_dict['Step Time'] = main_st
    temp_dict['Swing Percent'] = dom_swp
    temp_dict['Stance Time'] = main_stance
    temp_dict['Adjacent Step Time'] = (rec_st/dom_gct)*100
    temp_dict['Adjacent Swing Percent'] =( rec_swt/dom_gct)*100
    temp_dict['Adjacent Stance Time'] = (rec_stance/dom_gct)*100
    temp_dict['Cycles Per Lab']=dom_step_count





    return temp_dict


