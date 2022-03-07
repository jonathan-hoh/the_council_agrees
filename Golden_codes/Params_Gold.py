#%%

#***********************************#
#                                   #
#             Packages              #        
#                                   #
#***********************************#
import math
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import allantools as AT
from scipy import signal as sig
from scipy import stats as stat
import line_profiler
import ipywidgets as wdg
import dsp_py as dsp
import colorednoise as cn

#%%
# Yes, there are simpler ways to do this, but this is the selection function which allows you to choose the speed of hardware and signals #


#***********************************#
#             Standard              #
#             Constants             #        
#                                   #
#***********************************#
def std_params():
 global adc_clk 
 adc_clk = 512 * 10 ** 6 # adc clock rate

 global T_adc
 T_adc = 1/(adc_clk) # adc period

 global fpga_clk 
 fpga_clk = 256 * 10 ** 6 # fpga fabric clock rate

 global FFT_length
 FFT_length = 1024 # Length of FFT used in spectrometer
 
 global accum_len
 accum_len = 2 ** 23 # number of clock cycles of accumulation in every dump of data
 #accum_len = 2 ** 13
 
 global accum_time
 accum_time = ((adc_clk)/FFT_length)/accum_len # amount of time one accumulation cycle integrates (seconds)



 # Frequency Parameters #

 global source_freq
 source_freq = 10 * 10 ** 6

 global square_freq 
 square_freq = 10 * 10 ** 3

 global fs_fft 
 fs_fft = adc_clk/FFT_length  # Define a new frequency to represent the rate at which the FFT is produced

 global nyq_fft 
 nyq_fft = fs_fft/2           # Nyquist frequency of FFT production rate

#***********************************#
#            Downclocked            #
#             Constants             #        
#                                   #
#***********************************#
def slow_params():
 global adc_clk 
 adc_clk = 512 * 10 ** 3 # adc clock rate

 global T_adc
 T_adc = 1/(adc_clk) # adc period

 global fpga_clk 
 fpga_clk = 256 * 10 ** 3 # fpga fabric clock rate

 global FFT_length
 FFT_length = 128 # Length of FFT used in spectrometer
 
 global accum_len
 accum_len = 2 ** 13 # number of clock cycles of accumulation in every dump of data
 
 global accum_time
 accum_time = ((adc_clk)/FFT_length)/accum_len # amount of time one accumulation cycle integrates (seconds)



 # Frequency Parameters #

 global source_freq
 source_freq = 10 * 10 ** 3

 global square_freq 
 square_freq = 100

 global fs_fft 
 fs_fft = adc_clk/FFT_length  # Define a new frequency to represent the rate at which the FFT is produced

 global nyq_fft 
 nyq_fft = fs_fft/2           # Nyquist frequency of FFT production rate
 

#***********************************#
#             Fullspeed             #
#             Constants             #        
#                                   #
#***********************************#
def full_params():
 global adc_clk 
 adc_clk = 512 * 10 ** 6 # adc clock rate

 global T_adc
 T_adc = 1/(adc_clk) # adc period

 global fpga_clk 
 fpga_clk = 256 * 10 ** 6 # fpga fabric clock rate

 global FFT_length
 FFT_length = 1024 # Length of FFT used in spectrometer
 
 global accum_len
 accum_len = 2 ** 23 # number of clock cycles of accumulation in every dump of data
 
 global accum_time
 accum_time = ((adc_clk)/FFT_length)/accum_len # amount of time one accumulation cycle integrates (seconds)



 # Frequency Parameters #

 global source_freq
 source_freq = 100 * 10 ** 6

 global square_freq 
 square_freq = 100 * 10 ** 3

 global fs_fft 
 fs_fft = adc_clk/FFT_length  # Define a new frequency to represent the rate at which the FFT is produced

 global nyq_fft 
 nyq_fft = fs_fft/2           # Nyquist frequency of FFT production rate


#%%

std_params()


#%%


#***********************************#
#                                   #
#            Functions              #       
#                                   #
#***********************************#


################################################
# Real wave generation function:               #
# Input: amp, frequency and amount of time     #
# Output : Array of sampled points             #
################################################

def real_wave(amp, freq, time, phase=0):
    omega = 2 * np.pi * freq    
    wave = amp*np.cos(omega * time + phase)
    return wave

################################################
# Wave generation function:                    #
# Input: amp, frequency and amount of time     #
# Output : I and Q arrays for wave             #
################################################

def cool_wave(amp, freq, time, phase=0):
    omega = 2 * np.pi * freq    
    wave = amp * np.exp(1j * omega * time + phase)
    i = np.real(wave)
    q = np.imag(wave)
    return i,q
    
#########################################################
#    Complex multipler function:                        #
#    Input: I_1, Q_1, I_2, Q_2                          #
#    Output: Mixed wave in quadrature (I_mix, Q_mix)    #
#########################################################

def c_mult(I_1, Q_1, I_2, Q_2): 
    I_mix = I_1 * I_2 - Q_1 * Q_2 
    Q_mix = I_1 * Q_2 + Q_1 * I_2 
    return I_mix, Q_mix

#########################################################
#    Real Multiplier (std. multiply)                    #
#    Input: signal_1, signal_2                          #
#    Output: Mixed(vector multiplied) waveform result   #
#########################################################

def real_mix(wave_1, wave_2):
    mix_wave = wave_1 * wave_2
    return mix_wave


#%%

#########################################################
#    Other Functions:                                   #
#    Magnitude and Intensity
#    Single-ended FFT                                   #
#    FFT for I,Q signal                                 #
#    Noise Inclusion                                    # 
#    Chopping                                           #
#    File Saving                                        #
#                                                       # 
#########################################################
def magnitude(sig_1, sig_2): #takes the magnitude of a complex value/vector
    mag = np.sqrt(np.square(sig_1) + np.square(sig_2))
    return mag

def intensify(sig_1, sig_2):
    intense = np.square(sig_1) + np.square(sig_2)
    return intense

def sq_trans(sig):
    sq_wave = sig/np.abs(sig)
    return sq_wave

# Function to covert axis from sample space to seconds
def samp_2_sec(samp_vec):
    sec_vec = samp_vec * T_adc
    return sec_vec

def samp_2_sec_2(samp_vec):
    sec_vec = samp_vec * T_adc
    return sec_vec*1024

# =============================================================================
# Try out somethign different than a Butterworth filter
# Also throw out first 
# =============================================================================
def lowpass_i_q(data_i, data_q, order = 7, cutoff = nyq_fft * 0.01, fsamp = fs_fft):
    B, A = sig.butter(order, cutoff, output='ba', fs = fsamp)
    filt_i = sig.filtfilt(B,A,data_i)
    filt_q = sig.filtfilt(B,A,data_q)
    return(filt_i, filt_q)

# Cassie says to try simplest averaging filter

#%%


#########################################################
#    Other Functions:                                   #
#    Single-ended FFT                                   #
#    FFT for I,Q signal                                 #
#    Noise Inclusion                                    # 
#    Chopping                                           #
#    File Saving                                        #
#                                                       # 
#########################################################

def fft(signal):
    spectrum = np.fft.rfft(signal, n = FFT_length)
    intensity = np.real(spectrum)**2 + np.imag(spectrum)**2
    return spectrum, intensity

def fft_IQ(signal_i, signal_q):
    spectrum = np.fft.fft(signal_i + 1j*(signal_q), n = FFT_length)
    intensity = np.real(spectrum)**2 + np.imag(spectrum)**2
    return spectrum, intensity

def noisify(signal_i, signal_q, POWER):
    w_noise = np.random.normal(0, POWER, signal_i.shape)
    noisy_sig_i = signal_i * w_noise
    w_noise = np.random.normal(0, POWER, signal_q.shape)
    noisy_sig_q = signal_q * w_noise
    return noisy_sig_i, noisy_sig_q

def GET_TO_DA_CHOPPAH(signal, timespace):
    sq_wave = 0.5 * (sig.square(2 * np.pi * square_freq * timespace) + 1)
    chopped_wave = signal * sq_wave
    return chopped_wave

def complex_choppin(sig_i, sig_q, clk_chop, timespace):
    sq_wave_i = 0.5 * (sig.square(2 * np.pi * clk_chop * timespace) + 1) 
    sq_wave_q = 0.5 * (sig.square(2 * np.pi * clk_chop * timespace + (np.pi/4)) + 1)
    chop_sig_i, chop_sig_q = c_mult(sig_i, sig_q, sq_wave_i, sq_wave_q)
    return chop_sig_i, chop_sig_q
    
def save_data(file_name, data):
    file = open(file_name, 'w')
    writer = csv.writer(file)
    writer.writerow(data)
 


#%%


################################################
#           Allan Variance Functions           #
#                                              #
#                                              #
################################################

######################################################
# Basic Allan Variance for single timestream; 
# 30 data point resolution; 
# Returns plot, white noise line, error and tau vector 
######################################################

def allan_var(timestream, run_time, res=30):
    rate = 1/(accum_time)
    tau = np.logspace(0, run_time/5, res)
    (tau2, adevs, adev_err, n) = AT.oadev(timestream, rate, data_type="freq", taus=tau)
    avars = np.square(adevs)
    white_line = (avars[0]*(tau2**-1))
    return avars, white_line, adev_err, tau2   

################################################
# Allan Variance Plotter:                      #
# Input: Timestream of data and bin number     #
# Output : Plot of allan variance              #
################################################


def allan_plot(data1, chan):
    num_sec = len(data1)/(1/accum_time)
    #tau = np.logspace(0, 1, 30)
    tau = np.logspace(0, np.log10(num_sec/5), 30)
    rate = 1/accum_time # 1/16 seconds of integration per sample
    
    # now take data and push through allan deviation function 
    (tau2, adevs, adev_err, n) = AT.oadev(data1, rate, data_type="freq", taus=tau)
    avars = np.square(adevs) # square allan dev to get allan var
    # Make white noise t^-1 line
    white_line = (avars[0]*(tau2**-1))  
    
    
    # Plot ur shit bro                   
    plot = plt.loglog(tau2, avars) 
    plt.loglog(tau2,white_line)   
   # plt.errorbar(tau2, avars, yerr = (avars[::]/np.sqrt((num_sec/tau2[::]))), ecolor='g')
    plt.title('Allan Variance for Lock-in Spectrometer (Bin %d)'%(chan))
    plt.xlabel('Integration Times (s)')
    plt.ylabel('Power^2 (arbitrary(?) units)')
    plt.show()
    
################################################
# Allan Variance Comparitor:                   #
# Input: 2 Timestreams of Data and Bin Number  #
# Output : Plot comparing allan variances      #
################################################


def allan_plot_compare(data1, data2, chan):
    
    # First, figure out how long the timestreams are (assuming a 23 bit accumulator)
    num_sec = len(data1)/(1/accum_time)
    #tau = np.logspace(0, 1, 30)
    tau = np.logspace(0, np.log10(num_sec/5), 30)
    rate = 16 # around 1/16 seconds of integration per sample
    
    # Now take data and push through allan deviation function 
    (tau2, adevs, adev_err, n) = AT.oadev(data1, rate, data_type="freq", taus=tau)
    # Square allan dev to get allan var
    avars = np.square(adevs)
    # Make white noise t^-1 line
    white_line = (avars[0]*(tau2**-1))  
    
    # now for second set of data
    (tau3, adevs2, adev_err2, n2) = AT.oadev(data2, rate, data_type="freq", taus=tau)
    avars2 = np.square(adevs2) # square allan dev to get allan var
    white_line2 = (avars2[0]*(tau2**-1)) 
    
    # Plot ur shit bro
    fig, axs = plt.subplots(2, sharex=True)
    axs[0].loglog(tau2, avars, label ="hoh_spec allan variance") 
    axs[0].loglog(tau2, avars2, label ="lock-in allan variance")
    axs[0].loglog(tau2,white_line)  
    axs[0].loglog(tau2,white_line2)
    axs[0].errorbar(tau2, avars, yerr = 2*(avars[::]/np.sqrt((num_sec/tau2[::]))), ecolor='g')
    axs[0].errorbar(tau2, avars2, yerr = 2*(avars2[::]/np.sqrt((num_sec/tau2[::]))), ecolor='g')
   
    ratio = avars/avars2
    axs[1].loglog(tau2, ratio)
    
    plt.title('Allan Variance Comparison for Lock-in Spectrometer (Bin %d)'%(chan))
    plt.xlabel('Integration Times (s)')
    plt.ylabel('Power^2 (arbitrary(?) units)')
    plt.show()
    
    return(np.average(ratio))
    


#%%


#***********************************#
#                                   #
#            Functions              #        
#       (Under Construction)        #
#***********************************#

    
#         function for reading in saved data sets. Input to function must be a string with a .csv format
# def read_data(file):
#    with open(file) as csv_file:
#    csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
#    for row in csv_reader:
#     data = next(csv_reader)   


#%%

###############################################################################################
#                                                                                             #
#                                  Begin Main Simulation Script                               #
#                                                                                             #
###############################################################################################


#%%

#***********************************#
#                                   #
#           Make FFT Frames         #        
#                                   #
#***********************************#

    ##### for a 1024 pt FFT, one FFT frame will take 1024/adc_clk (seconds) ##### 


    # frame time == Time for 1 FFT to populate
frame_time = FFT_length / (adc_clk)

    # frame_freq == how many FFT frames are created in one second
frame_freq = adc_clk/FFT_length

    #The number of fft frames to be created equals the run time of the test over the amount of time it takes to FFT a frame                        

    # accum_frames == number of FFT frames created over one accumulation
accum_frames = int(accum_time/frame_time)

    # Now create array which contains frequency span of FFT in order to have correctly scaled FFT plots
timestep = 1/adc_clk
    
    # note this value comes from the linspace used for the time array. length/number of samples
    # this array will be used to properly map the x-axis in Fourier space plots
fft_freq = np.fft.fftfreq(FFT_length, d=timestep)
rfft_freq = np.fft.rfftfreq(FFT_length, d=timestep)


#%%

accum_frames


#%%

print('Parameter Initialization Successful')
print('Good job nerd')

