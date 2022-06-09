# MEGA-COMBINED SCRIPT OF SIMULATION

# necessary imports
import math
import csv
import numpy as np
from scipy import signal as sig #take out

# prompting user for channel and runtime inputs
chan = input("input channel:")
chan = int(chan)
run_time = input("input runtime:")
run_time = int(run_time)


#***********************************#
#             Standard              #
#             Constants             #
#***********************************#
def std_params():

 #General Parameters
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
 source_freq = 10 * 10 ** 6 # frequency of the source wave

 global square_freq
 square_freq = 10 * 10 ** 3 # frequency of square wave

 global fs_fft
 fs_fft = adc_clk/FFT_length  # Define a new frequency to represent the rate at which the FFT is produced

 global nyq_fft
 nyq_fft = fs_fft/2           # Nyquist frequency of FFT production rate



#calling the parameters used
std_params()


#***********************************#
#             Wave                  #
#           Functions               #
#***********************************#


#Real wave generation function:
#Input: amp, frequency and amount of time
#Output : Array of sampled points
def real_wave(amp, freq, time, phase=0):
    omega = 2 * np.pi * freq
    wave = amp*np.cos(omega * time + phase)
    return wave

#Complex multipler function:
#Input: I_1, Q_1, I_2, Q_2
#Output: Mixed wave in quadrature (I_mix, Q_mix)
def c_mult(I_1, Q_1, I_2, Q_2):
    I_mix = I_1 * I_2 - Q_1 * Q_2
    Q_mix = I_1 * Q_2 + Q_1 * I_2
    return I_mix, Q_mix

#Real Multiplier (std. multiply)
#Input: signal_1, signal_2
#Output: Mixed(vector multiplied) waveform result
def real_mix(wave_1, wave_2):
    mix_wave = wave_1 * wave_2
    return mix_wave


#***********************************#
#               Misc.               #
#             Functions             #
#***********************************#

# takes the magnitude of a complex value/vector
def magnitude(sig_1, sig_2):
    mag = np.sqrt(np.square(sig_1) + np.square(sig_2))
    return mag

# taking the intensity complex value/vector
def intensify(sig_1, sig_2):
    intense = np.square(sig_1) + np.square(sig_2)
    return intense

#CHANGE FUNCTION!!
# creates the output of i and  q going through a lowpass filter
def lowpass_i_q(data_i, data_q, order = 7, cutoff = nyq_fft * 0.01, fsamp = fs_fft):
    B, A = sig.butter(order, cutoff, output='ba', fs = fsamp)
    filt_i = sig.filtfilt(B,A,data_i)
    filt_q = sig.filtfilt(B,A,data_q)
    return(filt_i, filt_q)

# Single-ended FFT
def fft(signal):
    spectrum = np.fft.rfft(signal, n = FFT_length)
    intensity = np.real(spectrum)**2 + np.imag(spectrum)**2
    return spectrum, intensity

# Chopping
def GET_TO_DA_CHOPPAH(signal, timespace):
    sq_wave = 0.5 * (sig.square(2 * np.pi * square_freq * timespace) + 1)
    chopped_wave = signal * sq_wave
    return chopped_wave

# saving data to text file
def save_data(file_name, data):
    file = open(file_name, 'w')
    writer = csv.writer(file)
    writer.writerow(data)


###############################################################################################
#                                                                                             #
#                                  Begin Main Simulation Script                               #
#                                                                                             #
###############################################################################################


#***********************************#
#                                   #
#           Make FFT Frames         #
#                                   #
#***********************************#

##### for a 1024 pt FFT, one FFT frame will take 1024/adc_clk (seconds) #####
frame_time = FFT_length / (adc_clk) # frame time == Time for 1 FFT to populate
frame_freq = adc_clk/FFT_length # frame_freq == how many FFT frames are created in one second
accum_frames = int(accum_time/frame_time) # accum_frames == number of FFT frames created over one accumulation

# Now create array which contains frequency span of FFT in order to have correctly scaled FFT plots
timestep = 1/adc_clk #frequency of fft span

# note fft_freq comes from the linspace used for the time array. length/number of samples
# this array will be used to properly map the x-axis in Fourier space plots
fft_freq = np.fft.fftfreq(FFT_length, d=timestep) #describe variable jon
rfft_freq = np.fft.rfftfreq(FFT_length, d=timestep) #describe variable jon


accum_frames #display accum_frames value
print('Parameter Initialization Successful')

# Define how many seconds of data should be simulated and how many accumulations are required
num_accum = int(run_time * (1/accum_time)) # number of total accumulations in simulation time

# create file name for saving data in csv if desired
file_name = '%d_sec_lockin_sim_gold_bincompare.csv'%(run_time)

#Make an empty array to be filled with accumulations
raw_lock_accums_i = np.zeros((num_accum, FFT_length))
raw_lock_accums_q = np.zeros((num_accum, FFT_length))
raw_intnsty = np.zeros((num_accum, FFT_length))
n_raw_lock_accums_i = np.zeros((num_accum, FFT_length))
n_raw_lock_accums_q = np.zeros((num_accum, FFT_length))
n_raw_intnsty = np.zeros((num_accum, FFT_length))
filt_accums_i = np.zeros((num_accum, FFT_length))
filt_accums_q = np.zeros((num_accum, FFT_length))
final_intsty_out = np.zeros((num_accum, FFT_length))
n_filt_accums_i = np.zeros((num_accum, FFT_length))
n_filt_accums_q = np.zeros((num_accum, FFT_length))
n_final_intsty_out = np.zeros((num_accum, FFT_length))


#***********************************#
#                                   #
#           Creating Time           #
#                                   #
#***********************************#

# mimicing a sampled fpga
# BEGINNING OF LOOP
for i in range(num_accum):
    print('we are on accumulation number %d out of %d'%(i+1, num_accum))

    ##### Create an array with the times of the FFT frames #####

    frame_times = np.linspace(i * frame_time, i * frame_time + (accum_frames-1) * frame_time, (accum_frames)  )

    # Create an array of times that will be used to create the "pieces" of the wave
    # Populate time array with lengths to be used later
    timespace = np.linspace(np.linspace(frame_times[0], frame_times[1], FFT_length),
                            np.linspace(frame_times[accum_frames-2], frame_times[accum_frames-1], FFT_length),
                            num = accum_frames-2)


    #***********************************#
    #                                   #
    #            Signals                #
    #   (Creation and Timestreaming)    #
    #                                   #
    #***********************************#


    signal = real_wave(1, source_freq, timespace) # tone of interest
    chop_sig = GET_TO_DA_CHOPPAH(signal, timespace)

    # Adding white noise
    w_noise = np.random.normal(0, .12, chop_sig.shape)

    # Adding pink noise
    beta = 1                             # the exponent for pink noise
    samples = signal.shape               # number of samples to generate (mimic the dimensions of the signal)
    noisy_sig = chop_sig + w_noise

    # Now put the unchopped noise signal through PFB
    spectra = (np.fft.fft(chop_sig, n = FFT_length))*(2/FFT_length)
    n_spectra = (np.fft.fft(noisy_sig, n = FFT_length))*(2/FFT_length)

    # Take transpose of FFT matrix to get channel timestreams again

    (t_streams, n_t_streams) = (np.transpose(spectra), np.transpose(n_spectra))
    (t_streams_i, n_t_streams_i) = (np.real(t_streams), np.real(n_t_streams))
    (t_streams_q, n_t_streams_q) = (np.imag(t_streams), np.imag(n_t_streams))
    (t_stream_mag, n_t_stream_mag) = (magnitude(t_streams_i, t_streams_q), magnitude(n_t_streams_i, n_t_streams_q))


    #########################################
    #    Mixing Channel Timestreams Down    #
    #########################################

    # Create time array to control internally generated wave
    timespace2 = np.linspace(i * frame_time, i * frame_time + (accum_frames-1) * frame_time, (accum_frames)-2)

    # Create generated signal inside FPGA at square wave frequency
    sq_i =(sig.square(2 * np.pi * square_freq * timespace2))                # I component of generated signal (square)
    sq_q =(sig.square(2 * np.pi * square_freq * timespace2 + (np.pi/2)))    # Q component of generated signal (square)

    # Mix together timestreams and chops
    (downmix_i, downmix_q) = c_mult(t_stream_mag, 0, sq_i, sq_q)
    (n_downmix_i, n_downmix_q) = c_mult(n_t_stream_mag, 0, sq_i, sq_q)

    # Unfiltered data (JFD!)
    raw_accum_i = np.sum(downmix_i,1)
    n_raw_accum_i = np.sum(n_downmix_i,1)
    raw_accum_q = np.sum(downmix_q,1)
    n_raw_accum_q = np.sum(n_downmix_q,1)
    raw_intnsty_vec = intensify(downmix_i, downmix_q)
    n_raw_intnsty_vec = intensify(n_downmix_i, n_downmix_q)
    raw_lock_accums_i[i] = raw_accum_i
    n_raw_lock_accums_i[i] = n_raw_accum_i
    raw_lock_accums_q[i] =  raw_accum_q
    n_raw_lock_accums_q[i] = n_raw_accum_q
    raw_intnsty[i] = np.sum(raw_intnsty_vec, 1)
    n_raw_intnsty[i] = np.sum(n_raw_intnsty_vec, 1)

    # Filtering stage
    ((filt_mix_i, filt_mix_q), (n_filt_mix_i, n_filt_mix_q)) = (lowpass_i_q(downmix_i, downmix_q), lowpass_i_q(n_downmix_i, n_downmix_q))

    # Accumulate I and Q separately (JFD!)
    (filt_accum_i, n_filt_accum_i) = (np.sum(filt_mix_i, 1), np.sum(n_filt_mix_i, 1))
    (filt_accum_q, n_filt_accum_q) = (np.sum(filt_mix_q, 1), np.sum(n_filt_mix_q, 1))
    filt_accums_i[i] = filt_accum_i
    n_filt_accums_i[i] = n_filt_accum_i
    filt_accums_q[i] =filt_accum_q
    n_filt_accums_q[i] = n_filt_accum_q

    # Take filtered intensity
    (filt_intsty, n_filt_intsty) = (intensify(filt_mix_i, filt_mix_q), intensify(n_filt_mix_i, n_filt_mix_q))
    (intsty_accum, n_intsty_accum) = (np.sum(filt_intsty,1), np.sum(n_filt_intsty, 1))
    final_intsty_out[i] = intsty_accum
    n_final_intsty_out[i] = n_intsty_accum



final_channel_ints = np.transpose(final_intsty_out)
n_final_channel_ints = np.transpose(n_final_intsty_out)

#Saving data to files (csv)
save_data('%d_sec_datamine_final_intsty_out.csv'%(run_time), final_channel_ints)
save_data('%d_sec_datamine_n_final_intsty_out.csv'%(run_time), n_final_channel_ints)
save_data('%d_sec_datamine_raw_intnsty.csv'%(run_time), raw_intnsty)
save_data('%d_sec_datamine_n_raw_intnsty.csv'%(run_time), n_raw_intnsty)
save_data('%d_sec_datamine_filt_accums_i.csv'%(run_time), filt_accums_i)
save_data('%d_sec_datamine_filt_accums_q.csv'%(run_time), filt_accums_q)
save_data('%d_sec_datamine_n_filt_accums_i.csv'%(run_time), n_filt_accums_i)
save_data('%d_sec_datamine_n_filt_accums_q.csv'%(run_time), n_filt_accums_q)
