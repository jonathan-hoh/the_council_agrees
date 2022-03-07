#!/usr/bin/env python
# coding: utf-8

# In[9]:


# Run parameter file to initialize packages, constants, functions, and timespacing
get_ipython().run_line_magic('run', 'C:/Users/hohjo/Documents/Doctoral_Work/Jarrahi_Work/lock_in_software_sim/Golden_Codes/Params_Gold.ipynb')


# In[10]:


###############################################################################################
#                                                                                             #
#                                        Lock-in Simulation (100 Mhz Tone)                    #
#                                                                                             #
###############################################################################################

#***********************************#
############################        #
# User Inputs:             #        #
#    Channel and Run time  #        #
############################        #
chan = 100                          #
run_time = 20                      #
#***********************************#
        
    # Define how many seconds of data should be simulated and how many accumulations are required

num_accum = int(run_time * (1/accum_time)) # number of total accumulations in simulation time

    # create file name for saving data in csv if desired

file_name = '%d_sec_lockin_sim_gold_unchopped.csv'%(run_time)   

    #Make an empty array to be filled with accumulations

raw_lock_accums_i = np.zeros(num_accum)
raw_lock_accums_q = np.zeros(num_accum)
raw_intnsty = np.zeros(num_accum)

filt_accums_i = np.zeros(num_accum)
filt_accums_q = np.zeros(num_accum)
final_intsty_out = np.zeros(num_accum)
sig_max = 0

#***********************************#
#                                   #
#           Creating Time           #        
#                                   #
#***********************************#

for i in range(num_accum):
    print('we are on accumulation number %d out of %d'%(i+1, num_accum))

        ##### Create an array with the times of the FFT frames #####

    frame_times = np.linspace(i * frame_time, i * frame_time + (accum_frames-1) * frame_time, (accum_frames)  )

        ############################################################################################################    
        # Create an array of times that will be used to create the "pieces" of the wave                            #  
        # Populate time array with lengths to be used later                                                        #
        # This is an absolutely crazy vectorization of a previous loop I had, but it runs 100 times faster. Sorry. #
        ############################################################################################################

    timespace = np.linspace(np.linspace(frame_times[0], frame_times[1], FFT_length), 
                            np.linspace(frame_times[accum_frames-2], frame_times[accum_frames-1], FFT_length),
                            num = accum_frames-2)


#***********************************#
#                                   #
#            Signals                #
#   (Creation and Timestreaming)    #
#                                   #
#***********************************#

        # tone of interest
    signal = real_wave(1, source_freq, timespace)
    
            # Lets make some noiiissseeee
    
     ##### Now add some white noise #####
    w_noise = np.random.normal(0, .12, signal.shape)
    
        ##### And some pink noise #####
    beta = 1                                            # the exponent for pink noise
    samples = signal.shape                               # number of samples to generate (mimic the dimensions of the signal)
    y = cn.powerlaw_psd_gaussian(beta, samples)
    
    noisy_sig = signal + w_noise + y   
        
        # Now put the unchopped noise signal through PFB

    spectra = (np.fft.fft(noisy_sig, n = FFT_length))*(2/FFT_length) 
        
        # Once again, take transpose of FFT matrix to get channel timestreams
    
    t_streams = np.transpose(spectra)
    t_streams_i = np.real(t_streams)
    t_streams_q = np.imag(t_streams)
    
    t_stream_mag = magnitude(t_streams_i, t_streams_q)
   

    #########################################
    #    Mixing Channel Timestreams Down    #
    #########################################   

        # Create time array to control internally generated wave 

    timespace2 = np.linspace(i * frame_time, i * frame_time + (accum_frames-1) * frame_time, (accum_frames)-2)

        # Create generated signal inside FPGA at square wave frequency 

    sq_i =(sig.square(2 * np.pi * square_freq * timespace2)) 
    sq_q =(sig.square(2 * np.pi * square_freq * timespace2 + (np.pi/2)))    
            
        # Mix together timestreams and chops
    
    downmix_i, downmix_q = c_mult(t_stream_mag[20], 0, sq_i, sq_q)
    
    downmix_intsty = intensify(downmix_i, downmix_q)
    
        # For sanity checks, lets pocket the unfiltered data (JFD!) #
    raw_accum_i = np.sum(downmix_i)
    raw_accum_q = np.sum(downmix_q)
    raw_intnsty_vec = intensify(downmix_i, downmix_q)
    
    raw_lock_accums_i[i] = raw_accum_i
    raw_lock_accums_q[i] = raw_accum_q
    raw_intnsty[i] = np.sum(raw_intnsty_vec)
   
    ##### Filtering stage #####
   
    (filt_mix_i, filt_mix_q) = lowpass_i_q(downmix_i, downmix_q)

        #### Accumulate I and Q separately (JFD!) ####
    filt_accum_i = np.sum(filt_mix_i)
    filt_accum_q = np.sum(filt_mix_q)
    filt_accums_i[i] = filt_accum_i
    filt_accums_q[i] = filt_accum_q
    
    #### Take filtered intensity ####
     
    filt_intsty = intensify(filt_mix_i, filt_mix_q)
    intsty_accum = np.sum(filt_intsty)
    final_intsty_out[i] = intsty_accum


# In[14]:


plt.plot(filt_accums_i)


# In[15]:


plt.plot(filt_intsty)


# In[ ]:




