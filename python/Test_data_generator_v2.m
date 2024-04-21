% Test_data_generator_v2.m
% Mark McHenry 
% Weishan Zhang: generate 32x32 groundtruth maps and save in mat
% Oct/5/2023 -- Oct
% Estimate the receiver power  

close all; clear all; clc;
output_file='test_data_high_power_1.xlsx';

%  Determine proploss for all distances
[h1 h2 loss] = function_readITU_R_P528_3();  %  Read the ITU model tables (spreadsheets)
ITUtable.loss = loss;

%  Input parameters   [range of resonable value]
fMHz=[1200];  %   [125 MHz - 15,500 MHz]
ht_m = [15];  %   antenna height  must be within 1.5 to 20000 meters (h1)  [1.5 m to 30 m]
hr_m = [ 1000 ];     %   antenna height must be within 1000 to 20000 meters (h2)  [1000 m to 20000 m]  
desired.t = [.5];  %  Fraction of time propagation is this value or less  (.05 to .95) [.01 to .05]
desired.freq_MHz = fMHz;   desired.h1 = ht_m; desired.h2 = hr_m;

Pt_dBm=70;  % transmit power in dBm
transmitter_box_km=100; % size of box that limits transmitter location in km

NF_dB=5;  % receiver noise figure in dB
BW_MHz=10; % signal bandwidth in MHz
noise_dBm=-174+NF_dB+10*log10(BW_MHz*1e6);
number_receivers=7; % number of receivers 
Nx = 32;
Ny = 32;
map_size = [Nx, Ny];

receiver_box_km=100; %size of box that limits receiver in km

numTrain = 10000;
numTest = 1000;
Traindata = cell(1, numTrain);
Testdata = cell(1, numTest);

size = [fix(Nx),fix(Ny)];
gridlen = receiver_box_km/Nx; % grid width in km
number_measurements=10;
tic
for ii=1:numTrain
    x_TX_km = randi([1, 32])*gridlen; y_TX_km = randi([1, 32])*gridlen;
    Traindata{ii}=zeros(map_size);
    
    for jj=1:Nx
        for kk=1:Ny
            x_RX_km = jj*gridlen; y_RX_km = kk*gridlen;
%             x_RX_km = (jj-.5)*gridlen; y_RX_km = (kk-.5)*gridlen;
            desired.distance_km  = sqrt((x_RX_km - x_TX_km)^2 + (y_RX_km - y_TX_km)^2) ;   
            propLoss_dB = -function_interpolateITU_R_P528_3_v2(desired.t, desired.freq_MHz, desired.distance_km, desired.h1, desired.h2, ITUtable);
            Pr_dBm=10*log10( 10^((Pt_dBm+propLoss_dB)/10) + 10^(noise_dBm/10)); % determine signal and add noise
            Traindata{ii}(jj,kk) = Pr_dBm;
        end
    end
end

for ii=1:numTest
    x_TX_km = randi([1, 32])*gridlen; y_TX_km = randi([1, 32])*gridlen;
    Testdata{ii}=zeros(map_size);
    
    for jj=1:Nx
        for kk=1:Ny
            x_RX_km = jj*gridlen; y_RX_km = kk*gridlen;
%             x_RX_km = (jj-.5)*gridlen; y_RX_km = (kk-.5)*gridlen;            
            desired.distance_km  = sqrt((x_RX_km - x_TX_km)^2 + (y_RX_km - y_TX_km)^2) ;   
            propLoss_dB = -function_interpolateITU_R_P528_3_v2(desired.t, desired.freq_MHz, desired.distance_km, desired.h1, desired.h2, ITUtable);
            Pr_dBm=10*log10( 10^((Pt_dBm+propLoss_dB)/10) + 10^(noise_dBm/10)); % determine signal and add noise
            Testdata{ii}(jj,kk) = Pr_dBm;
        end
    end
end

s = pcolor(Testdata{1});
set(s, 'EdgeColor', 'none');
save('map_list1w.mat', 'Traindata','Testdata');
toc


