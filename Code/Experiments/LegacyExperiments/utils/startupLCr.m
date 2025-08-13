% BMK 082713
% Calls functions from LCr.m written by Hiroki
% Loads the gamma correction table for psychtoolbox
% Prepares the LCr for an experiment
%     HDMI mode
%     LED to 274,91,274
%     Disable gamma correction, etc
% Make sure to "disconnect" LCr on GUI by TI before connecting via matlab

function startupLCr()
    
    %Load the gamma correction table for psychtoolbox
    %load('gamma.mat', 'gamma');
    
    l = LCr_matlab;
    l.connect; 
    l.setDisplayMode('02'); % HDMI external video mode

	l.disableImageProcessing;
    l.getImageProcessingStatus % make sure that the image processing is all disabled!!
    
    %uncomment the lines below to set LED currents
    %l.getLEDCurrent; % LED current status
    %l.setLEDCurrent(rLED,gLED,bLED); % set LED current (0-274)
    
    l.disconnect; % make sure to disconnect LCr before using GUI provided by TI
end