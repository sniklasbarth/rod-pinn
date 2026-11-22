function [P_Mat,t] = smooth_input_complex(P_max,TMAX,t_sample,base_freq,validation_flag)
    % P_max = 100000;
    % TMAX = 10;
    t = 0:t_sample:TMAX;
    % freq1 = 0.2; % orig: 0.7 (first actuator)
    % freq2 = 0.6;    % orig: 0.6 (second actuator)
    freq1 = base_freq;
    freq2 = base_freq;
    
    omega_ = 2*pi*freq1;
    omega2_ = -2*pi*freq2;
    
    phi_ = 2*pi/3;
    AMP = 0.7;
    tau = 0.05;

    if validation_flag
        pfunc = @(t,phi,omega) AMP*P_max*(1-exp(-4*tau.*t.^2)).*(0.3*sin(omega.*t+phi)-(0.25*sin(4*omega.*t+phi/2)+0.2*sin(8*omega.*t+phi))+0.8)
    else
        pfunc = @(t,phi,omega) AMP*P_max*(1-exp(-4*tau.*t.^2)).*((0.3*sin(omega.*t+phi)+0.5)+(0.2*sin(10*omega.*t+phi)+(0.2*sin(5*omega.*t+phi))+0.3))
    end
    p21 = pfunc(t,0,omega_);
    p31 = pfunc(t,phi_,omega_);
    p11 = pfunc(t,phi_*2,omega_);
    
    p22 = 0*pfunc(t,0, omega2_);
    p32 = 0*pfunc(t,phi_,omega2_);
    p12 = 0*pfunc(t,phi_*2,omega2_);
    
    P_Mat = [p11' p21' p31' p12' p22' p32'];
    
    % close all
    % figure
    % % plot(t,[p1',p2',p3'])
    % plot(t,P_Mat)
end
