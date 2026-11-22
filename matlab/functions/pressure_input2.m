function p = pressure_input2(amp,T_ON,T_OFF)
%PRESSURE_INPUT2 Pressure trajectory of the PINN training simulation: three
% staggered tanh pulses, one per chamber, within [T_ON, T_OFF].
% p(t) returns the ABSOLUTE pressures (1x3, Pa).

p0 = 100000;

% p = @(t_)1*amp*p0*(1-exp(-4*t_.^2))'.*[1,1,1]+p0*[1,1,1]%.*([sin(omega*t_'),sin(omega*t_'+2*pi/3),sin(omega*t_'+4*pi/3)]) + p0;
AMP = @(t_) amp*p0*(1/2*(tanh(50*(t_-T_ON))-tanh(50*(t_-T_OFF))));

dt_2nd=2;
dt_on=1;
t_on1=0.5;
t_on2=0.75;
t_on3=1;
t_off1=t_on1+dt_on;
t_off2=t_on2+dt_on;
t_off3=t_on3+dt_on;
tanh_steep=5; %5

AMP1 = @(t_) (1/2*(tanh(tanh_steep*(t_-t_on1))-tanh(tanh_steep*(t_-t_off1)))) ; % + (1/2*(tanh(tanh_steep*(t_-t_on1-dt_2nd))-tanh(tanh_steep*(t_-t_off1-dt_2nd))));
AMP2 = @(t_) (1/2*(tanh(tanh_steep*(t_-t_on2))-tanh(tanh_steep*(t_-t_off2)))) ; % + (1/2*(tanh(tanh_steep*(t_-t_on2-dt_2nd))-tanh(tanh_steep*(t_-t_off2-dt_2nd))));
AMP3 = @(t_) (1/2*(tanh(tanh_steep*(t_-t_on3))-tanh(tanh_steep*(t_-t_off3)))) ; % + (1/2*(tanh(tanh_steep*(t_-t_on3-dt_2nd))-tanh(tanh_steep*(t_-t_off3-dt_2nd))));

% AMP1 = @(t_) (1/2*(tanh(tanh_steep*(t_-t_on1))-tanh(tanh_steep*(t_-t_off1)))) ; % + (1/2*(tanh(tanh_steep*(t_-t_on1-dt_2nd))-tanh(tanh_steep*(t_-t_off1-dt_2nd))));
% AMP2 = @(t_) 0.5*(1/2*(tanh(tanh_steep*(t_-t_on1))-tanh(tanh_steep*(t_-t_off1)))) ; % + (1/2*(tanh(tanh_steep*(t_-t_on2-dt_2nd))-tanh(tanh_steep*(t_-t_off2-dt_2nd))));
% AMP3 = @(t_) 0.1*(1/2*(tanh(tanh_steep*(t_-t_on1))-tanh(tanh_steep*(t_-t_off1)))) ; % + (1/2*(tanh(tanh_steep*(t_-t_on3-dt_2nd))-tanh(tanh_steep*(t_-t_off3-dt_2nd))));

% CHAMBERS = @(t_)(0.5*[sin(omega(t_)*t_'),sin(omega(t_)*t_'+2*pi/3),sin(omega(t_)*t_'+4*pi/3)]+0.5);
%CHAMBERS = @(t_)([0.5*sin(omega(t_)*t_')+0.5,0,0]);
%p = @(t_)AMP(t_)'.*CHAMBERS(t_)+p0*[1,1,1]; % environment pressure incl.
p = @(t_)AMP(t_)'*[AMP1(t_),AMP2(t_),AMP3(t_)]+p0*[1,1,1]; % environment pressure incl.
% figure, fplot(p,[T_ON T_OFF])

end