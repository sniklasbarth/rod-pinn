function data = load_measurement(csv_file)
%LOAD_MEASUREMENT Loads a test bench measurement (data/exp2_measurement.csv).
% Columns: time in s, gauge pressures p1-p3 in Pa, end-effector position
% x, y, z in m, end-effector orientation as unit quaternion qw, qx, qy, qz,
% base reaction forces fx, fy, fz in N and moments mx, my, mz in Nm.

T = readmatrix(csv_file);
data.time = T(:,1);
data.p_is = T(:,2:4);
data.x_is = T(:,5:7);
data.h_is = T(:,8:11);
data.force_torque = T(:,12:17);
end
