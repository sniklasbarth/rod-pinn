function x = toDevice(x)
%TODEVICE moves x to the GPU if MATLAB can use one (NVIDIA GPU with CUDA and
% Parallel Computing Toolbox), otherwise x stays on the CPU, e.g. on macOS.
% gather() returns CPU arrays unchanged, so results are read back the same
% way in both cases.
persistent useGPU
if isempty(useGPU)
    useGPU = canUseGPU();
    if useGPU
        d = gpuDevice;
        fprintf('PINN evaluation on GPU (%s)\n', d.Name);
    else
        fprintf('PINN evaluation on CPU (no CUDA GPU available to MATLAB)\n');
    end
end
if useGPU
    x = gpuArray(x);
end
end
