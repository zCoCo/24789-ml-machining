function [fftdata]=fftPlot(data,a,b,Hzz)
engaged=data(a:b);
L = length(engaged);
Ts =1/ Hzz; % 0.000005;                                     % Sampling Interval (sec)
Fs = 1/Ts;                                              % Sampling Frequency
Fn = Fs/2;                                              % Nyquist Frequency
engagedc = engaged- mean(engaged);                     % Subtract Mean (‘0 Hz’) Component
FTv = fft(engagedc)/L;                                  % Fourier Transform
Fv = linspace(0, 1, fix(L/2)+1)*Fn;                     % Frequency Vector (Hz)
Iv = 1:length(Fv); % Index Vector
fftdata=abs(FTv(Iv))*2;
figure()
plot(Fv, abs(FTv(Iv))*2);
grid
xlabel('Frequency (Hz)')
ylabel('Amplitude')
hold on
end