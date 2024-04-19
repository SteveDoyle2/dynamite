# dynamite
structural dynamics tools

## TimeSeries

## DFT (Discrete Fourier Transform)

## PSD (Power Spectral Density)

$X: DFT$

$X^*$: conjugate transpose of $DFT$

$$ PSD = \frac{X X^*} {\Delta f} $$

## Miles Equation

$$ zeta = \frac{1}{2Q} $$

$$  \rho_i = f_i / f_n $$

$$ x_{GRMS}(f_n, \zeta) = \sqrt{ \frac{\pi f_n}{4 \zeta} PSD(f_n) } $$

Note that the base input amplitude is taken at the natural frequency.  The derivation assumes that this amplitude is constant across the entire frequency domain, from zero to infinity.

Miles equation should only be used if the power spectral density amplitude is flat within one octave on either side of the natural frequency. 

[source](http://www.vibrationdata.com/tutorials2/vrs.pdf)

## VRS (Vibration Response Spectra)

$$ zeta = \frac{1}{2Q} $$


$$ x_{GRMS}(f_n, \zeta) = \sqrt{  \displaystyle {\Sigma_{i=1}^N} \frac{1 + (2 \zeta \rho_i)^2}{(1-\rho_i^2)^2 + (2 \zeta \rho_i)^2} PSD(f_i) } $$

$ \zeta$: damping ratio

$ f_n$: natural frequency (Hz)

$ Q$: amplification factor (10 = 5% damping)

[source](http://www.vibrationdata.com/tutorials2/vrs.pdf)

## SRS (Shock Response Spectra)
