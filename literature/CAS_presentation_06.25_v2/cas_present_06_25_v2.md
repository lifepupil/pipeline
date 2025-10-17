# Phase Amplitude Coupling (PAC) in the Cognitive-Affective Stroop (CAS) task

## Task description and analysis objective
5 stimulus types: color words and words characterized as "neutral", "negative", "alcohol related"; response is to identify color of word by button press.
The word is on screen for entire duration of the 2.25 second interval and then is replaced by a different word. About 35 trials of each stimulus, which are pseudo-randomly arranged.

This elicits an initial P3A response to word replacement accompanied by continuous multi-frequency activity even after a button push response approximately at the middle of the stimulus duration.

A major feature of the activity, observed in most subjects, is PAC,
when the amplitude of a higher frequency oscillation has an oscillation pattern correlated with a lower frequency oscillation. In this case the amplitude of the higher frequency oscillation is called the modulated signal and the lower frequency signal is called the modulating signal.

Theta-Beta coupling (3-7 Hz -- 12-24 Hz) is a well documented brain phenomenon and was obvious from the first in the examination of CAS results. The ultimate measurement is to quantify the phase relation between the beta _amplitude_ and the theta _signal_, both with regard to the consistency of the phase relations over time and the amount of phase difference between the oscillations. One quantity of interest which is not discussed here is the coefficient of variation of the amplitude but visual inspection suggests it is not insignificant.

The first 5 pages illustrate the data used to determine the PAC; the following 2 pages illustrate the results of the PAC.

## Data processing
### Raw data
The raw data without separation between trials is bandpass filtered by use of the continuous wavelet transform using the complex Morlet wavelet at 20 scales ranging from 35 Hz to 3 Hz with 5 bands per octave. The output is complex valued and provides the bandpass filtered signal and its amplitude and phase. The Morlet wavelet is a complex sinusoid in an exp(-t^2) window spanning about 4 wavelengths of the frequency. The computation uses the Fourier and inverse Fourier transforms for speed. For the plots in the collection, means of stimulus types with incorrect or artifactual trials removed were used. All the methods exhibited here can be used on individual trials.

### Analysis procedures

#### Analysis of global (multi_electrode) phase synchrony
The global phase synchrony values for an individual sample point are determined by the following equation
$ PS = abs(sum_s(Z(s)))/(sum_s(abs(Z(s)))) $
where s ranges across electrodes. The angle of the complex value of the right side is the mean of the angles at the individual electrodes and not particularly meaningful in this context.
Results of this analysis are illustrated on pages 3 through 7.

#### Analysis of amplitude time series
In order to estimate the appropriate modulating frequency for a given amplitude time series, the intervals between successive extrema of the time series, which represent half wavelength periods, are determined and their mean after removal of outliers is used to determine the modulating frequency, which in this case is constrained to one of the 20 frequencies used in the first phase of data processing. For the purposes of compatibility in this analysis, the modulating frequencies of the 16 Hz Beta amplitude time series in all 31 electrodes was determined and 5.33 Hz was chosen as the modulating frequency for the entire group to accommodate the range of observed modulating frequencies which range from 4 to 6 Hz.

Since only 5 frequency scales per octave were used for filtering, it would be possible to refilter the entire raw data series of any electrode with a custom scale for the modulating frequency to give a more precise estimate of PAC for the modulated frequency. However, the use of custom scales would introduce further complications so this procedure was not implemented in this analysis.

#### PAC synchrony determination
The modulating frequency time series is the result of the wavelet transform, and phase is directly encoded in the complex values of the time series, here indicated as W(t). The phase of the modulated amplitude time series is determined by determining the local extrema of the series and linearly transforming the ascending values to the interval from -\pi to 0 and the descending values to the interval 0 to \pi. This constrains the frequency of the modulated signal whose phase can be determine depending on the sampling rate. For the purposes of calculating the phase synchrony as described below, only the phase of the modulated amplitude time series in complex form is required; a new time series Z(t) is created to provide this:
$ Z(t) = exp(i * phase(t)) $
The methods for attributing amplitude to this time series are not discussed here, as they are not part of the synchrony determination.

The complex valued phase synchrony of two complex time series Z(t) and W(t) is
$ PS = sum_t(Z(t) * conj(W(t)))/(sum_t(abs(Z(t) * conj(W(t)))) $
The absolute value of PS is what is generally called the phase synchrony or, when the absolute values of the elements of the time series Z(t) and W(t) are all 1, the phase_locking index, and the angle is the phase difference between the series. The absolute value of the output is always between 0 and 1, regardless of the amplitudes of the inputs. In the phase synchrony calculation used to determine PAC the phase_locking index is calculated. Interchanging W(t) and Z(t) changes only the sign of angle considered on the range -\pi to \pi and does not change any of the plotted reprersentations.

Results of this analysis are illustrated on pages 6 and 7.


## Illustrations provided

### Page 1: Nine electrode plot of the congruent condition:

Solid lines are the beta (16-Hz) amplitude; dashed lines are the theta signal (4 Hz). The anterior electrodes are blue, the central are green, and the posterior are red. The color coding can be assumed to be similar across all plots unless otherwise indicated.

### Page 2: Central electrode plot of the congruent and incongruent cases:
Line coding as above, but in this case the theta signal is 5 1/3 Hz because more detailed analysis of beta amplitude suggested that frequency. The temporal/case pattern of phase agreement/discrepancy between the 3 electrodes is worth noting. This will be explicitly indicated in some subsequent plots.

### Page 3: Central electrode plot of the congruent and incongruent cases with global phase synchrony curve:
The plot on page 3 has been overlaid with the global phase synchrony of the entire set of 31 electrodes at each sample point. The scale for this quantity is on the right side of the plot. The global phase synchrony is high when the beta signal phases are the same at each electrode and low when they have considerable variation. While the periodicity of the global phase synchrony and the amplitude curves are not identical, dips in the global phase synchrony seem to coincide with dips in the amplitude. It should be noted that when the global phase synchrony is high the phase distributions at different times are not necessarily close when a desynchronization/resynchronization has happened between the two times. In intervals of uninterrupted high values it is safe to assume the distributions are similar to each other.

### Page 4: Central electrode plot of the congruent and incongruent cases with global phase synchrony curve:
Now the solid line are the beta _signal_ with the global phase synchrony overlaid. Even with only 3 electrodes it is easy to see the overlay of signal trajectories in the high global phase synchrony sequences and the separation of trajectories in the desynchronization/resynchronization phases. It would be possible to calculate the phase synchronies for selected pairs of electrodes to determine more about regional characteristics.

### Page 5: Central electrode plot of the congruent and incongruent cases with global phase synchrony curve and amplitude curves:
This is the previous plot with the amplitude curves overlaid. It might be better with thinner lines. It just combines information on previous plots.

### Pages 6 & 7: Central electrode plots of the phase synchrony between the theta signal and beta amplitude
I suggest you examine these plots in conjunction with the top plot on page 2, which has the theta signals and beta amplitudes superimposed; set the magnification to 200% so you can easily study the trajectories of the solid and dashed lines of the same color over the first second particularly.

In these plots the two phase synchrony measures from the calculation of phase synchrony of the theta signal and the the beta amplitude with imputed phase are superimposed, with the blue line the phase synchrony value and the green line the cosine of the phase difference between the two time series. The use of the cosine eliminates the discontinuities in the phase trajectories and is easy to interpret with cos(0) (no angular difference) = 1 and cos(pi) = cos(-pi) = -1. The cosine curve has been linearly transformed to the interval from 0 to 1 to make the plot more readable. The differences between the two pages, aside from the fact that the x-axes are different in length (my bad) is that on page 6 the synchrony values are calculated over a 1/8 second long moving window (2 beta wavelengths -- 1/2 theta wavelength) advanced by 1/32 second for each iteration, while on page 7 the window is 1/4 second long with the same advance. The phase synchrony values are limited in length and position by the phase imputation method which operates on time segments spanning of one half wavelength of the modulated signal. The red dotted line is the global phase synchrony seen in black in previous pages.
What is striking in these plots is the switch between periods in which the angle between the series is 0 (extrema coincide) and periods in which the angle between the series is pi (180 degrees, extrema are opposites). Also noteworthy is the difference between FZ and the other electrodes in terms of the timing and duration of the opposite extrema intervals.
The effect of the longer time window is to smooth the phase synchrony measures and to lower the value of the phase synchrony, but it has little effect on the angular distance except for FZ where the trajectories for the second half-second of the plot are very different for the two time windows. Clearly a change in angular distance over time should be reflected in a decrease in synchrony, but in the short period around 1/2 second in CZ there seems to be a discrepancy in the phase synchrony trajectories between the short window and long window plots.

Note that the phase synchrony lines begin at different times for the different electrodes. This can be understood to some degree by examining the top panel of page 3, where the algorithm for identifying ascending and descending intervals is having trouble in the period between the first and relatively small global phase desynchronization about 75 milliseconds into the record and the second and much larger synchronization about 300 milliseconds into the record.

## Conclusion
This is really a beginning focused on how to directly characterize PAC as an "instantaneous" measure, and it depends on how long your instant is.
