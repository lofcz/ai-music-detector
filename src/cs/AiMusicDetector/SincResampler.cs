using System;

namespace AiMusicDetector;

/// <summary>
/// Sinc interpolation resampler matching torchaudio.functional.resample with sinc_interp_hann.
/// 
/// This is a direct port of torchaudio's resampling algorithm from:
/// https://github.com/pytorch/audio/blob/main/torchaudio/functional/functional.py
/// 
/// Default parameters match torchaudio:
/// - resampling_method='sinc_interp_hann'
/// - lowpass_filter_width=6
/// - rolloff=0.99
/// </summary>
public static class SincResampler
{
    /// <summary>
    /// Resample audio using sinc interpolation with Hann window (matches torchaudio exactly).
    /// </summary>
    public static float[] Resample(
        float[] input,
        int origFreq,
        int newFreq,
        int lowpassFilterWidth = 6,
        double rolloff = 0.99)
    {
        if (origFreq == newFreq)
            return (float[])input.Clone();

        // Calculate GCD to find rational resampling ratio (same as torchaudio)
        int gcd = GCD(origFreq, newFreq);
        int origFreqReduced = origFreq / gcd;
        int newFreqReduced = newFreq / gcd;

        // Build the resampling kernel (matches _get_sinc_resample_kernel)
        var (kernels, width) = GetSincResampleKernel(
            origFreqReduced, newFreqReduced, lowpassFilterWidth, rolloff);

        // Apply the kernel (matches _apply_sinc_resample_kernel)
        return ApplySincResampleKernel(input, origFreqReduced, newFreqReduced, kernels, width);
    }

    /// <summary>
    /// Port of torchaudio's _get_sinc_resample_kernel function.
    /// Returns: (kernels[newFreq, filterLen], width)
    /// </summary>
    private static (double[,] kernels, int width) GetSincResampleKernel(
        int origFreq,
        int newFreq,
        int lowpassFilterWidth,
        double rolloff)
    {
        // base_freq = min(orig_freq, new_freq) * rolloff
        double baseFreq = Math.Min(origFreq, newFreq) * rolloff;

        // width = ceil(lowpass_filter_width * orig_freq / base_freq)
        int width = (int)Math.Ceiling(lowpassFilterWidth * origFreq / baseFreq);

        // Filter length: 2*width + orig_freq
        int filterLen = 2 * width + origFreq;

        // Build the kernel bank: kernels[newFreq, filterLen]
        double[,] kernels = new double[newFreq, filterLen];

        // Scale factor
        double scale = baseFreq / origFreq;

        // For each output phase (0 to newFreq-1)
        for (int phase = 0; phase < newFreq; phase++)
        {
            // torchaudio: t = torch.arange(0, -new_freq, -1)[:, None, None] / new_freq + idx
            // where idx = torch.arange(-width, width + orig_freq) / orig_freq
            // So for phase p: t_p = -p/new_freq + idx
            double phaseOffset = -(double)phase / newFreq;

            for (int i = 0; i < filterLen; i++)
            {
                // idx[i] = (i - width) / orig_freq
                double idx = (double)(i - width) / origFreq;

                // t = (phase_offset + idx) * base_freq
                double t = (phaseOffset + idx) * baseFreq;

                // Clamp t to [-lowpass_filter_width, lowpass_filter_width]
                t = Math.Max(-lowpassFilterWidth, Math.Min(lowpassFilterWidth, t));

                // Hann window: cos(t * pi / lowpass_filter_width / 2)^2
                double window = Math.Cos(t * Math.PI / lowpassFilterWidth / 2);
                window = window * window;

                // Sinc: sin(t * pi) / (t * pi), with t=0 -> 1
                double tPi = t * Math.PI;
                double sinc;
                if (Math.Abs(tPi) < 1e-10)
                {
                    sinc = 1.0;
                }
                else
                {
                    sinc = Math.Sin(tPi) / tPi;
                }

                // Kernel value
                kernels[phase, i] = sinc * window * scale;
            }
        }

        return (kernels, width);
    }

    /// <summary>
    /// Port of torchaudio's _apply_sinc_resample_kernel function.
    /// Uses convolution with stride=origFreq.
    /// </summary>
    private static float[] ApplySincResampleKernel(
        float[] input,
        int origFreq,
        int newFreq,
        double[,] kernels,
        int width)
    {
        int inputLen = input.Length;
        int filterLen = kernels.GetLength(1);

        // Pad input: (width, width + orig_freq)
        int padLeft = width;
        int padRight = width + origFreq;
        int paddedLen = inputLen + padLeft + padRight;
        double[] padded = new double[paddedLen];
        for (int i = 0; i < inputLen; i++)
        {
            padded[padLeft + i] = input[i];
        }
        // Zero padding on both sides (already initialized to 0)

        // Apply convolution with stride=origFreq
        // Output of conv1d: for each starting position (stride origFreq), apply each of newFreq filters
        // Then reshape: transpose and flatten
        
        // Number of conv output positions
        int numPositions = (paddedLen - filterLen) / origFreq + 1;

        // Conv output shape: [numPositions, newFreq]
        // After transpose: [newFreq, numPositions]
        // After reshape: [newFreq * numPositions]
        double[] convOut = new double[numPositions * newFreq];

        for (int pos = 0; pos < numPositions; pos++)
        {
            int startIdx = pos * origFreq;
            for (int phase = 0; phase < newFreq; phase++)
            {
                double sum = 0;
                for (int k = 0; k < filterLen; k++)
                {
                    sum += padded[startIdx + k] * kernels[phase, k];
                }
                // Transpose: output[phase, pos] -> linear index = phase * numPositions + pos
                // But torchaudio does transpose(1,2) then reshape, so it's interleaved:
                // After transpose(1,2) the shape is [batch, numPositions, newFreq]
                // After reshape it's [batch, numPositions * newFreq]
                // So the order is: pos0_phase0, pos0_phase1, ..., pos0_phaseN, pos1_phase0, ...
                convOut[pos * newFreq + phase] = sum;
            }
        }

        // Target length: ceil(new_freq * length / orig_freq)
        int targetLength = (int)Math.Ceiling((double)newFreq * inputLen / origFreq);
        targetLength = Math.Min(targetLength, convOut.Length);

        float[] output = new float[targetLength];
        for (int i = 0; i < targetLength; i++)
        {
            output[i] = (float)convOut[i];
        }

        return output;
    }

    /// <summary>
    /// Resample stereo audio (2 channels) and average to mono, matching torchaudio's order:
    /// resample first, then average channels (audio.mean(dim=0)).
    /// </summary>
    public static float[] ResampleStereoToMono(
        float[] leftChannel,
        float[] rightChannel,
        int inputSampleRate,
        int outputSampleRate,
        int lowpassFilterWidth = 6,
        double rolloff = 0.99)
    {
        // Resample each channel first (like torchaudio)
        var leftResampled = Resample(leftChannel, inputSampleRate, outputSampleRate, lowpassFilterWidth, rolloff);
        var rightResampled = Resample(rightChannel, inputSampleRate, outputSampleRate, lowpassFilterWidth, rolloff);

        // Average to mono (matches audio.mean(dim=0))
        int length = Math.Min(leftResampled.Length, rightResampled.Length);
        float[] mono = new float[length];
        for (int i = 0; i < length; i++)
        {
            mono[i] = (leftResampled[i] + rightResampled[i]) * 0.5f;
        }

        return mono;
    }

    /// <summary>
    /// Binary GCD (Stein's algorithm) - uses only subtraction and bit shifts, no division.
    /// </summary>
    private static int GCD(int a, int b)
    {
        if (a == 0) return b;
        if (b == 0) return a;

        // Find common factors of 2
        int shift = 0;
        while (((a | b) & 1) == 0)
        {
            a >>= 1;
            b >>= 1;
            shift++;
        }

        // Remove remaining factors of 2 from a
        while ((a & 1) == 0)
            a >>= 1;

        do
        {
            // Remove factors of 2 from b
            while ((b & 1) == 0)
                b >>= 1;

            // Ensure a <= b
            if (a > b)
            {
                int temp = a;
                a = b;
                b = temp;
            }

            b -= a;
        } while (b != 0);

        return a << shift;
    }
}
