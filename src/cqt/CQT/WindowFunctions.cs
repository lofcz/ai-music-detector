using System;

namespace CQT;

/// <summary>
/// Window functions for signal processing.
/// </summary>
public static class WindowFunctions
{
    /// <summary>
    /// Hann (Hanning) window with periodic/fftbins mode.
    /// </summary>
    public static double[] Hann(int length)
    {
        double[] window = new double[length];
        for (int i = 0; i < length; i++)
        {
            window[i] = 0.5 * (1.0 - Math.Cos(2.0 * Math.PI * i / length));
        }
        return window;
    }

    /// <summary>
    /// Hann (Hanning) window with symmetric mode.
    /// </summary>
    public static double[] HannSymmetric(int length)
    {
        double[] window = new double[length];
        for (int i = 0; i < length; i++)
        {
            window[i] = 0.5 * (1.0 - Math.Cos(2.0 * Math.PI * i / (length - 1)));
        }
        return window;
    }

    /// <summary>
    /// Hamming window.
    /// </summary>
    public static double[] Hamming(int length)
    {
        double[] window = new double[length];
        for (int i = 0; i < length; i++)
        {
            window[i] = 0.54 - 0.46 * Math.Cos(2.0 * Math.PI * i / (length - 1));
        }
        return window;
    }

    /// <summary>
    /// Rectangular (no window).
    /// </summary>
    public static double[] Rectangular(int length)
    {
        double[] window = new double[length];
        for (int i = 0; i < length; i++)
            window[i] = 1.0;
        return window;
    }
}