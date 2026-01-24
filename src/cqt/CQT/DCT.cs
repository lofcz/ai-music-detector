using System;

namespace CQT;

/// <summary>
/// Discrete Cosine Transform.
/// </summary>
public static class DCT
{
    /// <summary>
    /// Compute DCT-II with orthonormal normalization using FFT.
    /// </summary>
    public static double[] ComputeType2(double[] input)
    {
        int N = input.Length;
        switch (N)
        {
            case 0:
                return [];
            case 1:
                return [input[0]];
        }

        if (N >= 16 && (N & (N - 1)) == 0)
        {
            return ComputeType2FFT(input);
        }
            
        return ComputeType2Direct(input);
    }

    /// <summary>
    /// Direct O(N²) DCT-II computation.
    /// </summary>
    public static double[] ComputeType2Direct(double[] input)
    {
        int N = input.Length;
        double[] output = new double[N];

        double scale0 = Math.Sqrt(1.0 / N);
        double scaleK = Math.Sqrt(2.0 / N);
        double piOverN = Math.PI / N;

        for (int k = 0; k < N; k++)
        {
            double sum = 0.0;
            double factor = piOverN * k;
            for (int n = 0; n < N; n++)
            {
                sum += input[n] * Math.Cos(factor * (2 * n + 1) / 2.0);
            }
            output[k] = k == 0 ? sum * scale0 : sum * scaleK;
        }

        return output;
    }

    /// <summary>
    /// FFT-based O(N log N) DCT-II computation.
    /// </summary>
    private static double[] ComputeType2FFT(double[] input)
    {
        int N = input.Length;
        double[] reordered = new double[N];
            
        for (int n = 0; n < N / 2; n++)
        {
            reordered[n] = input[2 * n];
            reordered[N - 1 - n] = input[2 * n + 1];
        }

        Complex[] complex = new Complex[N];
            
        for (int i = 0; i < N; i++)
            complex[i] = new Complex(reordered[i], 0);
            
        FFT.Forward(complex);

        double[] output = new double[N];
        double scale0 = Math.Sqrt(1.0 / N);
        double scaleK = Math.Sqrt(2.0 / N);

        for (int k = 0; k < N; k++)
        {
            double angle = -Math.PI * k / (2.0 * N);
            double cos = Math.Cos(angle);
            double sin = Math.Sin(angle);
            double val = complex[k].Real * cos - complex[k].Imaginary * sin;
            output[k] = k == 0 ? val * scale0 : val * scaleK;
        }

        return output;
    }

    /// <summary>
    /// Compute DCT-II along the first axis of a 2D array (optimized).
    /// Input: [nFreq, nTime], Output: [nFreq, nTime]
    /// </summary>
    public static double[,] ComputeType2AlongAxis0(double[,] input)
    {
        int nRows = input.GetLength(0);
        int nCols = input.GetLength(1);
        double[,] output = new double[nRows, nCols];
        double[] column = new double[nRows];
        double scale0 = Math.Sqrt(1.0 / nRows);
        double scaleK = Math.Sqrt(2.0 / nRows);
        double piOverN = Math.PI / nRows;
        double[,] cosTable = new double[nRows, nRows];
            
        for (int k = 0; k < nRows; k++)
        {
            double factor = piOverN * k;
            for (int n = 0; n < nRows; n++)
            {
                cosTable[k, n] = Math.Cos(factor * (2 * n + 1) / 2.0);
            }
        }

        for (int col = 0; col < nCols; col++)
        {
            for (int row = 0; row < nRows; row++)
                column[row] = input[row, col];
                
            for (int k = 0; k < nRows; k++)
            {
                double sum = 0.0;
                for (int n = 0; n < nRows; n++)
                {
                    sum += column[n] * cosTable[k, n];
                }
                output[k, col] = (k == 0) ? sum * scale0 : sum * scaleK;
            }
        }

        return output;
    }

    /// <summary>
    /// Compute DCT-III (inverse of DCT-II) with orthonormal normalization.
    /// </summary>
    public static double[] ComputeType3(double[] input)
    {
        int N = input.Length;
        double[] output = new double[N];

        double scale0 = Math.Sqrt(1.0 / N);
        double scaleK = Math.Sqrt(2.0 / N);

        for (int n = 0; n < N; n++)
        {
            double sum = input[0] * scale0;
            for (int k = 1; k < N; k++)
            {
                sum += input[k] * scaleK * Math.Cos(Math.PI * k * (2 * n + 1) / (2.0 * N));
            }
            output[n] = sum;
        }

        return output;
    }
}

/// <summary>
/// Pre-computed DCT processor for repeated transforms of the same size.
/// </summary>
public class DCTProcessor
{
    private readonly int _size;
    private readonly double _scale0;
    private readonly double _scaleK;
    private readonly double[,] _cosTable;

    public DCTProcessor(int size)
    {
        _size = size;
        _scale0 = Math.Sqrt(1.0 / size);
        _scaleK = Math.Sqrt(2.0 / size);
        _cosTable = new double[size, size];
            
        double piOverN = Math.PI / size;
        for (int k = 0; k < size; k++)
        {
            double factor = piOverN * k;
            for (int n = 0; n < size; n++)
            {
                _cosTable[k, n] = Math.Cos(factor * (2 * n + 1) / 2.0);
            }
        }
    }

    /// <summary>
    /// Compute DCT-II.
    /// </summary>
    public void Compute(double[] input, double[] output)
    {
        for (int k = 0; k < _size; k++)
        {
            double sum = 0.0;
            for (int n = 0; n < _size; n++)
            {
                sum += input[n] * _cosTable[k, n];
            }
            output[k] = k == 0 ? sum * _scale0 : sum * _scaleK;
        }
    }
}