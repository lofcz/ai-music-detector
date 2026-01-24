using System;
using System.Collections.Concurrent;

namespace CQT;

/// <summary>
/// Fast Fourier Transform. Arbitrary sizes.
/// </summary>
public static class FFT
{
    private static readonly ConcurrentDictionary<int, FFTPlan> _pow2Plans = new ConcurrentDictionary<int, FFTPlan>();
    private static readonly ConcurrentDictionary<int, BluesteinPlan> _bluesteinPlans = new ConcurrentDictionary<int, BluesteinPlan>();

    public static void Forward(Complex[] data)
    {
        int n = data.Length;
        if (n <= 1) return;
        if (IsPowerOf2(n))
            ForwardPow2(data, GetPow2Plan(n));
        else
            BluesteinForward(data, GetBluesteinPlan(n));
    }

    public static void Inverse(Complex[] data)
    {
        int n = data.Length;
        if (n <= 1) return;
        if (IsPowerOf2(n))
        {
            FFTPlan plan = GetPow2Plan(n);
            for (int i = 0; i < n; i++)
                data[i] = data[i].Conjugate;
            ForwardPow2(data, plan);
            double inv = 1.0 / n;
            for (int i = 0; i < n; i++) 
                data[i] = new Complex(data[i].Real * inv, -data[i].Imaginary * inv);
        }
        else
        {
            BluesteinInverse(data, GetBluesteinPlan(n));
        }
    }

    private static FFTPlan GetPow2Plan(int n) => _pow2Plans.GetOrAdd(n, size => new FFTPlan(size));
    private static BluesteinPlan GetBluesteinPlan(int n) => _bluesteinPlans.GetOrAdd(n, size => new BluesteinPlan(size));
    
    private static void ForwardPow2(Complex[] data, FFTPlan plan)
    {
        ForwardPow2(data, plan, plan.N);
    }
        
    private static void ForwardPow2(Complex[] data, FFTPlan plan, int n)
    {
        int[] bitRev = plan.BitRev;
        double[] cos = plan.Cos;
        double[] sin = plan.Sin;
        
        for (int i = 0; i < n; i++)
        {
            int j = bitRev[i];
            if (j > i) { (data[i], data[j]) = (data[j], data[i]);
            }
        }
        
        for (int len = 2, half = 1; len <= n; len <<= 1, half <<= 1)
        {
            int step = n / len;
            for (int i = 0; i < n; i += len)
            {
                int ti = 0;
                for (int j = 0; j < half; j++, ti += step)
                {
                    double c = cos[ti], s = sin[ti];
                    Complex u = data[i + j];
                    Complex v = data[i + j + half];
                    double vr = v.Real * c - v.Imaginary * s;
                    double vi = v.Real * s + v.Imaginary * c;
                    data[i + j] = new Complex(u.Real + vr, u.Imaginary + vi);
                    data[i + j + half] = new Complex(u.Real - vr, u.Imaginary - vi);
                }
            }
        }
    }
    
    private static void BluesteinForward(Complex[] data, BluesteinPlan plan)
    {
        int n = plan.N, m = plan.M;
        Complex[] bk = plan.Bk;
        Complex[] bkf = plan.Bkf;
        FFTPlan pow2Plan = GetPow2Plan(m);
        Complex[] a = new Complex[m];
        
        for (int k = 0; k < n; k++) a[k] = data[k] * bk[k];
        
        ForwardPow2(a, pow2Plan, m);
        for (int k = 0; k < m; k++) a[k] *= bkf[k];

        for (int i = 0; i < m; i++) a[i] = a[i].Conjugate;
        ForwardPow2(a, pow2Plan, m);
        double invM = 1.0 / m;
        for (int i = 0; i < m; i++) a[i] = new Complex(a[i].Real * invM, -a[i].Imaginary * invM);

        for (int k = 0; k < n; k++) data[k] = a[k] * bk[k];
    }

    private static void BluesteinInverse(Complex[] data, BluesteinPlan plan)
    {
        int n = plan.N, m = plan.M;
        Complex[] bkInv = plan.BkInv;
        Complex[] bkfInv = plan.BkfInv;
        FFTPlan pow2Plan = GetPow2Plan(m);
        Complex[] a = new Complex[m];

        for (int k = 0; k < n; k++) a[k] = data[k] * bkInv[k];
            
        ForwardPow2(a, pow2Plan, m);
        for (int k = 0; k < m; k++) a[k] = a[k] * bkfInv[k];
            
        for (int i = 0; i < m; i++) a[i] = a[i].Conjugate;
        ForwardPow2(a, pow2Plan, m);
        double invM = 1.0 / m;
        for (int i = 0; i < m; i++) a[i] = new Complex(a[i].Real * invM, -a[i].Imaginary * invM);
            
        double invN = 1.0 / n;
        for (int k = 0; k < n; k++) data[k] = a[k] * bkInv[k] * invN;
    }
    
    public static Complex[] Rfft(double[] realData)
    {
        int n = realData.Length;
        Complex[] complex = new Complex[n];
        for (int i = 0; i < n; i++) complex[i] = new Complex(realData[i], 0);
        Forward(complex);
        Complex[] result = new Complex[n / 2 + 1];
        for (int i = 0; i < result.Length; i++) result[i] = complex[i];
        return result;
    }

    public static double[] Irfft(Complex[] spectrum, int outputLength)
    {
        Complex[] full = new Complex[outputLength];
        int copyLen = Math.Min(spectrum.Length, outputLength);
        for (int i = 0; i < copyLen; i++) full[i] = spectrum[i];
        for (int k = 1; k < outputLength / 2; k++)
            if (k < spectrum.Length) full[outputLength - k] = spectrum[k].Conjugate;
        Inverse(full);
        double[] result = new double[outputLength];
        for (int i = 0; i < outputLength; i++) result[i] = full[i].Real;
        return result;
    }

    public static Complex[] ForwardReal(double[] realData)
    {
        Complex[] complex = new Complex[realData.Length];
        for (int i = 0; i < realData.Length; i++) complex[i] = new Complex(realData[i], 0);
        Forward(complex);
        return complex;
    }

    public static Complex[] ForwardReal(double[] realData, int fftLength)
    {
        Complex[] complex = new Complex[fftLength];
        int copyLen = Math.Min(realData.Length, fftLength);
        for (int i = 0; i < copyLen; i++) complex[i] = new Complex(realData[i], 0);
        Forward(complex);
        return complex;
    }
    
    public static bool IsPowerOf2(int n) => n > 0 && (n & (n - 1)) == 0;

    public static int NextPowerOf2(int n)
    {
        if (n <= 0) return 1;
        n--; n |= n >> 1; n |= n >> 2; n |= n >> 4; n |= n >> 8; n |= n >> 16;
        return n + 1;
    }
}

internal sealed class FFTPlan
{
    public readonly int N;
    public readonly int[] BitRev;
    public readonly double[] Cos;
    public readonly double[] Sin;

    public FFTPlan(int n)
    {
        N = n;
        int bits = 0;
        for (int temp = n; temp > 1; temp >>= 1) bits++;
            
        BitRev = new int[n];
        for (int i = 0; i < n; i++)
        {
            int r = 0, v = i;
            for (int b = 0; b < bits; b++) { r = (r << 1) | (v & 1); v >>= 1; }
            BitRev[i] = r;
        }
            
        Cos = new double[n / 2];
        Sin = new double[n / 2];
        for (int i = 0; i < n / 2; i++)
        {
            double angle = -2.0 * Math.PI * i / n;
            Cos[i] = Math.Cos(angle);
            Sin[i] = Math.Sin(angle);
        }
    }
}

internal sealed class BluesteinPlan
{
    public readonly int N;
    public readonly int M;
    public readonly Complex[] Bk;      // Forward chirp
    public readonly Complex[] Bkf;     // FFT of padded conjugate chirp
    public readonly Complex[] BkInv;   // Inverse chirp
    public readonly Complex[] BkfInv;  // FFT of padded conjugate inverse chirp

    public BluesteinPlan(int n)
    {
        N = n;
        M = FFT.NextPowerOf2(2 * n - 1);
        Bk = new Complex[n];
        BkInv = new Complex[n];
        
        for (int k = 0; k < n; k++)
        {
            double angle = Math.PI * k * k / n;
            Bk[k] = new Complex(Math.Cos(-angle), Math.Sin(-angle));
            BkInv[k] = new Complex(Math.Cos(angle), Math.Sin(angle));
        }
            
        Bkf = ComputeBkf(Bk);
        BkfInv = ComputeBkf(BkInv);
    }

    private Complex[] ComputeBkf(Complex[] bk)
    {
        Complex[] b = new Complex[M];
        b[0] = bk[0].Conjugate;
        for (int k = 1; k < N; k++)
        {
            Complex c = bk[k].Conjugate;
            b[k] = c;
            b[M - k] = c;
        }
        
        FFT.Forward(b);
        return b;
    }
}

public class FFTProcessor
{
    private readonly int _size;
        
    public FFTProcessor(int size)
    {
        if (size <= 0 || (size & (size - 1)) != 0)
            throw new ArgumentException("Size must be a power of 2", nameof(size));
        _size = size;
    }

    public void Forward(Complex[] data)
    {
        if (data.Length != _size)
            throw new ArgumentException($"Data length must be {_size}", nameof(data));
        FFT.Forward(data);
    }

    public void ForwardReal(double[] input, Complex[] output)
    {
        if (input.Length > _size || output.Length != _size)
            throw new ArgumentException("Invalid array lengths");
        for (int i = 0; i < input.Length; i++) output[i] = new Complex(input[i], 0);
        for (int i = input.Length; i < _size; i++) output[i] = Complex.Zero;
        FFT.Forward(output);
    }
}