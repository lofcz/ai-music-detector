using System;

namespace CQT;

/// <summary>
/// SOXR HQ quality 2:1 decimation.
/// </summary>
public static class SoxrDecimate
{
    private const double Precision = 20.0;
    private const double StopbandBegin = 1.0;
    
    private static double[] _filterCoeffsFreq;
    private static double[] _filterCoeffsTime;
    private static int _numTaps;
    private static int _dftLength;
    private static readonly object _initLock = new object();
    private static bool _initialized;
    
    public static double[] Decimate2(double[] input, bool scale = false)
    {
        EnsureInitialized();
            
        int inputLen = input.Length;
        int outputLen = inputLen / 2;

        double[] output = inputLen < _numTaps ? SimpleDecimate(input) : OverlapSaveDecimate(input, outputLen);
            
        if (scale)
        {
            double scaleFactor = Math.Sqrt(2.0);
            for (int i = 0; i < output.Length; i++)
            {
                output[i] *= scaleFactor;
            }
        }

        return output;
    }

    private static void EnsureInitialized()
    {
        if (_initialized) return;
        lock (_initLock)
        {
            if (_initialized) return;
            InitializeFilter();
            _initialized = true;
        }
    }
    
    private static void InitializeFilter()
    {
        double att = (Precision + 1) * LinearToDb(2.0);
        double passbandEnd = ComputePassbandEnd(Precision);
        double Fp = passbandEnd * 0.5;
        double Fs = StopbandBegin * 0.5;
        double Fn = 1.0; 
        int numTaps = 0;
        const int k = -4;
        double beta = -1.0;
            
        double[] h = DesignLpf(Fp, Fs, Fn, att, ref numTaps, k, ref beta);
        _numTaps = numTaps;
        double d = Math.Log(numTaps) / Math.Log(2.0);
        const int minDft = 10;
        const int largeDft = 15;
        int dftPow = Math.Max(minDft, Math.Min((int)(d + 2.77), Math.Max((int)(d + 1.77), largeDft)));
        _dftLength = 1 << dftPow;
        
        int offset = _dftLength - numTaps + 1;
        
        _filterCoeffsTime = h;
            
        double[] filterTime = new double[_dftLength];
        for (int i = 0; i < numTaps; i++)
        {
            int idx = (i + offset) & (_dftLength - 1);
            filterTime[idx] = h[i];
        }
        
        _filterCoeffsFreq = new double[_dftLength];
        Array.Copy(filterTime, _filterCoeffsFreq, _dftLength);
        RdftForward(_filterCoeffsFreq);
    }
    
    private static double[] DesignLpf(double Fp, double Fs, double Fn, double att,
        ref int numTaps, int k, ref double beta)
    {
        int phases = Math.Max(k, 1);
        int modulo = Math.Max(-k, 1);
        double rho = phases == 1 ? 0.5 : (att < 120 ? 0.63 : 0.75);

        Fp /= Math.Abs(Fn);
        Fs /= Math.Abs(Fn);

        double trBw = 0.5 * (Fs - Fp);
        trBw /= phases;
        Fs /= phases;
        trBw = Math.Min(trBw, 0.5 * Fs);
        double Fc = Fs - trBw;

        KaiserParams(att, Fc, trBw, ref beta, ref numTaps);
        
        if (numTaps == 0)
        {
            numTaps = 256;
        }
        numTaps = phases > 1
            ? numTaps / phases * phases + phases - 1
            : (numTaps + modulo - 2) / modulo * modulo + 1;
        
        return MakeLpf(numTaps, Fc, beta, rho, phases);
    }

    private static double ComputePassbandEnd(double precisionBits)
    {
        double rej = precisionBits * LinearToDb(2.0);
        double to3Db = 1.0 - InvFResp(-3.0, rej);
        return 1.0 - 0.05 / to3Db;
    }

    private static double LinearToDb(double x)
    {
        return 20.0 * Math.Log10(x);
    }

    private static double DbToLinear(double x)
    {
        return Math.Exp(x * (Math.Log(10.0) * 0.05));
    }
    
    private static double InvFResp(double drop, double a)
    {
        double x = SinePhi(a);
        double s;
        drop = DbToLinear(drop);
        s = drop > 0.5 ? 1.0 - drop : drop;
        x = Math.Asin(Math.Pow(s, 1.0 / SinePow(x))) / x;
        return drop > 0.5 ? x : 1.0 - x;
    }

    private static double SinePhi(double x)
    {
        return ((2.0517e-07 * x - 1.1303e-04) * x + 0.023154) * x + 0.55924;
    }

    private static double SinePow(double x)
    {
        return Math.Log(0.5) / Math.Log(Math.Sin(x * 0.5));
    }
    
    private static void KaiserParams(double att, double Fc, double trBw, 
        ref double beta, ref int numTaps)
    {
        if (beta < 0)
            beta = KaiserBeta(att, trBw * 0.5 / Fc);
            
        double attNorm;
        if (att < 60)
        {
            attNorm = (att - 7.95) / (2.285 * Math.PI * 2);
        }
        else
        {
            attNorm = ((0.0007528358 - 1.577737e-05 * beta) * beta + 0.6248022) * beta + 0.06186902;
        }
            
        if (numTaps == 0)
            numTaps = (int)Math.Ceiling(attNorm / trBw + 1);
    }

    static readonly double[,] kaiserBetaCoefs = {
        {-6.784957e-10, 1.02856e-05, 0.1087556, -0.8988365 + 0.001},
        {-6.897885e-10, 1.027433e-05, 0.10876, -0.8994658 + 0.002},
        {-1.000683e-09, 1.030092e-05, 0.1087677, -0.9007898 + 0.003},
        {-3.654474e-10, 1.040631e-05, 0.1087085, -0.8977766 + 0.006},
        {8.106988e-09, 6.983091e-06, 0.1091387, -0.9172048 + 0.015},
        {9.519571e-09, 7.272678e-06, 0.1090068, -0.9140768 + 0.025},
        {-5.626821e-09, 1.342186e-05, 0.1083999, -0.9065452 + 0.05},
        {-9.965946e-08, 5.073548e-05, 0.1040967, -0.7672778 + 0.085},
        {1.604808e-07, -5.856462e-05, 0.1185998, -1.34824 + 0.1},
        {-1.511964e-07, 6.363034e-05, 0.1064627, -0.9876665 + 0.18}
    };
    
    private static double KaiserBeta(double att, double trBw)
    {
        switch (att)
        {
            case >= 60:
            {
                double realm = Math.Log(trBw / 0.0005) / Math.Log(2.0);
                int idx0 = Math.Max(0, Math.Min((int)realm, kaiserBetaCoefs.GetLength(0) - 1));
                int idx1 = Math.Max(0, Math.Min(1 + (int)realm, kaiserBetaCoefs.GetLength(0) - 1));
                
                double b0 = ((kaiserBetaCoefs[idx0, 0] * att + kaiserBetaCoefs[idx0, 1]) * att + kaiserBetaCoefs[idx0, 2]) * att + kaiserBetaCoefs[idx0, 3];
                double b1 = ((kaiserBetaCoefs[idx1, 0] * att + kaiserBetaCoefs[idx1, 1]) * att + kaiserBetaCoefs[idx1, 2]) * att + kaiserBetaCoefs[idx1, 3];
                
                return b0 + (b1 - b0) * (realm - (int)realm);
            }
            case > 50:
                return 0.1102 * (att - 8.7);
            case > 20.96:
                return 0.58417 * Math.Pow(att - 20.96, 0.4) + 0.07886 * (att - 20.96);
            default:
                return 0;
        }
    }
    
    private static double[] MakeLpf(int numTaps, double Fc, double beta, double rho, double scale)
    {
        int m = numTaps - 1;
        double[] h = new double[numTaps];
        double mult = scale / BesselI0.Compute(beta);
        double mult1 = 1.0 / (0.5 * m + rho);
            
        for (int i = 0; i <= m / 2; i++)
        {
            double z = i - 0.5 * m;
            double x = z * Math.PI;
            double y = z * mult1;
            double sinc = x != 0 ? Math.Sin(Fc * x) / x : Fc;
            double arg = 1 - y * y;
            double window = arg >= 0 ? BesselI0.Compute(beta * Math.Sqrt(arg)) * mult : 0;
                
            h[i] = sinc * window;
            
            if (m - i != i)
                h[m - i] = h[i];
        }
            
        return h;
    }

    private static double[] OverlapSaveDecimate(double[] input, int outputLen)
    {
        int M = 2;
        int inputLen = input.Length;
        int filterLen = _numTaps;
        int convLen = inputLen + filterLen - 1;
        int fftLen = FFT.NextPowerOf2(convLen);
        double[] inputPad = new double[fftLen];
        Array.Copy(input, inputPad, inputLen);
        
        double[] filterPad = new double[fftLen];
        Array.Copy(_filterCoeffsTime, filterPad, filterLen);
        
        RdftForward(inputPad);
        RdftForward(filterPad);
        OrderedConvolve(inputPad, filterPad);
        RdftBackward(inputPad);
        
        int filterDelay = (filterLen - 1) / 2;
        double[] output = new double[outputLen];
        
        for (int n = 0; n < outputLen; n++)
        {
            int convIdx = M * n + filterDelay;
            if (convIdx >= 0 && convIdx < convLen)
            {
                output[n] = inputPad[convIdx];
            }
        }
            
        return output;
    }
    
    private static double[] SimpleDecimate(double[] input)
    {
        int outputLen = input.Length / 2;
        double[] output = new double[outputLen];
        for (int i = 0; i < outputLen; i++)
            output[i] = input[i * 2];
        return output;
    }
    
    private static void RdftForward(double[] data)
    {
        int n = data.Length;
        Complex[] complex = new Complex[n];
        for (int i = 0; i < n; i++)
            complex[i] = new Complex(data[i], 0);
            
        FFT.Forward(complex);
        
        data[0] = complex[0].Real;
        data[1] = complex[n / 2].Real;
        for (int i = 1; i < n / 2; i++)
        {
            data[2 * i] = complex[i].Real;
            data[2 * i + 1] = complex[i].Imaginary;
        }
    }
    
    private static void RdftBackward(double[] data)
    {
        int n = data.Length;
        Complex[] complex = new Complex[n];
        complex[0] = new Complex(data[0], 0);
        complex[n / 2] = new Complex(data[1], 0);
        
        for (int i = 1; i < n / 2; i++)
        {
            complex[i] = new Complex(data[2 * i], data[2 * i + 1]);
            complex[n - i] = new Complex(data[2 * i], -data[2 * i + 1]);
        }
            
        FFT.Inverse(complex);
            
        for (int i = 0; i < n; i++)
            data[i] = complex[i].Real;
    }
    
    private static void OrderedConvolve(double[] a, double[] b)
    {
        int n = a.Length;
        a[0] *= b[0];
        a[1] *= b[1];

        for (int i = 2; i < n; i += 2)
        {
            double tmp = a[i];
            a[i] = b[i] * tmp - b[i + 1] * a[i + 1];
            a[i + 1] = b[i + 1] * tmp + b[i] * a[i + 1];
        }
    }
    
    public static (int NumTaps, int DftLength, double[] FilterCoeffs) GetFilterInfo()
    {
        EnsureInitialized();
        return (_numTaps, _dftLength, (double[])_filterCoeffsTime.Clone());
    }
}