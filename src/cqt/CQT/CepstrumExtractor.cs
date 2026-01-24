using System;

namespace CQT;

/// <summary>
/// CQT-Cepstrum feature extractor using librosa-compatible CQT.
/// Thread-safe: each Extract() call uses local buffers.
/// </summary>
public class CepstrumExtractor
{
    private readonly DCTProcessor _dctProcessor;

    public int NumCoefficients { get; }
    public int NumBins { get; }
    public LibrosaCQT CQT { get; }

    public CepstrumExtractor(
        int sampleRate,
        double fMin,
        int nBins,
        int binsPerOctave = 12,
        int hopLength = 512,
        int nCoeffs = 24,
        double filterScale = 1.0,
        double sparsity = 0.01,
        bool scale = true)
    {
        CQT = new LibrosaCQT(sampleRate, fMin, nBins, binsPerOctave, hopLength,
            filterScale, sparsity, scale);
        NumBins = nBins;
        NumCoefficients = Math.Min(nCoeffs, nBins);
        _dctProcessor = new DCTProcessor(nBins);
    }

    /// <summary>
    /// Extract cepstrum features from audio.
    /// Thread-safe: uses local buffers for computation.
    /// </summary>
    public double[,] Extract(double[] audio)
    {
        double[,] cqtMag = CQT.Compute(audio);
        int nBins = cqtMag.GetLength(0);
        int nFrames = cqtMag.GetLength(1);

        double[,] result = new double[NumCoefficients, nFrames];
        
        // Thread-local buffers to ensure thread safety
        double[] logColumn = new double[nBins];
        double[] dctColumn = new double[nBins];
            
        for (int t = 0; t < nFrames; t++)
        {
            for (int k = 0; k < nBins; k++)
            {
                logColumn[k] = Math.Log(cqtMag[k, t] + 1e-6);
            }

            _dctProcessor.Compute(logColumn, dctColumn);
                
            for (int k = 0; k < NumCoefficients; k++)
            {
                result[k, t] = dctColumn[k];
            }
        }

        return result;
    }

    /// <summary>
    /// Extract cepstrum from a segment of audio.
    /// </summary>
    public double[,] ExtractSegment(double[] audio, int segmentStart, int segmentLength)
    {
        double[] segment = new double[segmentLength];
        int copyLen = Math.Min(segmentLength, audio.Length - segmentStart);
        if (copyLen > 0)
            Array.Copy(audio, segmentStart, segment, 0, copyLen);
        return Extract(segment);
    }

    /// <summary>
    /// Extract cepstrum and return as flattened 1D array (row-major).
    /// </summary>
    public double[] ExtractFlat(double[] audio)
    {
        double[,] result2D = Extract(audio);
        int nRows = result2D.GetLength(0);
        int nCols = result2D.GetLength(1);
            
        double[] flat = new double[nRows * nCols];
        int idx = 0;
        for (int r = 0; r < nRows; r++)
        {
            for (int c = 0; c < nCols; c++)
            {
                flat[idx++] = result2D[r, c];
            }
        }
        return flat;
    }
}