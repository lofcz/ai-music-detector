using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace AiMusicDetector;

/// <summary>
/// ONNX Runtime wrapper for AI music detection model inference.
/// </summary>
public class OnnxInference : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly string _outputName;
    private readonly int _inputSize;
    private bool _disposed;

    /// <summary>
    /// Creates an OnnxInference instance from a model file.
    /// </summary>
    /// <param name="modelPath">Path to the ONNX model file</param>
    /// <param name="useGpu">Whether to use GPU acceleration (requires CUDA)</param>
    public OnnxInference(string modelPath, bool useGpu = false)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}");

        var options = new SessionOptions();
        
        if (useGpu)
        {
            try
            {
                options.AppendExecutionProvider_CUDA();
            }
            catch
            {
                // Fall back to CPU if CUDA is not available
                options.AppendExecutionProvider_CPU();
            }
        }
        else
        {
            options.AppendExecutionProvider_CPU();
        }

        _session = new InferenceSession(modelPath, options);
        
        // Get input/output metadata
        var inputMeta = _session.InputMetadata.First();
        var outputMeta = _session.OutputMetadata.First();
        
        _inputName = inputMeta.Key;
        _outputName = outputMeta.Key;
        
        // Get input size (second dimension after batch)
        var inputDims = inputMeta.Value.Dimensions;
        _inputSize = inputDims.Length > 1 ? inputDims[1] : inputDims[0];
    }

    /// <summary>
    /// Creates an OnnxInference instance from a byte array.
    /// </summary>
    /// <param name="modelData">ONNX model data</param>
    /// <param name="useGpu">Whether to use GPU acceleration</param>
    public OnnxInference(byte[] modelData, bool useGpu = false)
    {
        var options = new SessionOptions();
        
        if (useGpu)
        {
            try
            {
                options.AppendExecutionProvider_CUDA();
            }
            catch
            {
                options.AppendExecutionProvider_CPU();
            }
        }
        else
        {
            options.AppendExecutionProvider_CPU();
        }

        _session = new InferenceSession(modelData, options);
        
        var inputMeta = _session.InputMetadata.First();
        var outputMeta = _session.OutputMetadata.First();
        
        _inputName = inputMeta.Key;
        _outputName = outputMeta.Key;
        
        var inputDims = inputMeta.Value.Dimensions;
        _inputSize = inputDims.Length > 1 ? inputDims[1] : inputDims[0];
    }

    /// <summary>
    /// Gets the expected input size (number of features).
    /// </summary>
    public int InputSize => _inputSize;

    /// <summary>
    /// Run inference on a single fakeprint.
    /// </summary>
    /// <param name="fakeprint">Fakeprint feature vector</param>
    /// <returns>AI probability (0.0 = Real, 1.0 = AI-Generated)</returns>
    public float Predict(float[] fakeprint)
    {
        var results = PredictBatch(new[] { fakeprint });
        return results[0];
    }

    /// <summary>
    /// Run inference on a batch of fakeprints.
    /// </summary>
    /// <param name="fakeprints">Array of fakeprint feature vectors</param>
    /// <returns>Array of AI probabilities</returns>
    public float[] PredictBatch(float[][] fakeprints)
    {
        if (fakeprints.Length == 0)
            return Array.Empty<float>();

        int batchSize = fakeprints.Length;
        int featureSize = fakeprints[0].Length;

        // Create input tensor
        var inputData = new float[batchSize * featureSize];
        for (int i = 0; i < batchSize; i++)
        {
            Array.Copy(fakeprints[i], 0, inputData, i * featureSize, featureSize);
        }

        var inputTensor = new DenseTensor<float>(inputData, new[] { batchSize, featureSize });
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
        };

        // Run inference
        using var results = _session.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        // Extract probabilities
        var probabilities = new float[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            probabilities[i] = outputTensor[i, 0];
        }

        return probabilities;
    }

    /// <summary>
    /// Get model metadata.
    /// </summary>
    /// <returns>Dictionary of metadata key-value pairs</returns>
    public Dictionary<string, string> GetMetadata()
    {
        var metadata = new Dictionary<string, string>();
        
        foreach (var kvp in _session.ModelMetadata.CustomMetadataMap)
        {
            metadata[kvp.Key] = kvp.Value;
        }
        
        metadata["input_name"] = _inputName;
        metadata["output_name"] = _outputName;
        metadata["input_size"] = _inputSize.ToString();
        
        return metadata;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _session?.Dispose();
            _disposed = true;
            GC.SuppressFinalize(this);
        }
    }
}
