using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace AiMusicDetector;

/// <summary>
/// Type of ONNX model for AI music detection.
/// </summary>
public enum ModelType
{
    /// <summary>Regression model using fakeprint features (2D input: batch x features)</summary>
    Regression,
    
    /// <summary>CNN model using cepstrum features (4D input: batch x channels x height x width)</summary>
    CNN
}

/// <summary>
/// ONNX Runtime wrapper for AI music detection model inference.
/// Supports both regression (2D) and CNN (4D) model types.
/// </summary>
public class OnnxInference : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly string _outputName;
    private readonly int[] _inputDimensions;
    private readonly ModelType _modelType;
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
        
        // Store input dimensions and detect model type
        _inputDimensions = inputMeta.Value.Dimensions;
        _modelType = DetectModelType(_inputDimensions);
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
        
        // Store input dimensions and detect model type
        _inputDimensions = inputMeta.Value.Dimensions;
        _modelType = DetectModelType(_inputDimensions);
    }

    /// <summary>
    /// Gets the detected model type (Regression or CNN).
    /// </summary>
    public ModelType ModelType => _modelType;

    /// <summary>
    /// Gets the input dimensions of the model.
    /// </summary>
    public int[] InputDimensions => _inputDimensions;

    /// <summary>
    /// Gets the expected input size (number of features for regression models).
    /// </summary>
    public int InputSize => _inputDimensions.Length > 1 ? _inputDimensions[1] : _inputDimensions[0];

    /// <summary>
    /// Detects the model type from input dimensions.
    /// </summary>
    private static ModelType DetectModelType(int[] dimensions)
    {
        // 4D input (batch, channels, height, width) = CNN
        // 2D input (batch, features) = Regression
        return dimensions.Length >= 4 ? ModelType.CNN : ModelType.Regression;
    }

    /// <summary>
    /// Applies sigmoid activation function.
    /// </summary>
    private static float Sigmoid(float x)
    {
        return 1.0f / (1.0f + MathF.Exp(-x));
    }

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
    /// Run inference on a batch of fakeprints (for regression models).
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
    /// Run inference on a single cepstrum feature map (for CNN models).
    /// Input is reshaped to 4D tensor [1, 1, n_coeffs, n_frames].
    /// </summary>
    /// <param name="cepstrum">Cepstrum features [n_coeffs, n_frames]</param>
    /// <param name="applySigmoid">Whether to apply sigmoid to the output (default: true)</param>
    /// <returns>AI probability (0.0 = Real, 1.0 = AI-Generated)</returns>
    public float PredictCNN(float[,] cepstrum, bool applySigmoid = true)
    {
        int nCoeffs = cepstrum.GetLength(0);
        int nFrames = cepstrum.GetLength(1);

        // Create 4D input tensor [1, 1, n_coeffs, n_frames]
        var inputData = new float[1 * 1 * nCoeffs * nFrames];
        int idx = 0;
        for (int c = 0; c < nCoeffs; c++)
        {
            for (int f = 0; f < nFrames; f++)
            {
                inputData[idx++] = cepstrum[c, f];
            }
        }

        var inputTensor = new DenseTensor<float>(inputData, new[] { 1, 1, nCoeffs, nFrames });
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
        };

        // Run inference
        using var results = _session.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();
        
        float logit = outputTensor.GetValue(0);
        return applySigmoid ? Sigmoid(logit) : logit;
    }

    /// <summary>
    /// Run batch inference on multiple cepstrum feature maps (for CNN models).
    /// </summary>
    /// <param name="cepstrums">Array of cepstrum features, each [n_coeffs, n_frames]</param>
    /// <param name="applySigmoid">Whether to apply sigmoid to the output (default: true)</param>
    /// <returns>Array of AI probabilities</returns>
    public float[] PredictCNNBatch(float[][,] cepstrums, bool applySigmoid = true)
    {
        if (cepstrums.Length == 0)
            return Array.Empty<float>();

        // For now, process one at a time (batching with variable-size inputs is complex)
        var results = new float[cepstrums.Length];
        for (int i = 0; i < cepstrums.Length; i++)
        {
            results[i] = PredictCNN(cepstrums[i], applySigmoid);
        }
        return results;
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
        metadata["input_size"] = InputSize.ToString();
        metadata["model_type"] = _modelType.ToString();
        metadata["input_dimensions"] = string.Join("x", _inputDimensions);
        
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
