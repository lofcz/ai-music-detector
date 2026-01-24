using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace CQT.Tests;

[TestClass]
public class DCTTests
{
    private const double Tolerance = 1e-10;

    [TestMethod]
    public void DCT_Type2_InverseIsType3()
    {
        var original = new double[] { 1, 2, 3, 4, 5, 6, 7, 8 };

        var dct = DCT.ComputeType2(original);
        var recovered = DCT.ComputeType3(dct);

        for (int i = 0; i < original.Length; i++)
        {
            Assert.AreEqual(original[i], recovered[i], 1e-10, $"Value at {i}");
        }
    }

    [TestMethod]
    public void DCT_Type2_MatchesPythonReference()
    {
        if (!TestDataLoader.TestDataExists("dct_simple"))
        {
            Assert.Inconclusive("Test data not generated. Run python generate_test_data.py");
            return;
        }

        var testCase = TestDataLoader.LoadTestCase("dct_simple");
        var input = TestDataLoader.GetDoubleArray(testCase.GetProperty("input"));
        var expected = TestDataLoader.GetDoubleArray(testCase.GetProperty("expected"));

        var result = DCT.ComputeType2(input);

        for (int i = 0; i < result.Length; i++)
        {
            Assert.AreEqual(expected[i], result[i], 1e-10, $"Value at {i}");
        }
    }

    [TestMethod]
    public void DCT_Type2_Random_MatchesPythonReference()
    {
        if (!TestDataLoader.TestDataExists("dct_random"))
        {
            Assert.Inconclusive("Test data not generated. Run python generate_test_data.py");
            return;
        }

        var testCase = TestDataLoader.LoadTestCase("dct_random");
        var input = TestDataLoader.GetDoubleArray(testCase.GetProperty("input"));
        var expected = TestDataLoader.GetDoubleArray(testCase.GetProperty("expected"));

        var result = DCT.ComputeType2(input);

        for (int i = 0; i < result.Length; i++)
        {
            Assert.AreEqual(expected[i], result[i], 1e-10, $"Value at {i}");
        }
    }

    [TestMethod]
    public void DCT_2D_AlongAxis0_MatchesPythonReference()
    {
        if (!TestDataLoader.TestDataExists("dct_2d"))
        {
            Assert.Inconclusive("Test data not generated. Run python generate_test_data.py");
            return;
        }

        var testCase = TestDataLoader.LoadTestCase("dct_2d");
        var input = TestDataLoader.GetDouble2DArray(testCase.GetProperty("input"));
        var expected = TestDataLoader.GetDouble2DArray(testCase.GetProperty("expected"));

        var result = DCT.ComputeType2AlongAxis0(input);

        int nRows = expected.GetLength(0);
        int nCols = expected.GetLength(1);

        for (int i = 0; i < nRows; i++)
        {
            for (int j = 0; j < nCols; j++)
            {
                Assert.AreEqual(expected[i, j], result[i, j], 1e-10, $"Value at [{i},{j}]");
            }
        }
    }

    [TestMethod]
    public void DCT_EnergyCompaction()
    {
        // DCT should concentrate energy in low coefficients for smooth signals
        var smooth = new double[32];
        for (int i = 0; i < 32; i++)
            smooth[i] = Math.Sin(2 * Math.PI * i / 32);

        var dct = DCT.ComputeType2(smooth);

        // Most energy should be in first few coefficients
        double lowEnergy = 0, highEnergy = 0;
        for (int i = 0; i < 4; i++)
            lowEnergy += dct[i] * dct[i];
        for (int i = 4; i < 32; i++)
            highEnergy += dct[i] * dct[i];

        Assert.IsTrue(lowEnergy > highEnergy, "Low coefficients should have more energy");
    }
}