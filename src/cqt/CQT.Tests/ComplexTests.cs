using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace CQT.Tests;

[TestClass]
public class ComplexTests
{
    private const double Tolerance = 1e-10;

    [TestMethod]
    public void Complex_Magnitude_IsCorrect()
    {
        var c = new Complex(3, 4);
        Assert.AreEqual(5.0, c.Magnitude, Tolerance);
    }

    [TestMethod]
    public void Complex_Addition_IsCorrect()
    {
        var a = new Complex(1, 2);
        var b = new Complex(3, 4);
        var sum = a + b;

        Assert.AreEqual(4.0, sum.Real, Tolerance);
        Assert.AreEqual(6.0, sum.Imaginary, Tolerance);
    }

    [TestMethod]
    public void Complex_Multiplication_IsCorrect()
    {
        var a = new Complex(1, 2);
        var b = new Complex(3, 4);
        var product = a * b;

        // (1+2i)(3+4i) = 3 + 4i + 6i + 8i² = 3 + 10i - 8 = -5 + 10i
        Assert.AreEqual(-5.0, product.Real, Tolerance);
        Assert.AreEqual(10.0, product.Imaginary, Tolerance);
    }

    [TestMethod]
    public void Complex_Conjugate_IsCorrect()
    {
        var c = new Complex(3, 4);
        var conj = c.Conjugate;

        Assert.AreEqual(3.0, conj.Real, Tolerance);
        Assert.AreEqual(-4.0, conj.Imaginary, Tolerance);
    }

    [TestMethod]
    public void Complex_Exp_MatchesEulerFormula()
    {
        // e^(i*pi) = -1
        var result = Complex.Exp(new Complex(0, Math.PI));
        Assert.AreEqual(-1.0, result.Real, 1e-10);
        Assert.AreEqual(0.0, result.Imaginary, 1e-10);

        // e^(i*pi/2) = i
        result = Complex.Exp(new Complex(0, Math.PI / 2));
        Assert.AreEqual(0.0, result.Real, 1e-10);
        Assert.AreEqual(1.0, result.Imaginary, 1e-10);
    }

    [TestMethod]
    public void Complex_Division_IsCorrect()
    {
        var a = new Complex(1, 2);
        var b = new Complex(3, 4);
        var quotient = a / b;

        // (1+2i)/(3+4i) = (1+2i)(3-4i)/25 = (3-4i+6i+8)/25 = (11+2i)/25
        Assert.AreEqual(11.0 / 25.0, quotient.Real, Tolerance);
        Assert.AreEqual(2.0 / 25.0, quotient.Imaginary, Tolerance);
    }

    [TestMethod]
    public void Complex_FromPolarCoordinates_IsCorrect()
    {
        var c = Complex.FromPolarCoordinates(5, Math.PI / 4);
            
        Assert.AreEqual(5 * Math.Cos(Math.PI / 4), c.Real, Tolerance);
        Assert.AreEqual(5 * Math.Sin(Math.PI / 4), c.Imaginary, Tolerance);
    }
}