using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace CQT;

/// <summary>
/// Complex number.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public readonly struct Complex : IEquatable<Complex>
{
    public readonly double Real;
    public readonly double Imaginary;

    /// <summary>
    /// A complex number with real and imaginary parts equal to zero.
    /// </summary>
    public static readonly Complex Zero = new Complex(0.0, 0.0);

    /// <summary>
    /// A complex number with real part equal to one and imaginary part equal to zero.
    /// </summary>
    public static readonly Complex One = new Complex(1.0, 0.0);

    /// <summary>
    /// A complex number with real part equal to zero and imaginary part equal to one.
    /// </summary>
    public static readonly Complex ImaginaryOne = new Complex(0.0, 1.0);

    /// <summary>
    /// A complex number representing positive infinity.
    /// </summary>
    public static readonly Complex PositiveInfinity = new Complex(double.PositiveInfinity, double.PositiveInfinity);

    /// <summary>
    /// A complex number representing NaN.
    /// </summary>
    public static readonly Complex NaN = new Complex(double.NaN, double.NaN);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Complex(double real, double imaginary)
    {
        Real = real;
        Imaginary = imaginary;
    }

    /// <summary>
    /// Gets the magnitude (absolute value).
    /// </summary>
    public double Magnitude
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            if (double.IsNaN(Real) || double.IsNaN(Imaginary))
                return double.NaN;
            if (double.IsInfinity(Real) || double.IsInfinity(Imaginary))
                return double.PositiveInfinity;

            double a = Math.Abs(Real);
            double b = Math.Abs(Imaginary);

            if (a > b)
            {
                double tmp = b / a;
                return a * Math.Sqrt(1.0 + tmp * tmp);
            }
            if (a == 0.0)
            {
                return b;
            }
            else
            {
                double tmp = a / b;
                return b * Math.Sqrt(1.0 + tmp * tmp);
            }
        }
    }

    /// <summary>
    /// Gets the squared magnitude. Faster than Magnitude when you only need relative comparisons.
    /// </summary>
    public double MagnitudeSquared
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Real * Real + Imaginary * Imaginary;
    }

    /// <summary>
    /// Gets the phase (argument) of the complex number.
    /// </summary>
    public double Phase
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Imaginary == 0.0 && Real < 0.0 ? Math.PI : Math.Atan2(Imaginary, Real);
    }

    /// <summary>
    /// Gets the complex conjugate.
    /// </summary>
    public Complex Conjugate
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => new Complex(Real, -Imaginary);
    }

    /// <summary>
    /// Returns true if this is the zero complex number.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool IsZero() => Real == 0.0 && Imaginary == 0.0;

    /// <summary>
    /// Returns true if this is a real number (imaginary part is zero).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool IsReal() => Imaginary == 0.0;

    /// <summary>
    /// Returns true if either component is NaN.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool IsNaN() => double.IsNaN(Real) || double.IsNaN(Imaginary);

    /// <summary>
    /// Returns true if either component is infinite.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool IsInfinity() => double.IsInfinity(Real) || double.IsInfinity(Imaginary);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Complex FromPolarCoordinates(double magnitude, double phase)
    {
        return new Complex(
            magnitude * Math.Cos(phase),
            magnitude * Math.Sin(phase)
        );
    }

    /// <summary>
    /// Computes e^value (exponential function).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Complex Exp(Complex value)
    {
        double expReal = Math.Exp(value.Real);
        if (value.Imaginary == 0.0)
        {
            return new Complex(expReal, 0.0);
        }
        return new Complex(
            expReal * Math.Cos(value.Imaginary),
            expReal * Math.Sin(value.Imaginary)
        );
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Complex operator +(Complex left, Complex right)
        => new Complex(left.Real + right.Real, left.Imaginary + right.Imaginary);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Complex operator -(Complex left, Complex right)
        => new Complex(left.Real - right.Real, left.Imaginary - right.Imaginary);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Complex operator *(Complex left, Complex right)
        => new Complex(
            left.Real * right.Real - left.Imaginary * right.Imaginary,
            left.Real * right.Imaginary + left.Imaginary * right.Real
        );

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Complex operator *(Complex left, double right)
        => new Complex(left.Real * right, left.Imaginary * right);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Complex operator *(double left, Complex right)
        => new Complex(left * right.Real, left * right.Imaginary);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Complex operator /(Complex left, double right)
        => new Complex(left.Real / right, left.Imaginary / right);

    /// <summary>
    /// Division.
    /// </summary>
    public static Complex operator /(Complex dividend, Complex divisor)
    {
        if (dividend.IsZero() && divisor.IsZero())
        {
            return NaN;
        }

        if (divisor.IsZero())
        {
            return PositiveInfinity;
        }

        double a = dividend.Real;
        double b = dividend.Imaginary;
        double c = divisor.Real;
        double d = divisor.Imaginary;
            
        return Math.Abs(d) <= Math.Abs(c) ? InternalDiv(a, b, c, d, false) : InternalDiv(b, a, d, c, true);
    }
        
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Complex InternalDiv(double a, double b, double c, double d, bool swapped)
    {
        double r = d / c;
        double t = 1.0 / (c + d * r);
        double e, f;

        if (r != 0.0)
        {
            e = (a + b * r) * t;
            f = (b - a * r) * t;
        }
        else
        {
            e = (a + d * (b / c)) * t;
            f = (b - d * (a / c)) * t;
        }

        if (swapped)
        {
            f = -f;
        }

        return new Complex(e, f);
    }

    /// <summary>
    /// Division of a real number by a complex number.
    /// </summary>
    public static Complex operator /(double dividend, Complex divisor)
    {
        if (dividend == 0.0 && divisor.IsZero())
        {
            return NaN;
        }

        if (divisor.IsZero())
        {
            return PositiveInfinity;
        }

        double c = divisor.Real;
        double d = divisor.Imaginary;

        return Math.Abs(d) <= Math.Abs(c) ? InternalDiv(dividend, 0.0, c, d, false) : InternalDiv(0.0, dividend, d, c, true);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Complex operator -(Complex value)
        => new Complex(-value.Real, -value.Imaginary);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Complex operator +(Complex value)
        => value;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool Equals(Complex other)
        => Real == other.Real && Imaginary == other.Imaginary;

    public override bool Equals(object obj)
        => obj is Complex other && Equals(other);

    public override int GetHashCode()
    {
        int hash = 27;
        hash = (13 * hash) + Real.GetHashCode();
        hash = (13 * hash) + Imaginary.GetHashCode();
        return hash;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool operator ==(Complex left, Complex right) => left.Equals(right);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool operator !=(Complex left, Complex right) => !left.Equals(right);

    public override string ToString() => $"({Real}, {Imaginary}i)";

    /// <summary>
    /// Implicit conversion from double to Complex.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator Complex(double value) => new Complex(value, 0.0);
}