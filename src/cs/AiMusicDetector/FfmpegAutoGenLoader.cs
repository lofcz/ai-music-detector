using System;
using System.IO;
using System.Runtime.InteropServices;
using FFmpeg.AutoGen.Bindings.DynamicallyLoaded;

namespace AiMusicDetector;

internal static class FfmpegAutoGenLoader
{
    private static int _initialized;

    public static bool TryInitialize(out string? error)
    {
        error = null;
        if (System.Threading.Interlocked.CompareExchange(ref _initialized, 1, 0) != 0)
            return true;

        try
        {
            string? lastError = null;

            foreach (var candidate in GetCandidateLibraryPaths())
            {
                try
                {
                    DynamicallyLoadedBindings.LibrariesPath = candidate;
                    DynamicallyLoadedBindings.Initialize();
                    return true;
                }
                catch (Exception ex)
                {
                    lastError = ex.Message;
                }
            }

            error = lastError ?? "FFmpeg libraries not found in any known location.";
            System.Threading.Interlocked.Exchange(ref _initialized, 0);
            return false;
        }
        catch (Exception ex)
        {
            error = ex.Message;
            // allow retry attempts
            System.Threading.Interlocked.Exchange(ref _initialized, 0);
            return false;
        }
    }

    private static string[] GetCandidateLibraryPaths()
    {
        // Highest priority: explicit override.
        var overridePath = Environment.GetEnvironmentVariable("AIMUSICDETECTOR_FFMPEG_LIBS");
        if (!string.IsNullOrWhiteSpace(overridePath) && Directory.Exists(overridePath))
            return new[] { overridePath };

        var rid = GetPlatformRidFolder();
        var baseDir = AppContext.BaseDirectory;

        // NuGet-native-assets standard layout:
        // - runtimes/<rid>/native
        // Also try the app root itself, since many build/publish pipelines flatten native assets there.
        // Keep old layouts for local dev too.
        var list = new System.Collections.Generic.List<string>(capacity: 8)
        {
            Path.Combine(baseDir, "runtimes", rid, "native"),
            baseDir,
        };

        // Existing local-dev layouts:
        list.AddRange(GetLegacyBundledPaths(baseDir, rid));

        // Also search up the directory tree (useful for running from test projects, etc.).
        var current = baseDir;
        while (!string.IsNullOrEmpty(current))
        {
            list.AddRange(GetLegacyBundledPaths(current, rid));
            current = Directory.GetParent(current)?.FullName;
        }

        // Deduplicate + filter existing directories.
        var seen = new System.Collections.Generic.HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var result = new System.Collections.Generic.List<string>(capacity: list.Count);
        foreach (var p in list)
        {
            if (string.IsNullOrWhiteSpace(p))
                continue;
            if (!Directory.Exists(p))
                continue;
            if (seen.Add(p))
                result.Add(p);
        }

        return result.ToArray();
    }

    private static System.Collections.Generic.IEnumerable<string> GetLegacyBundledPaths(string baseDir, string rid)
    {
        // Probe common layouts:
        // - FFmpeg/bin/<rid> (recommended for manual bundling)
        // - FFmpeg/bin/x64 or x86 (AutoGen example layout on Windows)
        yield return Path.Combine(baseDir, "FFmpeg", "bin", rid);

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            var legacy = Environment.Is64BitProcess ? "x64" : "x86";
            yield return Path.Combine(baseDir, "FFmpeg", "bin", legacy);
        }
    }

    private static string GetPlatformRidFolder()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            return RuntimeInformation.ProcessArchitecture switch
            {
                Architecture.Arm64 => "win-arm64",
                _ => "win-x64"
            };
        }
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            return RuntimeInformation.ProcessArchitecture switch
            {
                Architecture.Arm64 => "linux-arm64",
                _ => "linux-x64"
            };
        }
        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            return RuntimeInformation.ProcessArchitecture switch
            {
                Architecture.Arm64 => "osx-arm64",
                _ => "osx-x64"
            };
        }

        return "unknown";
    }
}

