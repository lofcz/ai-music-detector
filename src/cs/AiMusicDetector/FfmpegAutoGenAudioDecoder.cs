using FFmpeg.AutoGen.Abstractions;

namespace AiMusicDetector;

internal static unsafe class FfmpegAutoGenAudioDecoder
{
    public static float[] DecodeToMonoF32(string filePath, int targetSampleRate, int maxDurationSeconds)
    {
        if (!FfmpegAutoGenLoader.TryInitialize(out var err))
            throw new InvalidOperationException($"FFmpeg libraries not available: {err}");

        ffmpeg.av_log_set_level(ffmpeg.AV_LOG_QUIET);

        AVFormatContext* fmt = null;
        AVCodecContext* codecCtx = null;
        SwrContext* swr = null;
        AVPacket* pkt = null;
        AVFrame* frame = null;

        try
        {
            fmt = ffmpeg.avformat_alloc_context();
            ThrowIfNull(fmt);

            if (ffmpeg.avformat_open_input(&fmt, filePath, null, null) < 0)
                throw new InvalidOperationException("avformat_open_input failed");

            if (ffmpeg.avformat_find_stream_info(fmt, null) < 0)
                throw new InvalidOperationException("avformat_find_stream_info failed");

            int streamIndex = ffmpeg.av_find_best_stream(fmt, AVMediaType.AVMEDIA_TYPE_AUDIO, -1, -1, null, 0);
            if (streamIndex < 0)
                throw new InvalidOperationException("No audio stream found");

            AVStream* st = fmt->streams[streamIndex];
            AVCodecParameters* codecpar = st->codecpar;
            AVCodec* dec = ffmpeg.avcodec_find_decoder(codecpar->codec_id);
            ThrowIfNull(dec);

            codecCtx = ffmpeg.avcodec_alloc_context3(dec);
            ThrowIfNull(codecCtx);
            if (ffmpeg.avcodec_parameters_to_context(codecCtx, codecpar) < 0)
                throw new InvalidOperationException("avcodec_parameters_to_context failed");

            if (ffmpeg.avcodec_open2(codecCtx, dec, null) < 0)
                throw new InvalidOperationException("avcodec_open2 failed");

            // Setup swresample to match: -ar target -af pan=mono|c0=0.5*c0+0.5*c1 -f f32le
            // Output: mono, float planar/interleaved doesn't matter since we'll request packed float.
            var outChLayout = new AVChannelLayout();
            ffmpeg.av_channel_layout_default(&outChLayout, 1);

            swr = ffmpeg.swr_alloc();
            ThrowIfNull(swr);

            // Set options
            ffmpeg.av_opt_set_chlayout(swr, "in_chlayout", &codecCtx->ch_layout, 0);
            ffmpeg.av_opt_set_int(swr, "in_sample_rate", codecCtx->sample_rate, 0);
            ffmpeg.av_opt_set_sample_fmt(swr, "in_sample_fmt", codecCtx->sample_fmt, 0);

            ffmpeg.av_opt_set_chlayout(swr, "out_chlayout", &outChLayout, 0);
            ffmpeg.av_opt_set_int(swr, "out_sample_rate", targetSampleRate, 0);
            ffmpeg.av_opt_set_sample_fmt(swr, "out_sample_fmt", AVSampleFormat.AV_SAMPLE_FMT_FLT, 0); // packed float

            // Force stereo -> mono to be simple average (0.5*L + 0.5*R), matching your ffmpeg CLI pan filter.
            if (codecCtx->ch_layout.nb_channels == 2)
            {
                double* matrix = stackalloc double[2];
                matrix[0] = 0.5;
                matrix[1] = 0.5;
                // stride = out_channels (1)
                ffmpeg.swr_set_matrix(swr, matrix, 1);
            }

            if (ffmpeg.swr_init(swr) < 0)
                throw new InvalidOperationException("swr_init failed");

            pkt = ffmpeg.av_packet_alloc();
            frame = ffmpeg.av_frame_alloc();
            ThrowIfNull(pkt);
            ThrowIfNull(frame);

            var output = new List<float>(capacity: targetSampleRate * Math.Max(1, maxDurationSeconds));
            long maxOutSamples = maxDurationSeconds > 0 ? (long)targetSampleRate * maxDurationSeconds : long.MaxValue;

            int skipRemaining = 0;
            int discardRemaining = 0;

            // Buffers for resampled output
            byte** outData = null;
            int outLinesize = 0;
            int outCapacitySamples = 0;

            while (output.Count < maxOutSamples)
            {
                int r = ffmpeg.av_read_frame(fmt, pkt);
                if (r < 0)
                    break;

                if (pkt->stream_index != streamIndex)
                {
                    ffmpeg.av_packet_unref(pkt);
                    continue;
                }

                // Demuxers inject AV_PKT_DATA_SKIP_SAMPLES (little-endian u32 skip, u32 discard, u16 reserved).
                // This matches ffmpeg_demux.c in your reference.
                ApplySkipDiscardFromPacket(pkt, ref skipRemaining, ref discardRemaining);

                // Send packet
                r = ffmpeg.avcodec_send_packet(codecCtx, pkt);
                ffmpeg.av_packet_unref(pkt);
                if (r < 0)
                    continue;

                while (true)
                {
                    r = ffmpeg.avcodec_receive_frame(codecCtx, frame);
                    if (r == ffmpeg.AVERROR(ffmpeg.EAGAIN) || r == ffmpeg.AVERROR_EOF)
                        break;
                    if (r < 0)
                        break;

                    ApplySkipDiscardToFrame(frame, ref skipRemaining, ref discardRemaining);

                    int inNbSamples = frame->nb_samples;
                    if (inNbSamples <= 0)
                    {
                        ffmpeg.av_frame_unref(frame);
                        continue;
                    }

                    // Estimate required output samples
                    long delay = ffmpeg.swr_get_delay(swr, codecCtx->sample_rate);
                    int outNb = (int)ffmpeg.av_rescale_rnd(delay + inNbSamples, targetSampleRate, codecCtx->sample_rate, AVRounding.AV_ROUND_UP);
                    if (outNb <= 0)
                    {
                        ffmpeg.av_frame_unref(frame);
                        continue;
                    }

                    if (outNb > outCapacitySamples)
                    {
                        // realloc output buffer
                        if (outData != null)
                        {
                            ffmpeg.av_freep(&outData[0]);
                            ffmpeg.av_freep(&outData);
                        }
                        outCapacitySamples = outNb;
                        int ret = ffmpeg.av_samples_alloc_array_and_samples(&outData, &outLinesize, 1, outCapacitySamples, AVSampleFormat.AV_SAMPLE_FMT_FLT, 0);
                        if (ret < 0)
                            throw new InvalidOperationException("av_samples_alloc_array_and_samples failed");
                    }

                    int converted = ffmpeg.swr_convert(swr, outData, outCapacitySamples, frame->extended_data, inNbSamples);
                    if (converted > 0)
                    {
                        int toCopy = converted;
                        long remaining = maxOutSamples - output.Count;
                        if (remaining < toCopy) toCopy = (int)remaining;

                        // outData[0] is packed float mono
                        var span = new ReadOnlySpan<float>(outData[0], toCopy);
                        for (int i = 0; i < span.Length; i++)
                            output.Add(span[i]);
                    }

                    ffmpeg.av_frame_unref(frame);
                }
            }

            // Flush decoder
            ffmpeg.avcodec_send_packet(codecCtx, null);
            while (output.Count < maxOutSamples)
            {
                int r = ffmpeg.avcodec_receive_frame(codecCtx, frame);
                if (r == ffmpeg.AVERROR_EOF || r == ffmpeg.AVERROR(ffmpeg.EAGAIN))
                    break;
                if (r < 0)
                    break;

                ApplySkipDiscardToFrame(frame, ref skipRemaining, ref discardRemaining);

                int inNbSamples = frame->nb_samples;
                if (inNbSamples <= 0)
                {
                    ffmpeg.av_frame_unref(frame);
                    continue;
                }

                long delay = ffmpeg.swr_get_delay(swr, codecCtx->sample_rate);
                int outNb = (int)ffmpeg.av_rescale_rnd(delay + inNbSamples, targetSampleRate, codecCtx->sample_rate, AVRounding.AV_ROUND_UP);
                if (outNb <= 0)
                {
                    ffmpeg.av_frame_unref(frame);
                    continue;
                }

                if (outNb > outCapacitySamples)
                {
                    if (outData != null)
                    {
                        ffmpeg.av_freep(&outData[0]);
                        ffmpeg.av_freep(&outData);
                    }
                    outCapacitySamples = outNb;
                    int ret = ffmpeg.av_samples_alloc_array_and_samples(&outData, &outLinesize, 1, outCapacitySamples, AVSampleFormat.AV_SAMPLE_FMT_FLT, 0);
                    if (ret < 0)
                        throw new InvalidOperationException("av_samples_alloc_array_and_samples failed");
                }

                int converted = ffmpeg.swr_convert(swr, outData, outCapacitySamples, frame->extended_data, inNbSamples);
                if (converted > 0)
                {
                    int toCopy = converted;
                    long remaining = maxOutSamples - output.Count;
                    if (remaining < toCopy) toCopy = (int)remaining;

                    var span = new ReadOnlySpan<float>(outData[0], toCopy);
                    for (int i = 0; i < span.Length; i++)
                        output.Add(span[i]);
                }

                ffmpeg.av_frame_unref(frame);
            }

            return output.ToArray();
        }
        finally
        {
            if (frame != null) ffmpeg.av_frame_free(&frame);
            if (pkt != null) ffmpeg.av_packet_free(&pkt);
            if (swr != null) ffmpeg.swr_free(&swr);
            if (codecCtx != null) ffmpeg.avcodec_free_context(&codecCtx);
            if (fmt != null) ffmpeg.avformat_close_input(&fmt);
        }
    }

    private static void ApplySkipDiscardFromPacket(AVPacket* pkt, ref int skipRemaining, ref int discardRemaining)
    {
        ulong size = 0;
        byte* sd = ffmpeg.av_packet_get_side_data(pkt, AVPacketSideDataType.AV_PKT_DATA_SKIP_SAMPLES, &size);
        if (sd == null || size < 10)
            return;

        uint skip = (uint)(sd[0] | (sd[1] << 8) | (sd[2] << 16) | (sd[3] << 24));
        uint discard = (uint)(sd[4] | (sd[5] << 8) | (sd[6] << 16) | (sd[7] << 24));

        // FFmpeg demuxers set skip only at stream start; reset each time side data appears.
        if (skip > 0)
            skipRemaining = (int)skip;
        if (discard > 0)
            discardRemaining = (int)discard;
    }

    private static void ApplySkipDiscardToFrame(AVFrame* frame, ref int skipRemaining, ref int discardRemaining)
    {
        if (skipRemaining <= 0 && discardRemaining <= 0)
            return;

        // Apply skip by advancing data pointers for packed/planar.
        // For simplicity, only handle common packed/planar audio formats via extended_data.
        int nbSamples = frame->nb_samples;
        int channels = frame->ch_layout.nb_channels;
        if (channels <= 0) return;

        int startSkip = 0;
        if (skipRemaining > 0)
        {
            startSkip = Math.Min(skipRemaining, nbSamples);
            skipRemaining -= startSkip;
        }

        int endDiscard = 0;
        if (discardRemaining > 0)
        {
            // Discard applies at stream end; we conservatively apply it to the tail frames as they arrive.
            endDiscard = Math.Min(discardRemaining, nbSamples - startSkip);
            discardRemaining -= endDiscard;
        }

        int remaining = nbSamples - startSkip - endDiscard;
        if (remaining <= 0)
        {
            frame->nb_samples = 0;
            return;
        }

        int bytesPerSample = ffmpeg.av_get_bytes_per_sample((AVSampleFormat)frame->format);
        if (bytesPerSample <= 0) return;

        bool planar = ffmpeg.av_sample_fmt_is_planar((AVSampleFormat)frame->format) != 0;
        if (planar)
        {
            for (int ch = 0; ch < channels; ch++)
            {
                frame->extended_data[ch] += startSkip * bytesPerSample;
            }
            frame->nb_samples = remaining;
        }
        else
        {
            // packed: one data plane
            frame->extended_data[0] += startSkip * bytesPerSample * channels;
            frame->nb_samples = remaining;
        }
    }

    private static void ThrowIfNull(void* ptr)
    {
        if (ptr == null) throw new OutOfMemoryException();
    }
}

