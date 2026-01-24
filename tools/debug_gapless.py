"""Debug MP3 gapless info parsing."""
import sys
from pathlib import Path

def parse_mp3_gapless(mp3_bytes: bytes) -> dict:
    """Parse MP3 gapless info similar to C# Mp3EncoderDelay."""
    
    def find_first_frame_header(data: bytes) -> int:
        for i in range(min(len(data) - 4, 10000)):
            if data[i] == 0xFF and (data[i + 1] & 0xE0) == 0xE0:
                layer = (data[i + 1] >> 1) & 0x03
                bit_rate_idx = (data[i + 2] >> 4) & 0x0F
                sample_rate_idx = (data[i + 2] >> 2) & 0x03
                if layer == 0x01 and bit_rate_idx != 0 and bit_rate_idx != 15 and sample_rate_idx != 3:
                    return i
        return -1
    
    frame_start = find_first_frame_header(mp3_bytes)
    if frame_start < 0:
        return {'error': 'No frame header found'}
    
    print(f"First frame at offset: {frame_start}")
    
    version = (mp3_bytes[frame_start + 1] >> 3) & 0x03
    mode = (mp3_bytes[frame_start + 3] >> 6) & 0x03
    
    samples_per_frame = 1152 if version == 3 else 576
    print(f"MPEG version: {version} ({'1' if version == 3 else '2/2.5'})")
    print(f"Samples per frame: {samples_per_frame}")
    print(f"Channel mode: {mode} ({'mono' if mode == 3 else 'stereo/joint/dual'})")
    
    if version == 3:  # MPEG1
        side_info_size = 17 if mode == 3 else 32
    else:
        side_info_size = 9 if mode == 3 else 17
    
    xing_offset = frame_start + 4 + side_info_size
    print(f"Xing header expected at offset: {xing_offset}")
    
    if xing_offset + 120 < len(mp3_bytes):
        xing_tag = mp3_bytes[xing_offset:xing_offset + 4].decode('latin1', errors='replace')
        print(f"Tag at xing offset: {repr(xing_tag)}")
        
        is_xing = xing_tag == 'Xing'
        is_info = xing_tag == 'Info'
        
        if is_xing or is_info:
            print(f"Found {'Xing' if is_xing else 'Info'} header!")
            
            flags = (mp3_bytes[xing_offset + 4] << 24) | (mp3_bytes[xing_offset + 5] << 16) | \
                    (mp3_bytes[xing_offset + 6] << 8) | mp3_bytes[xing_offset + 7]
            print(f"Flags: {flags:08x}")
            
            offset = xing_offset + 8
            
            if flags & 0x01:
                frames = (mp3_bytes[offset] << 24) | (mp3_bytes[offset+1] << 16) | \
                         (mp3_bytes[offset+2] << 8) | mp3_bytes[offset+3]
                print(f"  Frames: {frames}")
                offset += 4
            if flags & 0x02:
                size = (mp3_bytes[offset] << 24) | (mp3_bytes[offset+1] << 16) | \
                       (mp3_bytes[offset+2] << 8) | mp3_bytes[offset+3]
                print(f"  Size: {size}")
                offset += 4
            if flags & 0x04:
                print(f"  Has TOC (skipping 100 bytes)")
                offset += 100
            if flags & 0x08:
                quality = (mp3_bytes[offset] << 24) | (mp3_bytes[offset+1] << 16) | \
                          (mp3_bytes[offset+2] << 8) | mp3_bytes[offset+3]
                print(f"  Quality: {quality}")
                offset += 4
            
            # Encoder tag
            encoder_tag = mp3_bytes[offset:offset + 9].decode('latin1', errors='replace')
            print(f"Encoder tag: {repr(encoder_tag)}")
            
            is_lame = encoder_tag.startswith('LAME')
            is_lavf = encoder_tag.startswith('Lavf')
            is_lavc = encoder_tag.startswith('Lavc')
            
            if is_lame or is_lavf or is_lavc:
                print(f"Found encoder: {'LAME' if is_lame else 'Lavf' if is_lavf else 'Lavc'}")
                
                # Delay at offset+21
                delay_offset = offset + 21
                b0 = mp3_bytes[delay_offset]
                b1 = mp3_bytes[delay_offset + 1]
                b2 = mp3_bytes[delay_offset + 2]
                
                start_pad = (b0 << 4) | (b1 >> 4)
                end_pad = ((b1 & 0x0F) << 8) | b2
                
                print(f"Raw delay bytes: {b0:02x} {b1:02x} {b2:02x}")
                print(f"Start pad: {start_pad}")
                print(f"End pad: {end_pad}")
                
                # FFmpeg formula
                start_skip = start_pad + 528 + 1
                print(f"FFmpeg start_skip_samples: {start_skip}")
                
                total_skip = samples_per_frame + start_skip  # Xing frame + delay
                print(f"Total skip (including Xing frame): {total_skip}")
                
                return {
                    'has_xing': True,
                    'start_pad': start_pad,
                    'end_pad': end_pad,
                    'start_skip': start_skip,
                    'total_skip': total_skip,
                    'samples_per_frame': samples_per_frame
                }
    
    return {'error': 'No Xing/LAME header found'}


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python debug_gapless.py <mp3_file>")
        sys.exit(1)
    
    mp3_path = sys.argv[1]
    print(f"Analyzing: {mp3_path}\n")
    
    with open(mp3_path, 'rb') as f:
        mp3_bytes = f.read()
    
    result = parse_mp3_gapless(mp3_bytes)
    print(f"\nResult: {result}")
