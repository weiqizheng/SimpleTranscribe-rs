use std::fs::File;
use std::path::Path;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

const WHISPER_SAMPLE_RATE: u32 = 16000;

pub fn parse_audio_file(audio_path: &str) -> Vec<f32> {
    // Create a media source. Note that the MediaSource trait is automatically implemented for File,
    // among other types.
    let file = Box::new(File::open(Path::new(&audio_path)).unwrap());

    // Create the media source stream using the boxed media source from above.
    let mss = MediaSourceStream::new(file, Default::default());

    // Create a hint to help the format registry guess what format reader is appropriate. In this
    // example we'll leave it empty.
    let hint = Hint::new();

    // Use the default options when reading and decoding.
    let format_opts: FormatOptions = Default::default();
    let metadata_opts: MetadataOptions = Default::default();
    let decoder_opts: DecoderOptions = Default::default();

    // Probe the media source stream for a format.
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .unwrap();

    // Get the format reader yielded by the probe operation.
    let mut format = probed.format;

    // Get the default track.
    let track = format.default_track().unwrap();

    if let Some(sample_rate) = track.codec_params.sample_rate {
        if sample_rate != WHISPER_SAMPLE_RATE {
            panic!(
                "audio sample rate must be 16KHz, use {} to convert to mono,16KHz,f32 audio",
                "ffmpeg -i <input_audio_file> -ac 1 -ar 16000 -sample_fmt fltp <output_audio_file>"
            );
        }
    }

    if let Some(channels) = track.codec_params.channels {
        let channel_count = channels.count();
        if channel_count > 2 {
            panic!(
                "{} channels not supported, use {} to convert to mono,16KHz,f32 audio",
                channel_count,
                "ffmpeg -i <input_audio_file> -ac 1 -ar 16000 -sample_fmt fltp <output_audio_file>"
            );
        }
    }

    // Create a decoder for the track.
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &decoder_opts)
        .unwrap();

    // Store the track identifier, we'll use it to filter packets.
    let track_id = track.id;

    let mut sample_buf = None;

    let mut audio_data: Vec<f32> = vec![];
    loop {
        // Get the next packet from the format reader.
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(Error::ResetRequired) => {
                // The track list has been changed. Re-examine it and create a new set of decoders,
                // then restart the decode loop. This is an advanced feature and it is not
                // unreasonable to consider this "the end." As of v0.5.0, the only usage of this is
                // for chained OGG physical streams.
                unimplemented!();
            }
            Err(Error::IoError(_)) => {
                break;
            }
            Err(err) => {
                // A unrecoverable error occured, halt decoding.
                panic!("{}", err);
            }
        };
        // If the packet does not belong to the selected track, skip it.
        if packet.track_id() != track_id {
            continue;
        }

        // Decode the packet into audio samples, ignoring any decode errors.
        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                // The decoded audio samples may now be accessed via the audio buffer if per-channel
                // slices of samples in their native decoded format is desired. Use-cases where
                // the samples need to be accessed in an interleaved order or converted into
                // another sample format, or a byte buffer is required, are covered by copying the
                // audio buffer into a sample buffer or raw sample buffer, respectively. In the
                // example below, we will copy the audio buffer into a sample buffer in an
                // interleaved order while also converting to a f32 sample format.

                // If this is the *first* decoded packet, create a sample buffer matching the
                // decoded audio buffer format.
                if sample_buf.is_none() {
                    // Get the audio buffer specification.
                    let spec = *audio_buf.spec();

                    // Get the capacity of the decoded buffer. Note: This is capacity, not length!
                    let duration = audio_buf.capacity() as u64;

                    // Create the f32 sample buffer.
                    sample_buf = Some(SampleBuffer::<f32>::new(duration, spec));
                }

                if let Some(buf) = &mut sample_buf {
                    let is_stereo = audio_buf.spec().channels.count() == 2;
                    buf.copy_interleaved_ref(audio_buf);

                    let mut feed_buffer;
                    if is_stereo {
                        feed_buffer =
                            whisper_rs::convert_stereo_to_mono_audio(&buf.samples()).unwrap();
                    } else {
                        feed_buffer = buf.samples().to_vec();
                    }

                    // The samples may now be access via the `samples()` function.
                    audio_data.append(&mut feed_buffer);
                }
            }
            Err(Error::DecodeError(_)) => (),
            Err(_) => break,
        }
    }
    audio_data
}
