from speechbrain.processing.features import STFT, Filterbank, ContextWindow,DCT, InputNormalization
from speechbrain.processing.features import spectral_magnitude, Deltas
import torch
compute_STFT = STFT(
     sample_rate=16000, win_length=25, hop_length=10, n_fft=400
    ).cuda()
 
compute_fbanks = Filterbank(n_mels=40).cuda()

compute_cw = ContextWindow(left_frames=5, right_frames=5).cuda()
compute_mfccs = DCT(input_size=40, n_out=20).cuda()

inputnorm = InputNormalization().cuda()
def MFCC(signal) : 
    features = compute_STFT(signal)
    features = spectral_magnitude(features)
    features = compute_fbanks(features)

    features = compute_mfccs(features)
    compute_deltas = Deltas(input_size=20).cuda()
    delta1 = compute_deltas(features)
    delta2 = compute_deltas(delta1)
    features = torch.cat([features, delta1, delta2], dim=2)
 
    return features
signal_workers = ['F0semitoneFrom27.5Hz_sma3nz_amean',
       'F0semitoneFrom27.5Hz_sma3nz_stddevNorm',
       'F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope', 'loudness_sma3_amean',
       'loudness_sma3_stddevNorm',  'jitterLocal_sma3nz_amean',
       'jitterLocal_sma3nz_stddevNorm', 'shimmerLocaldB_sma3nz_amean',
       'shimmerLocaldB_sma3nz_stddevNorm',      'logRelF0-H1-H2_sma3nz_amean',
       'logRelF0-H1-H2_sma3nz_stddevNorm', 'logRelF0-H1-A3_sma3nz_amean',
       'logRelF0-H1-A3_sma3nz_stddevNorm', 'F1frequency_sma3nz_amean',
       'F1frequency_sma3nz_stddevNorm', 'F1bandwidth_sma3nz_amean',
       'F1bandwidth_sma3nz_stddevNorm', 'F1amplitudeLogRelF0_sma3nz_amean',
       'F1amplitudeLogRelF0_sma3nz_stddevNorm', 'F2frequency_sma3nz_amean',
       'F2frequency_sma3nz_stddevNorm', 'F2amplitudeLogRelF0_sma3nz_amean',
        "alphaRatio_sma3", 
       'alphaRatioV_sma3nz_amean', "Loudness_sma3",'F0final_sma', 'voicingFinalUnclipped_sma', 'jitterLocal_sma',
       'jitterDDP_sma', 'shimmerLocal_sma', 'logHNR_sma',
       'audspec_lengthL1norm_sma', 'audspecRasta_lengthL1norm_sma',
           'pcm_RMSenergy_sma', 'pcm_zcr_sma',
       'alphaRatioV_sma3nz_stddevNorm', 'hammarbergIndexV_sma3nz_amean',
       'slopeV500-1500_sma3nz_stddevNorm', 'alphaRatioUV_sma3nz_amean',
       'slopeUV500-1500_sma3nz_amean', 'loudnessPeaksPerSec',
       'VoicedSegmentsPerSec', 'MeanVoicedSegmentLengthSec',
       'StddevVoicedSegmentLengthSec', 'MeanUnvoicedSegmentLength',
       'StddevUnvoicedSegmentLength']
 available_workers = ["Loudness_sma3","mfcc","gammatone", "melfs", "F0final_sma",
                        "pcm_RMSenergy_sma",
                        "voicingFinalUnclipped_sma",
                        "logHNR_sma", "alphaRatio_sma3",
                        "audspecRasta_lengthL1norm_sma",
                        "pcm_zcr_sma", 
                        "audspec_lengthL1norm_sma0"] 
