import torch
import torchaudio
from pathlib import Path
from speechbrain.inference.speaker import SpeakerRecognition

# ----------
# CONSTANTS
# ----------

SAMPLE_RATE = 16_000

# -----------
# VOICE GATE
# -----------

class VoiceGate:
    def __init__(self, voiceprint_path="voiceprint.pt", threshold=0.30,
                 model_source="speechbrain/spkrec-ecapa-voxceleb",
                 savedir="pretrained_models/spkrec-ecapa-voxceleb"):
        
        self.model = SpeakerRecognition.from_hparams(source=model_source, savedir=savedir)
        self.voiceprint_path = Path(voiceprint_path)
        self.threshold = threshold
        self.voiceprint = torch.load(self.voiceprint_path) if self.voiceprint_path.exists() else None

    def _load_file(self, path):
        return self.model.load_audio(str(path))
 
    def _prepare_array(self, waveform, sample_rate):
        if not torch.is_tensor(waveform):
            waveform = torch.as_tensor(waveform, dtype=torch.float32)
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
        return waveform
    
    def _embed(self, waveform):
        embedding = self.model.encode_batch(waveform.unsqueeze(0))
        return embedding.squeeze()
    
    def enroll(self, wav_paths):
        embeddings = [self._embed(self._load_file(p)) for p in wav_paths]
        stacked = torch.stack(embeddings)
        voiceprint = torch.nn.functional.normalize(stacked.mean(dim=0), dim=0)
        torch.save(voiceprint, self.voiceprint_path)
        self.voiceprint = voiceprint
        return voiceprint
    
    def verify_waveform(self, waveform, sample_rate=SAMPLE_RATE):
        if self.voiceprint is None:
            raise RuntimeError("No voiceprint enrolled yet. Call enroll() first.")
        waveform = self._prepare_array(waveform, sample_rate)
        embedding = torch.nn.functional.normalize(self._embed(waveform), dim=0)
        score = torch.dot(embedding, self.voiceprint).item()
        return score, score >= self.threshold
 
    def verify_file(self, path):
        return self.verify_waveform(self._load_file(path), SAMPLE_RATE)