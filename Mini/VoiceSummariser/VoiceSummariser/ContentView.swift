//
//  ContentView.swift
//  VoiceSummariser
//
//  Created by Ekamveer Singh on 08/06/2026.
//

import SwiftUI
import AVFoundation

class AudioRecorder: NSObject {
    var audioRecorder: AVAudioRecorder?
    var recordingURL: URL?

    func requestPermissionAndStart(completion: @escaping (Bool) -> Void) {
        #if os(macOS)
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            completion(true)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .audio) { granted in
                DispatchQueue.main.async { completion(granted) }
            }
        default:
            completion(false)
        }
        #else
        AVAudioSession.sharedInstance().requestRecordPermission { granted in
            DispatchQueue.main.async { completion(granted) }
        }
        #endif
    }

    func startRecording() {
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.record, mode: .default)
            try session.setActive(true)
        } catch {
            print("Audio session setup failed: \(error)")
            return
        }
        #endif

        let tmpDir = FileManager.default.temporaryDirectory
        let fileURL = tmpDir.appendingPathComponent("recording.m4a")
        recordingURL = fileURL

        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatMPEG4AAC),
            AVSampleRateKey: 16000,
            AVNumberOfChannelsKey: 1,
            AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
        ]

        do {
            audioRecorder = try AVAudioRecorder(url: fileURL, settings: settings)
            audioRecorder?.delegate = self
            let prepared = audioRecorder?.prepareToRecord() ?? false
            print("Prepared to record: \(prepared)")
            let started = audioRecorder?.record() ?? false
            print("Recording started: \(started)")
        } catch {
            print("Failed to start recording: \(error)")
        }
    }

    func stopRecording() -> URL? {
        audioRecorder?.stop()
        #if os(iOS)
        try? AVAudioSession.sharedInstance().setActive(false)
        #endif
        return recordingURL
    }
}

extension AudioRecorder: AVAudioRecorderDelegate {
    func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
        print("Recording finished, success: \(flag)")
    }
    func audioRecorderEncodeErrorDidOccur(_ recorder: AVAudioRecorder, error: Error?) {
        print("Encode error: \(String(describing: error))")
    }
}

struct ContentView: View {
    @State private var recorder = AudioRecorder()
    @State private var isRecording = false
    @State private var transcription = ""
    @State private var summary = ""
    @State private var isLoading = false
    @State private var errorMessage = ""

    var body: some View {
        VStack(spacing: 24) {
            Text("Voice Summariser")
                .font(.largeTitle)
                .fontWeight(.bold)

            Button(action: toggleRecording) {
                Label(
                    isRecording ? "Stop Recording" : "Start Recording",
                    systemImage: isRecording ? "stop.circle.fill" : "mic.circle.fill"
                )
                .font(.title2)
                .padding()
                .background(isRecording ? Color.red.opacity(0.15) : Color.blue.opacity(0.15))
                .cornerRadius(12)
            }
            .disabled(isLoading)

            if isLoading {
                ProgressView("Processing...")
            }

            if !transcription.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Transcription")
                        .font(.headline)
                    Text(transcription)
                        .font(.body)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(8)
                }
            }

            if !summary.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Summary")
                        .font(.headline)
                    Text(summary)
                        .font(.body)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                        .background(Color.blue.opacity(0.08))
                        .cornerRadius(8)
                }
            }

            if !errorMessage.isEmpty {
                Text(errorMessage)
                    .foregroundStyle(.red)
                    .font(.caption)
            }

            Spacer()
        }
        .padding(32)
        .frame(minWidth: 500, minHeight: 500)
    }

    func toggleRecording() {
        if isRecording {
            isRecording = false
            guard let url = recorder.stopRecording() else { return }
            sendToBackend(fileURL: url)
        } else {
            recorder.requestPermissionAndStart { granted in
                guard granted else {
                    self.errorMessage = "Microphone permission denied."
                    return
                }
                self.transcription = ""
                self.summary = ""
                self.errorMessage = ""
                self.isRecording = true
                self.recorder.startRecording()
            }
        }
    }

    func sendToBackend(fileURL: URL) {
        isLoading = true
        let endpoint = URL(string: "http://localhost:8000/transcribe-and-summarise")!
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"

        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()
        let filename = fileURL.lastPathComponent
        let mimeType = "audio/m4a"
        guard let fileData = try? Data(contentsOf: fileURL) else {
            errorMessage = "Could not read recorded audio file."
            isLoading = false
            return
        }

        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"audio\"; filename=\"\(filename)\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: \(mimeType)\r\n\r\n".data(using: .utf8)!)
        body.append(fileData)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)

        request.httpBody = body

        URLSession.shared.dataTask(with: request) { data, _, error in
            DispatchQueue.main.async {
                isLoading = false
                if let error = error {
                    errorMessage = "Network error: \(error.localizedDescription)"
                    return
                }
                guard let data = data,
                      let json = try? JSONSerialization.jsonObject(with: data) as? [String: String] else {
                    errorMessage = "Invalid response from server"
                    return
                }
                transcription = json["transcription"] ?? ""
                summary = json["summary"] ?? ""
            }
        }.resume()
    }
}
