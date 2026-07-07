"use client";

import { useEffect, useRef, useState } from "react";
import type { PredictionResult } from "@/lib/types";
import { ResultCard } from "@/components/ResultCard";

export function UploadCard() {
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cameraActive, setCameraActive] = useState(false);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Make sure the camera light turns off if the user navigates away mid-stream.
  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  function stopCamera() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setCameraActive(false);
  }

  async function openCamera() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false,
      });
      streamRef.current = stream;
      setCameraActive(true);
    } catch {
      setError("Couldn't access the camera. Check camera permissions, or upload a photo instead.");
    }
  }

  // Attach the stream once the <video> element mounts (it only exists while cameraActive is true).
  useEffect(() => {
    if (cameraActive && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
    }
  }, [cameraActive]);

  function capturePhoto() {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    stopCamera();
    canvas.toBlob((blob) => blob && submitImage(blob, "capture.jpg"), "image/jpeg", 0.92);
  }

  async function submitImage(file: Blob, filename: string) {
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", file, filename);

      const response = await fetch("/api/predict", { method: "POST", body: formData });
      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload.error ?? payload.detail ?? "Prediction failed.");
      }

      setResult(payload as PredictionResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    submitImage(file, file.name);
  }

  return (
    <div className="mx-auto flex w-full max-w-xl flex-col gap-6">
      {cameraActive ? (
        <div className="flex flex-col gap-3 rounded-2xl border-2 border-seaweed-500 bg-black p-3">
          {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
          <video ref={videoRef} autoPlay playsInline muted className="w-full rounded-xl" />
          <div className="flex gap-3">
            <button
              type="button"
              onClick={capturePhoto}
              className="flex-1 rounded-xl bg-seaweed-500 py-3 font-medium text-white transition hover:bg-seaweed-600"
            >
              Capture
            </button>
            <button
              type="button"
              onClick={stopCamera}
              className="rounded-xl border border-white/40 px-4 py-3 font-medium text-white transition hover:bg-white/10"
            >
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-4 rounded-2xl border-2 border-dashed border-seaweed-500 bg-white p-8 text-center transition hover:bg-seaweed-50">
          <span className="text-lg font-medium text-seaweed-900">
            Photograph a Kappaphycus alvarezii specimen
          </span>
          <span className="text-sm text-slate-500">Open the camera, or upload an existing photo</span>

          <div className="flex w-full flex-col gap-2 sm:flex-row sm:justify-center">
            <button
              type="button"
              onClick={openCamera}
              className="rounded-xl bg-seaweed-500 px-5 py-3 font-medium text-white transition hover:bg-seaweed-600"
            >
              Open camera
            </button>
            <label className="cursor-pointer rounded-xl border border-seaweed-500 px-5 py-3 font-medium text-seaweed-900 transition hover:bg-seaweed-50">
              Upload photo
              <input
                type="file"
                accept="image/*"
                capture="environment"
                className="hidden"
                onChange={handleFileChange}
              />
            </label>
          </div>
        </div>
      )}

      {preview && !cameraActive && (
        <img src={preview} alt="Captured specimen" className="max-h-80 w-full rounded-xl object-contain" />
      )}

      {loading && <p className="text-center text-slate-500">Analyzing...</p>}
      {error && <p className="text-center text-red-600">{error}</p>}
      {result && <ResultCard result={result} />}
    </div>
  );
}
