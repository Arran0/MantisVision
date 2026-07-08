"use client";

import { useEffect, useRef, useState } from "react";
import type { PredictionResult } from "@/lib/types";
import { ResultCard } from "@/components/ResultCard";

// Give up on a stuck inference request instead of showing "Analysing…" forever.
// Kept above the API route's own timeout so a slow first response (e.g. a
// cold-starting inference host) is waited out rather than cut off here.
const REQUEST_TIMEOUT_MS = 62_000;

type Selection = { blob: Blob; filename: string; url: string };

export function UploadCard() {
  const [selection, setSelection] = useState<Selection | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cameraActive, setCameraActive] = useState(false);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const selectionRef = useRef<Selection | null>(null);
  selectionRef.current = selection;

  // Turn the camera off and revoke any object URL if the user navigates away.
  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      if (selectionRef.current) URL.revokeObjectURL(selectionRef.current.url);
    };
  }, []);

  function chooseImage(blob: Blob, filename: string) {
    // Stage the image for review — analysis only starts when the user taps Analyse.
    if (selection) URL.revokeObjectURL(selection.url);
    setSelection({ blob, filename, url: URL.createObjectURL(blob) });
    setResult(null);
    setError(null);
  }

  function reset() {
    if (selection) URL.revokeObjectURL(selection.url);
    setSelection(null);
    setResult(null);
    setError(null);
    setLoading(false);
  }

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
    canvas.toBlob((blob) => blob && chooseImage(blob, "capture.jpg"), "image/jpeg", 0.92);
  }

  async function analyze() {
    if (!selection) return;
    setResult(null);
    setError(null);
    setLoading(true);

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    try {
      const formData = new FormData();
      formData.append("file", selection.blob, selection.filename);

      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });

      const payload = await response.json().catch(() => null);

      if (!response.ok) {
        throw new Error(
          payload?.error ?? payload?.detail ?? `Prediction failed (HTTP ${response.status}).`
        );
      }
      if (!payload) throw new Error("The inference service returned an unexpected response.");

      setResult(payload as PredictionResult);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setError(
          "The inference service didn't respond in time. Confirm ML_API_URL points to a reachable, running service."
        );
      } else {
        setError(err instanceof Error ? err.message : "Something went wrong.");
      }
    } finally {
      clearTimeout(timer);
      setLoading(false);
    }
  }

  function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    chooseImage(file, file.name);
    event.target.value = ""; // allow re-selecting the same file
  }

  // ---- Camera view -------------------------------------------------------
  if (cameraActive) {
    return (
      <div className="w-full">
        <div className="mv-card overflow-hidden p-3">
          {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
          <video ref={videoRef} autoPlay playsInline muted className="w-full rounded-2xl bg-black" />
          <div className="mt-3 flex gap-3">
            <button type="button" onClick={capturePhoto} className="mv-btn-primary flex-1">
              Capture
            </button>
            <button type="button" onClick={stopCamera} className="mv-btn-secondary">
              Cancel
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ---- Review + analyse --------------------------------------------------
  if (selection) {
    return (
      <div className="flex w-full flex-col gap-5">
        <div className="mv-card overflow-hidden">
          <img
            src={selection.url}
            alt="Selected specimen"
            className="max-h-96 w-full bg-slate-100 object-contain"
          />
        </div>

        {error && (
          <div className="rounded-2xl border border-red-100 bg-red-50/80 px-4 py-3 text-center text-sm text-red-700 backdrop-blur">
            {error}
          </div>
        )}

        {loading ? (
          <div className="mv-card flex items-center justify-center gap-3 px-6 py-5 text-slate-600">
            <Spinner />
            <span className="font-medium">Analysing specimen…</span>
          </div>
        ) : result ? (
          <button type="button" onClick={reset} className="mv-btn-secondary w-full">
            Scan another photo
          </button>
        ) : (
          <div className="flex flex-col gap-3 sm:flex-row">
            <button type="button" onClick={analyze} className="mv-btn-primary flex-1">
              {error ? "Try again" : "Analyse"}
            </button>
            <button type="button" onClick={reset} className="mv-btn-secondary sm:w-auto">
              Choose another
            </button>
          </div>
        )}

        {result && <ResultCard result={result} />}
      </div>
    );
  }

  // ---- Empty state (pick a photo) ---------------------------------------
  return (
    <div className="w-full">
      <div className="mv-card flex flex-col items-center gap-5 px-6 py-10 text-center">
        <span className="text-xl font-semibold tracking-tight text-slate-800">
          Identify a seaweed specimen
        </span>
        <span className="max-w-xs text-sm text-slate-500">
          Take a photo or upload one, then tap Analyse to get species and health results.
        </span>

        <div className="flex w-full flex-col gap-3 sm:flex-row sm:justify-center">
          <button type="button" onClick={openCamera} className="mv-btn-primary">
            Open camera
          </button>
          <label className="mv-btn-secondary cursor-pointer">
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

      {error && <p className="mt-4 text-center text-sm text-red-600">{error}</p>}
    </div>
  );
}

function Spinner() {
  return (
    <svg className="h-5 w-5 animate-spin text-ocean-500" viewBox="0 0 24 24" fill="none" aria-hidden>
      <circle className="opacity-20" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path
        className="opacity-90"
        fill="currentColor"
        d="M4 12a8 8 0 0 1 8-8V0C5.4 0 0 5.4 0 12h4z"
      />
    </svg>
  );
}
