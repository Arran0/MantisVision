"use client";

import { useEffect, useRef, useState } from "react";
import type { PredictionResult } from "@/lib/types";
import { ResultCard } from "@/components/ResultCard";

// Give up on a stuck inference request instead of showing "Analysing…" forever.
// Kept above the API route's own timeout so a slow first response (e.g. a
// cold-starting inference host) is waited out rather than cut off here.
const REQUEST_TIMEOUT_MS = 62_000;

// The model resizes to a fixed 224x224 input anyway, so there's no benefit to
// uploading (and proxying through two hops) a full camera-resolution photo.
// This cap keeps plenty of headroom above that for the Grad-CAM overlay
// while cutting multi-MB phone photos down to a few hundred KB.
const MAX_UPLOAD_DIMENSION = 1024;
const UPLOAD_JPEG_QUALITY = 0.85;

type Selection = { blob: Blob; filename: string; url: string };

// Downscales + re-encodes as JPEG so large camera-roll photos don't dominate
// upload time. No-ops (aside from re-encoding) on images already within bounds.
async function resizeForUpload(blob: Blob): Promise<Blob> {
  const bitmap = await createImageBitmap(blob);
  try {
    const scale = Math.min(1, MAX_UPLOAD_DIMENSION / Math.max(bitmap.width, bitmap.height));
    const width = Math.round(bitmap.width * scale);
    const height = Math.round(bitmap.height * scale);

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return blob;
    ctx.drawImage(bitmap, 0, 0, width, height);

    const resized = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", UPLOAD_JPEG_QUALITY)
    );
    return resized ?? blob;
  } finally {
    bitmap.close();
  }
}

export function UploadCard() {
  const [selection, setSelection] = useState<Selection | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cameraActive, setCameraActive] = useState(false);
  // Drives the two-card slide: collapsed (explainer hidden behind collector)
  // until Analyse is tapped, then the cards separate.
  const [revealed, setRevealed] = useState(false);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const selectionRef = useRef<Selection | null>(null);
  selectionRef.current = selection;

  // Bumped on every analyze()/reset() so a request that's still in flight when
  // the user resets or starts a new one can't clobber state with a stale result.
  const requestIdRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);

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
    setRevealed(false);
  }

  function reset() {
    requestIdRef.current += 1;
    abortRef.current?.abort();
    abortRef.current = null;
    if (selection) URL.revokeObjectURL(selection.url);
    setSelection(null);
    setResult(null);
    setError(null);
    setLoading(false);
    setRevealed(false);
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

  // Swirl the background blobs while a scan is in flight; they ease back on stop.
  useEffect(() => {
    document.body.classList.toggle("mv-analyzing", loading);
    return () => document.body.classList.remove("mv-analyzing");
  }, [loading]);

  function capturePhoto() {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return;

    const scale = Math.min(1, MAX_UPLOAD_DIMENSION / Math.max(video.videoWidth, video.videoHeight));
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    stopCamera();
    canvas.toBlob(
      (blob) => blob && chooseImage(blob, "capture.jpg"),
      "image/jpeg",
      UPLOAD_JPEG_QUALITY
    );
  }

  async function analyze() {
    if (!selection) return;

    const id = ++requestIdRef.current;
    setResult(null);
    setError(null);
    setLoading(true);
    setRevealed(true); // slide the explainer card into view

    const controller = new AbortController();
    abortRef.current = controller;
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
      if (requestIdRef.current !== id) return; // superseded by a reset/new request

      if (!response.ok) {
        throw new Error(
          payload?.error ?? payload?.detail ?? `Prediction failed (HTTP ${response.status}).`
        );
      }
      if (!payload) throw new Error("The inference service returned an unexpected response.");

      setResult(payload as PredictionResult);
    } catch (err) {
      if (requestIdRef.current !== id) return; // superseded — ignore stale error too
      if (err instanceof DOMException && err.name === "AbortError") {
        setError(
          "The inference service didn't respond in time. Confirm ML_API_URL points to a reachable, running service."
        );
      } else {
        setError(err instanceof Error ? err.message : "Something went wrong.");
      }
    } finally {
      clearTimeout(timer);
      if (requestIdRef.current === id) setLoading(false);
    }
  }

  async function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = ""; // allow re-selecting the same file
    if (!file) return;
    const resized = await resizeForUpload(file);
    chooseImage(resized, file.name);
  }

  return (
    <div className="mv-stage" data-revealed={revealed}>
      {/* ---- Collector card -------------------------------------------- */}
      <div className="mv-slot mv-slot-collector">
        <section className="mv-card h-full w-full">
        <div className="flex h-full flex-col gap-4 p-5 sm:p-6">
          <p className="text-center text-sm font-semibold text-slate-500">
            Let&rsquo;s analyse a seaweed!
          </p>

          {/* Image holder — placeholder, staged photo, or live camera.
              Background matches the card so the placeholder blends in. */}
          <div className="relative flex-1 min-h-0 overflow-hidden rounded-[1.5rem] border border-white/60 bg-white/60">
            {cameraActive ? (
              // eslint-disable-next-line jsx-a11y/media-has-caption
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="h-full w-full bg-black object-cover"
              />
            ) : selection ? (
              <>
                <img
                  src={selection.url}
                  alt="Selected specimen"
                  className="h-full w-full object-contain"
                />
                <button
                  type="button"
                  onClick={reset}
                  aria-label="Start over with a new photo"
                  title="Start over"
                  className="mv-icon-btn absolute right-3 top-3"
                >
                  <CloseIcon />
                </button>
              </>
            ) : (
              <img
                src="/seaweed-placeholder.svg"
                alt=""
                aria-hidden
                className="h-full w-full object-contain p-6 opacity-90"
              />
            )}
          </div>

          {/* Actions */}
          {cameraActive ? (
            <div className="flex gap-3">
              <button type="button" onClick={capturePhoto} className="mv-btn-blue flex-1">
                Capture
              </button>
              <button type="button" onClick={stopCamera} className="mv-btn-orange">
                Cancel
              </button>
            </div>
          ) : (
            <div className="flex gap-2">
              <label className="mv-btn-orange mv-btn-sm flex-1 cursor-pointer">
                <UploadIcon />
                <span>Upload</span>
                <input
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={handleFileChange}
                />
              </label>
              <button type="button" onClick={openCamera} className="mv-btn-blue mv-btn-sm flex-1">
                <CameraIcon />
                <span>Camera</span>
              </button>
              <button
                type="button"
                onClick={analyze}
                disabled={!selection || loading}
                className="mv-btn-blue mv-btn-sm flex-1"
              >
                {!loading && <AnalyseIcon />}
                <span>{loading ? "Analysing…" : error ? "Try again" : "Analyse"}</span>
              </button>
            </div>
          )}
        </div>
        </section>
      </div>

      {/* ---- Explainer card -------------------------------------------- */}
      <div className="mv-slot mv-slot-explainer" aria-hidden={!revealed}>
        <section className="mv-card h-full w-full">
        <div className="h-full overflow-y-auto p-5 sm:p-6">
          {error ? (
            <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
              <span className="text-lg font-semibold text-coral-600">Analysis failed</span>
              <p className="max-w-xs text-sm text-slate-600">{error}</p>
            </div>
          ) : loading ? (
            <div className="flex h-full flex-col items-center justify-center gap-4 text-slate-600">
              <div className="mv-dots" aria-hidden>
                <span />
                <span />
                <span />
              </div>
              <span className="font-medium">Analysing specimen…</span>
            </div>
          ) : result ? (
            <ResultCard result={result} />
          ) : null}
        </div>
        </section>
      </div>
    </div>
  );
}

function CloseIcon() {
  return (
    <svg className="h-4 w-4" viewBox="0 0 20 20" fill="none" aria-hidden>
      <path d="M5 5l10 10M15 5L5 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg className="h-4 w-4 shrink-0" viewBox="0 0 20 20" fill="none" aria-hidden>
      <path
        d="M10 13V4M10 4l-3.5 3.5M10 4l3.5 3.5"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M4 13.5v1.25A1.25 1.25 0 0 0 5.25 16h9.5A1.25 1.25 0 0 0 16 14.75V13.5"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function CameraIcon() {
  return (
    <svg className="h-4 w-4 shrink-0" viewBox="0 0 20 20" fill="none" aria-hidden>
      <path
        d="M3 7.25A1.25 1.25 0 0 1 4.25 6h1.4l.72-1.2A1.25 1.25 0 0 1 7.44 4.2h5.12a1.25 1.25 0 0 1 1.07.6l.72 1.2h1.4A1.25 1.25 0 0 1 17 7.25v7.5A1.25 1.25 0 0 1 15.75 16H4.25A1.25 1.25 0 0 1 3 14.75Z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
      <circle cx="10" cy="10.5" r="2.6" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  );
}

function AnalyseIcon() {
  return (
    <svg className="h-4 w-4 shrink-0" viewBox="0 0 20 20" fill="none" aria-hidden>
      <circle cx="8.5" cy="8.5" r="5" stroke="currentColor" strokeWidth="1.6" />
      <path d="M15.5 15.5 12.4 12.4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}

