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

  function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    chooseImage(file, file.name);
    event.target.value = ""; // allow re-selecting the same file
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

          {/* Image holder — placeholder, staged photo, or live camera. */}
          <div className="relative flex-1 min-h-0 overflow-hidden rounded-[1.5rem] border border-white/50 bg-ocean-50/40">
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
            <div className="flex flex-col gap-3">
              <div className="flex gap-3">
                <label className="mv-btn-orange flex-1 cursor-pointer">
                  Upload photo
                  <input
                    type="file"
                    accept="image/*"
                    capture="environment"
                    className="hidden"
                    onChange={handleFileChange}
                  />
                </label>
                <button type="button" onClick={openCamera} className="mv-btn-blue flex-1">
                  Open camera
                </button>
              </div>

              <button
                type="button"
                onClick={analyze}
                disabled={!selection || loading}
                className="mv-btn-blue w-full"
              >
                {loading ? "Analysing…" : error ? "Try again" : "Analyse"}
              </button>

              {(selection || revealed) && (
                <button type="button" onClick={reset} className="mv-btn-orange w-full">
                  <PlusIcon />
                  Scan another
                </button>
              )}
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

function PlusIcon() {
  return (
    <svg className="h-4 w-4" viewBox="0 0 20 20" fill="none" aria-hidden>
      <path d="M10 4v12M4 10h12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}
