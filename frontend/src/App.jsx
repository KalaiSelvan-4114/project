import React, { useEffect, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:5000";
const IMG_FALLBACK_WIDTH = 640;
const IMG_FALLBACK_HEIGHT = 480;

function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const fileInputRef = useRef(null);
  const cameraInputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [cameraStream, setCameraStream] = useState(null);

  const handleFileChange = (e) => {
    const selected = e.target.files?.[0];
    setError("");
    setResult(null);
    if (!selected) {
      setFile(null);
      setPreviewUrl(null);
      return;
    }
    setFile(selected);
    setPreviewUrl(URL.createObjectURL(selected));
  };

  const handleClear = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError("");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    if (cameraInputRef.current) {
      cameraInputRef.current.value = "";
    }

    // Stop camera if open
    if (cameraStream) {
      cameraStream.getTracks().forEach((t) => t.stop());
      setCameraStream(null);
      setIsCameraOpen(false);
    }
  };

  const openFilePicker = () => {
    fileInputRef.current?.click();
  };

  const openCamera = async () => {
    setError("");

    // Prefer getUserMedia for real camera capture
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: "environment" } },
        });
        setCameraStream(stream);
        setIsCameraOpen(true);
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          // play will be triggered by browser automatically in most cases
        }
        return;
      } catch (err) {
        console.error("Camera access error:", err);
        setError(
          "Unable to access camera. Please allow camera permission or upload from device."
        );
      }
    }

    // Fallback: open native camera/file picker if available
    cameraInputRef.current?.click();
  };

  const closeCamera = () => {
    if (cameraStream) {
      cameraStream.getTracks().forEach((t) => t.stop());
      setCameraStream(null);
    }
    setIsCameraOpen(false);
  };

  const captureFromCamera = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const width = video.videoWidth || IMG_FALLBACK_WIDTH;
    const height = video.videoHeight || IMG_FALLBACK_HEIGHT;
    if (!width || !height) return;

    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, width, height);

    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        const capturedFile = new File([blob], "camera_capture.png", {
          type: "image/png",
        });
        const url = URL.createObjectURL(blob);

        setFile(capturedFile);
        setPreviewUrl(url);
        setResult(null);
        setError("");
        closeCamera();
      },
      "image/png",
      0.95
    );
  };

  // Clean up camera stream on unmount
  useEffect(() => {
    return () => {
      if (cameraStream) {
        cameraStream.getTracks().forEach((t) => t.stop());
      }
    };
  }, [cameraStream]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Please upload a skin image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok || data.status === "error") {
        setError(data.message || "Something went wrong. Please try again.");
        setLoading(false);
        return;
      }

      setResult(data);
    } catch (err) {
      console.error(err);
      setError("Unable to contact the server. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center px-4 py-6">
      <div className="w-full max-w-5xl bg-slate-900/70 border border-slate-700 rounded-2xl shadow-2xl p-6 sm:p-8">
        <header className="text-center mb-6">
          <h1 className="text-2xl sm:text-3xl font-semibold text-slate-50">
            Skin Disease Detection Using EfficientNet
          </h1>
          <p className="mt-2 text-sm text-slate-300">
            Upload a clear photo of the affected skin area. The model will detect if a
            disease is present and highlight the affected regions.
          </p>
        </header>

        <form
          onSubmit={handleSubmit}
          className="flex flex-col gap-4 sm:flex-row sm:items-start"
        >
          <div className="flex-1 space-y-3">
            <div>
              <span className="block text-sm font-medium text-slate-200 mb-2">
                Provide skin image
              </span>
              <div className="flex flex-col sm:flex-row gap-2">
                <button
                  type="button"
                  onClick={openFilePicker}
                  className="flex-1 inline-flex items-center justify-center px-3 py-2 rounded-lg border border-slate-600 bg-slate-800 text-sm font-medium text-slate-100 hover:border-sky-500 hover:bg-slate-700"
                >
                  Choose from device
                </button>
                <button
                  type="button"
                  onClick={openCamera}
                  className="flex-1 inline-flex items-center justify-center px-3 py-2 rounded-lg border border-slate-600 bg-slate-800 text-sm font-medium text-slate-100 hover:border-sky-500 hover:bg-slate-700"
                >
                  Capture using camera
                </button>
              </div>
              <p className="mt-2 text-xs text-slate-400">
                Recommended: well-lit, high-resolution image focusing on the affected skin
                region.
              </p>

              {/* Hidden inputs for file upload and camera capture */}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleFileChange}
              />
              <input
                ref={cameraInputRef}
                type="file"
                accept="image/*"
                capture="environment"
                className="hidden"
                onChange={handleFileChange}
              />
            </div>

            <div className="flex flex-wrap gap-2">
              <button
                type="submit"
                disabled={loading || !file}
                className="inline-flex items-center justify-center px-4 py-2 rounded-lg bg-sky-600 hover:bg-sky-500 disabled:bg-sky-700 text-sm font-semibold text-white"
              >
                {loading ? "Analyzing..." : "Analyze Image"}
              </button>
              <button
                type="button"
                onClick={handleClear}
                disabled={!file && !previewUrl && !result}
                className="inline-flex items-center justify-center px-4 py-2 rounded-lg border border-slate-600 text-sm font-semibold text-slate-100 hover:bg-slate-800 disabled:opacity-40"
              >
                Delete image & result
              </button>
            </div>

            {error && (
              <p className="text-sm text-red-400 font-medium">{error}</p>
            )}
          </div>

          <div className="flex-1 mt-4 sm:mt-0">
            {previewUrl && (
              <div>
                <p className="text-sm font-medium text-slate-200 mb-2">
                  Input image preview
                </p>
                <div className="aspect-square w-full max-w-xs mx-auto rounded-xl overflow-hidden border border-slate-700 bg-slate-800">
                  <img
                    src={previewUrl}
                    alt="Preview"
                    className="h-full w-full object-cover"
                  />
                </div>
              </div>
            )}
          </div>
        </form>

        {result && (
          <section className="mt-8">
            <div className="flex flex-wrap items-center gap-3 mb-4">
              <span className="inline-flex items-center rounded-full bg-slate-800 px-3 py-1 text-sm font-semibold text-slate-100">
                Disease Detected:&nbsp;
                {result.is_healthy ? "Healthy Skin" : result.prediction}
              </span>
              <span className="inline-flex items-center rounded-full bg-emerald-900/70 px-3 py-1 text-xs font-medium text-emerald-200">
                Confidence: {result.confidence?.toFixed(2)}%
              </span>
            </div>

            <p className="text-sm text-slate-200 mb-4">{result.message}</p>

            {!result.is_healthy && (
              <div className="grid gap-4 md:grid-cols-3">
                <div className="flex flex-col items-center">
                  <p className="text-sm font-medium text-slate-200 mb-2">
                    Input Image
                  </p>
                  <div className="aspect-square w-full rounded-xl overflow-hidden border border-slate-700 bg-slate-800">
                    <img
                      src={result.original_image}
                      alt="Input"
                      className="h-full w-full object-cover"
                    />
                  </div>
                </div>
                <div className="flex flex-col items-center">
                  <p className="text-sm font-medium text-slate-200 mb-2">
                    Segmentation (Disease Only)
                  </p>
                  <div className="aspect-square w-full rounded-xl overflow-hidden border border-slate-700 bg-slate-800">
                    <img
                      src={result.segmentation_image}
                      alt="Segmentation"
                      className="h-full w-full object-cover"
                    />
                  </div>
                </div>
                <div className="flex flex-col items-center">
                  <p className="text-sm font-medium text-slate-200 mb-2">
                    Disease Area Detected
                  </p>
                  <div className="aspect-square w-full rounded-xl overflow-hidden border border-slate-700 bg-slate-800">
                    <img
                      src={result.disease_overlay_image}
                      alt="Disease area"
                      className="h-full w-full object-cover"
                    />
                  </div>
                </div>
              </div>
            )}
          </section>
        )}

        <footer className="mt-8 text-center text-xs text-slate-500">
          This tool is for educational support only and is not a replacement for
          professional medical diagnosis. Please consult a dermatologist for clinical
          decisions.
        </footer>
      </div>

      {isCameraOpen && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/70 px-4">
          <div className="w-full max-w-md bg-slate-900 border border-slate-700 rounded-2xl p-4 space-y-3">
            <p className="text-sm font-medium text-slate-100">Camera preview</p>
            <div className="aspect-[3/4] w-full rounded-xl overflow-hidden bg-black">
              <video
                ref={videoRef}
                className="w-full h-full object-cover"
                autoPlay
                playsInline
                muted
              />
            </div>
            <div className="flex justify-end gap-2">
              <button
                type="button"
                onClick={closeCamera}
                className="px-3 py-2 text-sm rounded-lg border border-slate-600 text-slate-100 hover:bg-slate-800"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={captureFromCamera}
                className="px-3 py-2 text-sm rounded-lg bg-sky-600 hover:bg-sky-500 text-white font-semibold"
              >
                Capture
              </button>
            </div>
            <canvas ref={canvasRef} className="hidden" />
          </div>
        </div>
      )}
    </div>
  );
}

export default App;

