#!/bin/bash
# vm/fix_face_enrollment_ui.sh
# Updates the FaceEnrollButton component to:
#   1. Show the Jetson C++ sidecar as the enrollment method when online
#   2. Fall back to photo upload when Jetson is offline
#   3. Show enrollment status clearly per employee
#   4. Add "Enroll from Camera" button that triggers the C++ enroll server

set -e
PROJECT="$HOME/FRS_/FRS--Java-Verison"
cd "$PROJECT"

JETSON_IP="172.18.3.202"
JETSON_PORT="5000"

echo "=================================================="
echo " Updating Face Enrollment UI"
echo "=================================================="

# ── Write the updated FaceEnrollButton.tsx ────────────────────────────────────
cat > src/app/components/hr/FaceEnrollButton.tsx << 'TSXEOF'
import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { Progress } from '../ui/progress';
import { cn } from '../ui/utils';
import { lightTheme } from '../../../theme/lightTheme';
import { toast } from 'sonner';
import {
  Camera, Upload, ScanFace, Trash2, X, Info, Loader2,
  CheckCircle2, AlertTriangle, Wifi, WifiOff, RefreshCw,
} from 'lucide-react';
import { authConfig } from '../../config/authConfig';
import { useAuth } from '../../contexts/AuthContext';

export interface EnrollmentEmbedding {
  id: string;
  modelVersion?: string;
  qualityScore?: number;
  isPrimary?: boolean;
  enrolledAt?: string;
}

export interface EnrollmentStatus {
  enrolled: boolean;
  embeddingCount: number;
  embeddings: EnrollmentEmbedding[];
}

interface FaceEnrollButtonProps {
  employeeId: string;
  employeeName: string;
  enrolled?: boolean;
  onEnrolled?: (newStatus: EnrollmentStatus) => void;
  compact?: boolean;
}

type PanelState = 'idle' | 'selecting' | 'preview' | 'uploading' | 'enrolling-cam' | 'success' | 'error';

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
const MAX_SIZE_BYTES = 5 * 1024 * 1024;
const JETSON_SIDECAR = 'http://172.18.3.202:5000';

export const FaceEnrollButton: React.FC<FaceEnrollButtonProps> = ({
  employeeId, employeeName, enrolled, onEnrolled, compact,
}) => {
  const { accessToken } = useAuth();
  const [panelState, setPanelState] = useState<PanelState>('idle');
  const [isEnrolled, setIsEnrolled] = useState(authConfig.mode === 'mock' ? false : !!enrolled);
  const [status, setStatus] = useState<EnrollmentStatus | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [jetsonOnline, setJetsonOnline] = useState<boolean | null>(null);
  const [cameraEnrollCam, setCameraEnrollCam] = useState('entrance-cam-01');

  const inputRef = useRef<HTMLInputElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Check if Jetson sidecar is reachable
  const checkJetson = useCallback(async () => {
    try {
      const resp = await fetch(`${JETSON_SIDECAR}/health`, { signal: AbortSignal.timeout(3000) });
      setJetsonOnline(resp.ok);
    } catch {
      setJetsonOnline(false);
    }
  }, []);

  const primaryEmbedding = useMemo(() => {
    if (!status?.embeddings?.length) return null;
    return status.embeddings.find(e => e.isPrimary) || status.embeddings[0];
  }, [status]);

  // Load enrollment status on mount
  useEffect(() => {
    if (authConfig.mode === 'mock' || !accessToken) return;
    let mounted = true;
    const load = async () => {
      try {
        const r = await fetch(`${authConfig.apiBaseUrl}/employees/${employeeId}/enroll-face`, {
          headers: { Authorization: `Bearer ${accessToken}` },
        });
        if (!r.ok || !mounted) return;
        const d = await r.json();
        setStatus({ enrolled: !!d.enrolled, embeddingCount: d.embeddingCount || 0, embeddings: d.embeddings || [] });
        setIsEnrolled(!!d.enrolled);
      } catch { /* silent */ }
    };
    load();
    checkJetson();
    return () => { mounted = false; };
  }, [accessToken, employeeId, checkJetson]);

  useEffect(() => {
    return () => { if (previewUrl) URL.revokeObjectURL(previewUrl); };
  }, [previewUrl]);

  useEffect(() => {
    return () => { if (stream) stream.getTracks().forEach(t => t.stop()); };
  }, [stream]);

  // ── Compact badge mode ──────────────────────────────────────────────────────
  if (compact) {
    return isEnrolled ? (
      <Badge className="bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300">
        <ScanFace className="w-3 h-3 mr-1" />Face Enrolled
      </Badge>
    ) : (
      <Badge className="bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-300">
        Not Enrolled
      </Badge>
    );
  }

  const resetToIdle = () => {
    setPanelState('idle'); setSelectedFile(null); setPreviewUrl(null); setErrorMessage(null);
  };

  const validateFile = (file: File) => {
    if (!ACCEPTED_TYPES.includes(file.type)) {
      toast.error('Unsupported file type', { description: 'Upload a JPG, PNG, or WEBP image.' });
      return false;
    }
    if (file.size > MAX_SIZE_BYTES) {
      toast.error('File too large', { description: 'Maximum size is 5 MB.' });
      return false;
    }
    return true;
  };

  const handleFile = (file: File) => {
    if (!validateFile(file)) return;
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setPanelState('preview');
  };

  // ── Enroll from Jetson Camera (C++ runner flow) ─────────────────────────────
  const handleCameraEnroll = async () => {
    if (authConfig.mode === 'mock') {
      toast.success('Mock: enrollment triggered');
      return;
    }
    setPanelState('enrolling-cam');
    setErrorMessage(null);
    try {
      // Trigger the C++ enroll server on Jetson
      const resp = await fetch(`${JETSON_SIDECAR}/enroll`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ employee_id: String(employeeId), cam_id: cameraEnrollCam }),
        signal: AbortSignal.timeout(30000),  // 30s — camera connect + inference
      });

      const data = await resp.json().catch(() => ({}));

      if (resp.ok && data.success) {
        const newStatus: EnrollmentStatus = {
          enrolled: true, embeddingCount: 1, embeddings: [],
        };
        setIsEnrolled(true); setStatus(newStatus); setPanelState('success');
        onEnrolled?.(newStatus);
        toast.success(`${employeeName} enrolled`, {
          description: `Quality: ${data.confidence ? (data.confidence * 100).toFixed(0) + '%' : 'Good'}`,
        });
      } else {
        throw new Error(data.error || data.message || 'Enrollment failed');
      }
    } catch (e: any) {
      const isOffline = e?.name === 'AbortError' || e?.message?.includes('fetch');
      const msg = isOffline
        ? 'Jetson sidecar not reachable. Ensure frs-runner is running on the Jetson.'
        : (e?.message || 'Camera enrollment failed');
      setErrorMessage(msg);
      setPanelState('error');
      setJetsonOnline(false);
    }
  };

  // ── Enroll from photo upload (fallback) ────────────────────────────────────
  const handlePhotoEnroll = async () => {
    if (!selectedFile) return;
    setPanelState('uploading');
    setErrorMessage(null);

    if (authConfig.mode === 'mock') {
      setTimeout(() => {
        const s: EnrollmentStatus = { enrolled: true, embeddingCount: 1, embeddings: [] };
        setIsEnrolled(true); setStatus(s); setPanelState('success'); onEnrolled?.(s);
      }, 1500);
      return;
    }

    try {
      const form = new FormData();
      form.append('photo', selectedFile);
      const resp = await fetch(`${authConfig.apiBaseUrl}/employees/${employeeId}/enroll-face`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${accessToken}` },
        body: form,
      });

      const data = await resp.json().catch(() => ({}));

      if (resp.status === 503) {
        // Sidecar offline — try to give a helpful hint
        throw new Error(
          'The Jetson C++ sidecar is offline. Start the frs-runner service: ' +
          'sudo systemctl start frs-runner'
        );
      }
      if (!resp.ok) {
        const msg = data?.message || 'Enrollment failed';
        if (resp.status === 422) {
          if (msg.toLowerCase().includes('no face')) throw new Error('No face detected — use a clear, front-facing photo');
          if (msg.toLowerCase().includes('multiple')) throw new Error('Multiple faces detected — photo must show only this employee');
        }
        throw new Error(msg);
      }

      const newStatus: EnrollmentStatus = { enrolled: true, embeddingCount: 1, embeddings: [] };
      setIsEnrolled(true); setStatus(newStatus); setPanelState('success'); onEnrolled?.(newStatus);
    } catch (e) {
      setErrorMessage(e instanceof Error ? e.message : 'Enrollment failed');
      setPanelState('error');
    }
  };

  const handleDelete = async () => {
    if (authConfig.mode === 'mock') { setIsEnrolled(false); setStatus(null); resetToIdle(); return; }
    if (!window.confirm(`Remove ${employeeName}'s face enrollment? They will no longer be recognised by cameras.`)) return;
    try {
      const r = await fetch(`${authConfig.apiBaseUrl}/employees/${employeeId}/enroll-face`, {
        method: 'DELETE', headers: { Authorization: `Bearer ${accessToken}` },
      });
      if (!r.ok) throw new Error('Failed to remove enrollment');
      toast.success('Enrollment removed');
      setIsEnrolled(false); setStatus(null); resetToIdle();
    } catch (e) {
      toast.error('Delete failed', { description: e instanceof Error ? e.message : 'Unknown error' });
    }
  };

  const startWebcam = async () => {
    try {
      const media = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } });
      setStream(media); setPanelState('selecting');
      if (videoRef.current) { videoRef.current.srcObject = media; await videoRef.current.play(); }
    } catch { toast.error('Camera unavailable', { description: 'Allow camera access or upload a photo instead.' }); }
  };

  const captureWebcam = async () => {
    const video = videoRef.current, canvas = canvasRef.current;
    if (!video || !canvas) return;
    canvas.width = video.videoWidth || 640; canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (!blob) return;
      handleFile(new File([blob], 'webcam.jpg', { type: 'image/jpeg' }));
      if (stream) { stream.getTracks().forEach(t => t.stop()); setStream(null); }
    }, 'image/jpeg', 0.95);
  };

  return (
    <div className="space-y-4">
      {/* Enrolled status bar */}
      {isEnrolled && panelState === 'idle' && (
        <div className={cn('p-3 rounded-lg border flex items-start gap-3',
          lightTheme.background.secondary, lightTheme.border.default,
          'dark:bg-slate-900/40 dark:border-border')}>
          <CheckCircle2 className="w-4 h-4 text-emerald-500 mt-0.5" />
          <div className="flex-1">
            <p className={cn('text-xs font-bold', lightTheme.text.primary, 'dark:text-white')}>
              Face enrolled
            </p>
            <p className={cn('text-[11px]', lightTheme.text.muted, 'dark:text-slate-500')}>
              {status?.embeddingCount ?? 1} photo(s)
              {primaryEmbedding?.qualityScore ? ` · Quality ${(primaryEmbedding.qualityScore * 100).toFixed(0)}%` : ''}
              {primaryEmbedding?.enrolledAt ? ` · ${new Date(primaryEmbedding.enrolledAt).toLocaleDateString()}` : ''}
            </p>
          </div>
          <Button variant="ghost" size="icon" onClick={handleDelete} className="text-red-500 hover:text-red-600">
            <Trash2 className="w-4 h-4" />
          </Button>
        </div>
      )}

      {/* Idle state — choose method */}
      {panelState === 'idle' && (
        <div className={cn('border border-dashed rounded-xl p-5 space-y-4',
          lightTheme.border.default, lightTheme.background.card, 'dark:bg-slate-900 dark:border-border')}>

          {/* Jetson camera enroll — primary method */}
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-1.5">
                {jetsonOnline === true && <Wifi className="w-3.5 h-3.5 text-emerald-500" />}
                {jetsonOnline === false && <WifiOff className="w-3.5 h-3.5 text-red-400" />}
                {jetsonOnline === null && <Loader2 className="w-3.5 h-3.5 animate-spin text-slate-400" />}
                <span className={cn('text-xs font-semibold',
                  jetsonOnline === true ? 'text-emerald-600 dark:text-emerald-400' :
                  jetsonOnline === false ? 'text-red-500 dark:text-red-400' : 'text-slate-500')}>
                  Jetson C++ Runner {jetsonOnline === true ? '· Online' : jetsonOnline === false ? '· Offline' : '· Checking...'}
                </span>
              </div>
              <Button variant="ghost" size="icon" className="h-6 w-6" onClick={checkJetson}>
                <RefreshCw className="w-3 h-3" />
              </Button>
            </div>
            <Button
              className="w-full"
              disabled={jetsonOnline !== true}
              onClick={handleCameraEnroll}
            >
              <ScanFace className="w-4 h-4 mr-2" />
              {isEnrolled ? 'Re-Enroll from Camera' : 'Enroll from Camera'}
            </Button>
            {jetsonOnline === false && (
              <p className="text-[11px] text-red-500 dark:text-red-400">
                Start frs-runner on Jetson: <code className="font-mono">sudo systemctl start frs-runner</code>
              </p>
            )}
          </div>

          <div className="relative flex items-center">
            <div className="flex-grow border-t border-dashed border-slate-200 dark:border-slate-700" />
            <span className="mx-3 text-xs text-slate-400">or upload photo</span>
            <div className="flex-grow border-t border-dashed border-slate-200 dark:border-slate-700" />
          </div>

          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => { e.preventDefault(); const f = e.dataTransfer.files?.[0]; if (f) handleFile(f); }}
            className="space-y-3"
          >
            <div className="flex flex-wrap gap-2">
              <Button variant="outline" size="sm" onClick={() => inputRef.current?.click()}>
                <Upload className="w-4 h-4 mr-2" />Upload Photo
              </Button>
              <Button variant="outline" size="sm" onClick={startWebcam}>
                <Camera className="w-4 h-4 mr-2" />Use Webcam
              </Button>
            </div>
            <p className={cn('text-[10px]', lightTheme.text.muted, 'dark:text-slate-500')}>
              JPG/PNG/WEBP · max 5 MB · one face · front-facing · good lighting
            </p>
          </div>
        </div>
      )}

      {/* Webcam capture */}
      {panelState === 'selecting' && (
        <div className={cn('border rounded-xl p-4 space-y-4',
          lightTheme.background.card, lightTheme.border.default, 'dark:bg-slate-900 dark:border-border')}>
          <div className="relative overflow-hidden rounded-lg">
            <video ref={videoRef} className="w-full rounded-lg" />
            <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
              <div className="w-48 h-64 border-2 border-emerald-400/70 rounded-full bg-emerald-500/5" />
            </div>
          </div>
          <div className="flex gap-2">
            <Button className="flex-1" onClick={captureWebcam}><Camera className="w-4 h-4 mr-2" />Capture</Button>
            <Button variant="outline" className="flex-1" onClick={() => { if (stream) stream.getTracks().forEach(t=>t.stop()); setStream(null); resetToIdle(); }}>Cancel</Button>
          </div>
        </div>
      )}

      {/* Photo preview */}
      {panelState === 'preview' && previewUrl && (
        <div className={cn('border rounded-xl p-4 space-y-4',
          lightTheme.background.card, lightTheme.border.default, 'dark:bg-slate-900 dark:border-border')}>
          <div className="relative">
            <img src={previewUrl} alt="Preview" className="w-full rounded-lg object-cover max-h-64" />
            <Button variant="ghost" size="icon" className="absolute top-2 right-2 bg-white/80 dark:bg-slate-800/80" onClick={resetToIdle}>
              <X className="w-4 h-4" />
            </Button>
          </div>
          <Button onClick={handlePhotoEnroll} className="w-full">
            {isEnrolled ? 'Re-Enroll with This Photo' : 'Enroll with This Photo'}
          </Button>
        </div>
      )}

      {/* Uploading spinner */}
      {panelState === 'uploading' && (
        <div className={cn('border rounded-xl p-4 space-y-4',
          lightTheme.background.card, lightTheme.border.default, 'dark:bg-slate-900 dark:border-border')}>
          <div className="flex items-center gap-3">
            <Loader2 className="w-4 h-4 animate-spin text-blue-500" />
            <span className={cn('text-sm', lightTheme.text.secondary, 'dark:text-slate-300')}>
              Running ArcFace inference via sidecar…
            </span>
          </div>
          <Progress value={65} className="h-1.5" />
        </div>
      )}

      {/* Camera enrolling */}
      {panelState === 'enrolling-cam' && (
        <div className={cn('border rounded-xl p-4 space-y-4',
          lightTheme.background.card, lightTheme.border.default, 'dark:bg-slate-900 dark:border-border')}>
          <div className="flex items-center gap-3">
            <Loader2 className="w-4 h-4 animate-spin text-emerald-500" />
            <div>
              <p className={cn('text-sm font-medium', lightTheme.text.primary, 'dark:text-white')}>
                Enrolling from camera…
              </p>
              <p className={cn('text-xs', lightTheme.text.muted, 'dark:text-slate-500')}>
                Jetson capturing frame → YOLOv8 detection → ArcFace embedding (up to 30s)
              </p>
            </div>
          </div>
          <Progress value={40} className="h-1.5 animate-pulse" />
        </div>
      )}

      {/* Success */}
      {panelState === 'success' && (
        <div className="space-y-3">
          <div className="p-4 rounded-xl border border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900/40 dark:bg-emerald-900/20 dark:text-emerald-300 flex items-start gap-2">
            <CheckCircle2 className="w-4 h-4 mt-0.5 shrink-0" />
            <span className="text-sm">
              <strong>{employeeName}</strong> enrolled — will be recognised by cameras within 30 seconds.
            </span>
          </div>
          <Button variant="outline" size="sm" onClick={resetToIdle}>Add Another Photo</Button>
        </div>
      )}

      {/* Error */}
      {panelState === 'error' && (
        <div className="space-y-3">
          <div className="p-4 rounded-xl border border-red-200 bg-red-50 text-red-700 dark:border-red-900/40 dark:bg-red-900/20 dark:text-red-300 flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
            <span className="text-sm">{errorMessage || 'Enrollment failed.'}</span>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={resetToIdle}>Try Again</Button>
            {panelState === 'error' && (
              <Button variant="ghost" size="sm" onClick={checkJetson}>
                <RefreshCw className="w-3 h-3 mr-1" />Check Jetson
              </Button>
            )}
          </div>
        </div>
      )}

      <input ref={inputRef} type="file" accept={ACCEPTED_TYPES.join(',')} className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); e.currentTarget.value = ''; }} />
      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
};
TSXEOF

echo "  ✅ FaceEnrollButton.tsx updated"

# ── Rebuild frontend ──────────────────────────────────────────────────────────
echo ""
echo "Rebuilding frontend..."
docker compose build frontend 2>&1 | tail -3
docker compose up -d frontend

echo ""
echo "=================================================="
echo " ✅ Enrollment UI updated"
echo "=================================================="
echo ""
echo "What changed:"
echo "  • 'Enroll from Camera' button — triggers Jetson C++ runner directly"
echo "  • Live Jetson Online/Offline indicator with refresh button"
echo "  • Photo upload fallback still works (uses Jetson sidecar via backend)"
echo "  • Webcam capture still works"
echo "  • Delete enrollment with confirmation dialog"
echo "  • Better error messages with systemctl hint"
echo ""
echo "Hard refresh: Ctrl+Shift+R → HR Dashboard → Employee Management → click employee"
