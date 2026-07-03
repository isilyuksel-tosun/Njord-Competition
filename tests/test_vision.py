"""
test_vision_live.py

BuoyDetector'ı gerçek ZED2i kamera akışı üzerinde canlı olarak test eder.
Referans script'teki (ZED+YOLO) FPS / CPU / RAM / GPU raporlama ve
derinlik ölçüm mantığı, BuoyDetector.detect() API'sine uyarlanmıştır.

NOT: Bu dosya unittest.TestCase DEĞİLDİR – sonsuz kamera döngüsü içerdiği
için pytest/unittest runner'ı ile birlikte çalışmaz. Ayrı bir manuel
doğrulama script'i olarak çalıştırın:

    python3 test_vision_live.py --device cuda
"""
import argparse
import os
import re
import signal
import subprocess
import sys
import threading
import time

import cv2
import psutil
import pyzed.sl as sl

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)  # append yerine insert(0, ...) daha garanti

from config.camera_config import *
from config.vision_config import BUOY_MODEL_PATH
from vision.detector import BuoyDetector

# ---------------- Ayarlar ----------------
LOG_INTERVAL_SEC = 2.0  # Terminal raporlama aralığı (saniye)
MAX_DEPTH_M = 40.0

# ---- Global Döngü Kontrolü ----
is_running = True

# ---- GPU İzleme (arka plan thread'i, ana döngüyü BLOKLAMAZ) ----
_gpu_usage_lock = threading.Lock()
_gpu_usage_value = "%0 (henüz veri yok)"
_tegrastats_proc = None


def signal_handler(sig, frame):
    """Ctrl+C (SIGINT) gönderildiğinde programı güvenle kapatır."""
    global is_running
    print("\n[BİLGİ] Kapatma sinyali alındı. Çıkılıyor...")
    is_running = False


signal.signal(signal.SIGINT, signal_handler)


def _gpu_monitor_thread_nvidia_smi():
    """Masaüstü/nvidia-smi olan sistemler için: periyodik olarak GPU kullanımını okur."""
    global _gpu_usage_value
    while is_running:
        try:
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                encoding='utf-8', stderr=subprocess.DEVNULL, timeout=1.0
            )
            with _gpu_usage_lock:
                _gpu_usage_value = f"%{result.strip()}"
        except Exception:
            pass
        time.sleep(1.0)


def _gpu_monitor_thread_tegrastats():
    """Jetson için: tegrastats'ı BİR KERE, sürekli akan bir process olarak başlatır ve
    çıktısını arka planda okuyup son değeri _gpu_usage_value'ya yazar. Ana döngü asla
    process spawn etmez / readline ile bloklanmaz."""
    global _gpu_usage_value, _tegrastats_proc
    try:
        _tegrastats_proc = subprocess.Popen(
            ["tegrastats", "--interval", "1000"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
    except Exception:
        return

    for line in _tegrastats_proc.stdout:
        if not is_running:
            break
        match = re.search(r'GR3D(?:_FREQ)?\s*(\d+)%', line)
        if match:
            with _gpu_usage_lock:
                _gpu_usage_value = f"%{match.group(1)} (Tegrastats)"


def _gpu_monitor_thread_sysfs():
    """nvidia-smi ve tegrastats yoksa: sysfs'ten periyodik okuma (hızlı, bloklamaz)."""
    global _gpu_usage_value
    sysfs_paths = [
        "/sys/class/devfreq/17000000.gv11b/device/load",
        "/sys/class/devfreq/17000000.ga10b/device/load",
        "/sys/devices/gpu.0/load"
    ]
    while is_running:
        for path in sysfs_paths:
            try:
                with open(path, 'r') as f:
                    val = float(f.read().strip())
                    text = f"%{val / 10.0:.1f} (sysfs)" if val > 100 else f"%{val:.1f} (sysfs)"
                    with _gpu_usage_lock:
                        _gpu_usage_value = text
                    break
            except Exception:
                continue
        time.sleep(1.0)


def start_gpu_monitor():
    """Uygun GPU izleme yöntemini bir kere seçip arka planda başlatır (daemon thread)."""
    try:
        subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                encoding='utf-8', stderr=subprocess.DEVNULL, timeout=1.0)
        t = threading.Thread(target=_gpu_monitor_thread_nvidia_smi, daemon=True)
        t.start()
        return
    except Exception:
        pass

    try:
        subprocess.check_output(['which', 'tegrastats'], stderr=subprocess.DEVNULL, timeout=1.0)
        t = threading.Thread(target=_gpu_monitor_thread_tegrastats, daemon=True)
        t.start()
        return
    except Exception:
        pass

    t = threading.Thread(target=_gpu_monitor_thread_sysfs, daemon=True)
    t.start()


def get_gpu_usage() -> str:
    """Arka plan thread'inin en son yazdığı değeri anında döndürür (bloklamaz)."""
    with _gpu_usage_lock:
        return _gpu_usage_value


def parse_args():
    parser = argparse.ArgumentParser(description="BuoyDetector canlı ZED2i testi")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Modelin çalışacağı cihaz (varsayılan: cpu)")
    parser.add_argument("--headless", action="store_true",
                        help="cv2 penceresi açmadan, sadece terminal loglarıyla çalıştır")
    return parser.parse_args()


# noinspection D
def main():
    global is_running
    args = parse_args()

    start_gpu_monitor()

    print(f"[BİLGİ] ZED Kamera başlatılıyor... (Hedef Çözünürlük referansı: {CAMERA_WIDTH}x{CAMERA_HEIGHT})")
    zed = sl.Camera()
    init_params = sl.InitParameters()

    init_params.camera_resolution = CAMERA_RESOLUTION
    init_params.camera_fps = CAMERA_FPS
    init_params.depth_mode = DEPTH_MODE
    init_params.coordinate_units = COORDINATE_UNITS
    init_params.depth_minimum_distance = 0.3

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED açılamadı: {status}")

    # Sabit pozlama/gain ile hız kilidini kırıyoruz (ref. script ile aynı mantık)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 0)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 20)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 50)

    runtime_params = sl.RuntimeParameters()
    image_zed = sl.Mat()
    depth_zed = sl.Mat()

    print(f"[BİLGİ] BuoyDetector yükleniyor: {BUOY_MODEL_PATH} (device={args.device})")
    detector = BuoyDetector(model_path=BUOY_MODEL_PATH, device=args.device)
    if detector.model is None:
        raise RuntimeError("BuoyDetector modeli yüklenemedi.")

    print(f"[BİLGİ] Sınıflar: {detector.class_names}")

    last_log_time = time.time()
    frames_since_last_log = 0

    print(f"\n[BİLGİ] İşlem başladı. Her {LOG_INTERVAL_SEC} saniyede bir durum raporlanacak.")
    print("[BİLGİ] Durdurmak için terminalde Ctrl+C yapın.\n")

    # PENCEREYİ BOYUTLANDIRILABİLİR OLARAK OLUŞTURUYORUZ
    if not args.headless:
        cv2.namedWindow("BuoyDetector - Canli Test", cv2.WINDOW_NORMAL)

    try:
        while is_running:
            t_loop_start = time.time()

            if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                continue
            t_grab = time.time()

            current_time = time.time()
            frames_since_last_log += 1

            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
            t_retrieve = time.time()

            bgra_data = image_zed.get_data()
            frame = cv2.cvtColor(bgra_data, cv2.COLOR_BGRA2BGR)
            depth_map = depth_zed.get_data()
            t_convert = time.time()

            detections = detector.detect(frame, depth_map)
            t_detect = time.time()

            detected_objects = []

            for det in detections:
                label = det.get("class", "?")
                conf = det.get("confidence", 0.0)
                distance = det.get("distance", None)
                bbox = det.get("bbox", None)

                distance_text = f"{distance:.2f}m" if distance is not None else "N/A"
                detected_objects.append(f"{label} ({distance_text})")

                if bbox is not None:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Sınıf adı + güven skoru: kutunun sol üstü
                    label_text = f"{label} {conf:.2f}"
                    cv2.putText(frame, label_text, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Derinlik bilgisi: kutunun sağ üstü (metni sağa hizalamak için genişliğini ölç)
                    (text_w, _), _ = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    depth_x = max(0, x2 - text_w)
                    cv2.putText(frame, distance_text, (depth_x, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

            t_draw = time.time()

            elapsed_time = current_time - last_log_time
            if elapsed_time >= LOG_INTERVAL_SEC:
                fps = frames_since_last_log / elapsed_time
                cpu_usage = psutil.cpu_percent()
                ram_usage = psutil.virtual_memory().percent
                gpu_info = get_gpu_usage()

                print(f"--- [ {time.strftime('%H:%M:%S')} Sistem Durumu ] ---")
                print(f"FPS  : {fps:.1f}")
                print(f"CPU  : %{cpu_usage:.1f}")
                print(f"RAM  : %{ram_usage:.1f}")
                print(f"GPU  : {gpu_info}")
                print(f"Nesne: {', '.join(detected_objects) if detected_objects else 'Yok'}")
                print("-" * 35)

                last_log_time = current_time
                frames_since_last_log = 0

            if not args.headless:
                cv2.imshow("BuoyDetector - Canli Test", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    is_running = False
            t_show = time.time()

            # --- ZAMANLAMA TEŞHİSİ: Yavaş frame'leri yakala ---
            step_grab = t_grab - t_loop_start
            step_retrieve = t_retrieve - t_grab
            step_convert = t_convert - t_retrieve
            step_detect = t_detect - t_convert
            step_draw = t_draw - t_detect
            step_show = t_show - t_draw
            step_total = t_show - t_loop_start

            if step_total > 0.15:  # 150ms üzeri = gözle görülür takılma
                print(f"[SLOW FRAME] TOTAL={step_total * 1000:.0f}ms | "
                      f"grab={step_grab * 1000:.0f}ms retrieve={step_retrieve * 1000:.0f}ms "
                      f"convert={step_convert * 1000:.0f}ms detect={step_detect * 1000:.0f}ms "
                      f"draw={step_draw * 1000:.0f}ms show={step_show * 1000:.0f}ms")

    finally:
        print("[BİLGİ] Kaynaklar serbest bırakılıyor...")
        is_running = False
        if _tegrastats_proc is not None:
            try:
                _tegrastats_proc.terminate()
            except Exception:
                pass
        zed.close()
        if not args.headless:
            cv2.destroyAllWindows()
        print("[BİLGİ] Başarıyla kapatıldı.")


if __name__ == "__main__":
    main()
