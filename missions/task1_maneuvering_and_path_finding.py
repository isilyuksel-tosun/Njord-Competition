import time
from enum import Enum, auto
from pathlib import Path

import rclpy
from rclpy.node import Node

# Yardımcı fonksiyonlar (Kendi yazdıklarımız ve mavlink_utilities içindekiler)
from utils.mavlink_utilities import (
    create_mission_topics,
    create_mission_clients,
    wait_for_mission_services,
    call_set_mode,
    call_trigger_service,
    publish_cmd_vel,
    stop_vehicle,
    calculate_gps_distance,
    calculate_bearing
)
from utils.read_json import main as read_json

BASE_DIR = Path(__file__).resolve().parent.parent
WAYPOINT_PATH = BASE_DIR / "tests" / "waypoints" / "waypoints.json"

# ============================================================
# SAFETY PARAMS
# ============================================================
GPS_TIMEOUT_SEC = 2.0  # Bu süre GPS/heading gelmezse dur
GEOFENCE_RADIUS_M = 150.0  # Başlangıç noktasından max uzaklık
AVOID_ENTER_DIST_M = 3.0  # Kaçınma tetiklenme mesafesi
AVOID_EXIT_DIST_M = 4.0  # Kaçınma bitiş mesafesi


class MissionState(Enum):
    INIT = auto()  # Başlangıç konumu bekleniyor / WP0 doğrulanıyor
    NAVIGATING = auto()  # Normal waypoint takibi
    AVOIDING = auto()  # Şamandıra kaçınma
    FINISHED = auto()  # Görev tamamlandı
    FAILSAFE = auto()  # GPS kaybı / geofence ihlali / beklenmeyen hata


# ============================================================
# MISSION LOGIC
# ============================================================
class Task1Maneuvering:
    def __init__(self, node, mission_topics, mission_clients):
        self.node = node
        self.is_armed = False
        self.logger = node.get_logger()

        self.topics = mission_topics
        self.clients = mission_clients

        # Waypointleri JSON'dan oku
        self.waypoints = read_json(WAYPOINT_PATH)
        self.current_target_index = 0
        self.waypoint_tolerance = 1

        # Anlık konum verileri
        self.current_lat = None
        self.current_lon = None
        self.current_heading = 0.0
        self.last_angular_z = 0.0
        self.finished = False

        # --- Güvenlik / state machine alanları ---
        self.state = MissionState.INIT
        self.last_gps_time = None
        self.last_heading_time = None
        self.home_lat = None
        self.home_lon = None
        self.avoiding_class = None  # şu an hangi obje yüzünden kaçınıyoruz

    def update_gps(self, lat, lon, heading):
        """ROS 2 Node'undan gelen güncel GPS ve yönelim verilerini kaydeder."""
        self.current_lat = lat
        self.current_lon = lon
        self.current_heading = heading
        self.last_gps_time = time.monotonic()
        self.last_heading_time = time.monotonic()

        if self.home_lat is None:
            # İlk GPS okuması home/geofence merkezi olarak kaydedilir
            self.home_lat = lat
            self.home_lon = lon
            self.logger.info(f"Home position set: {lat:.6f}, {lon:.6f}")

    def _check_watchdog(self):
        """GPS/heading verisi zamanında gelmiyorsa FAILSAFE'e geç. True dönerse devam edilebilir."""
        now = time.monotonic()

        if self.last_gps_time is None:
            # Henüz hiç veri gelmedi, bu normal başlangıç durumu (FAILSAFE değil)
            return False

        if (now - self.last_gps_time) > GPS_TIMEOUT_SEC:
            if self.state != MissionState.FAILSAFE:
                self.logger.error(
                    f"GPS DATA NOT RECEIVED FOR OVER {GPS_TIMEOUT_SEC}s! FAILSAFE."
                )
            self.state = MissionState.FAILSAFE
            return False

        return True

    def _check_geofence(self):
        """Home noktasından çok uzaklaşıldıysa FAILSAFE'e geç. True dönerse sınır içinde."""
        if self.home_lat is None or self.current_lat is None:
            return True

        dist_from_home = calculate_gps_distance(
            self.home_lat, self.home_lon,
            self.current_lat, self.current_lon
        )

        if dist_from_home > GEOFENCE_RADIUS_M:
            if self.state != MissionState.FAILSAFE:
                self.logger.error(
                    f"GEOFENCE VIOLATION! {dist_from_home:.1f}m away from home "
                    f"(limit {GEOFENCE_RADIUS_M}m). FAILSAFE."
                )
            self.state = MissionState.FAILSAFE
            return False

        return True

    def _nearest_relevant_obstacle(self, detections):
        """Kaçınma menzilindeki en yakın red/green şamandırayı döndürür (None yoksa)."""
        candidates = [
            obj for obj in detections
            if obj.get("class") in ("red_buoy", "green_buoy")
               and obj.get("distance") is not None
               and 0 < obj["distance"] < AVOID_EXIT_DIST_M
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda o: o["distance"])

    # noinspection D
    def update(self, detections):
        """Sürekli çalışan ana kontrol döngüsü."""

        # ---------------------------------------------------------
        # 0. GÜVENLİK KONTROLLERİ (her şeyden önce)
        # ---------------------------------------------------------
        gps_ok = self._check_watchdog()

        if self.state == MissionState.FAILSAFE:
            stop_vehicle(self.topics.cmd_vel_pub)
            self.logger.warn("FAILSAFE active, vehicle stopped.", throttle_duration_sec=2.0)
            return

        if not gps_ok:
            # Henüz hiç GPS gelmedi (başlangıç), bekle
            self.logger.info("Waiting for GPS Data...", throttle_duration_sec=2.0)
            publish_cmd_vel(self.topics.cmd_vel_pub, linear_x=0.0, angular_z=0.0)
            return

        if not self._check_geofence():
            stop_vehicle(self.topics.cmd_vel_pub)
            return

        if not self.waypoints:
            self.logger.warn("Mission list is empty! Please check the route.", throttle_duration_sec=5.0)
            stop_vehicle(self.topics.cmd_vel_pub)
            return

        if self.current_target_index >= len(self.waypoints):
            if not self.finished:
                self.logger.info("ALL WAYPOINTS REACHED! MISSION COMPLETED!")
                stop_vehicle(self.topics.cmd_vel_pub)
                self.finished = True
                self.state = MissionState.FINISHED
            return

        target_gps = self.waypoints[self.current_target_index]
        target_lat = target_gps["lat"]
        target_lon = target_gps["lon"]

        distance = calculate_gps_distance(
            self.current_lat, self.current_lon,
            target_lat, target_lon
        )

        # ---------------------------------------------------------
        # 1. ENGELLERDEN KAÇINMA KONTROLÜ (hysteresis'li state)
        # ---------------------------------------------------------
        nearest = self._nearest_relevant_obstacle(detections)

        if self.state == MissionState.AVOIDING:
            # Zaten kaçınıyorsak, çıkış mesafesine kadar kaçınmaya devam et
            still_close = nearest is not None and nearest["distance"] < AVOID_EXIT_DIST_M
            if not still_close:
                self.logger.info("Avoidance completed, returning to normal navigation.")
                self.state = MissionState.NAVIGATING
            else:
                self._do_avoid(nearest)
                return
        else:
            # Normal seyirdeyken sadece giriş mesafesinden yakın olan tetikler
            if nearest is not None and nearest["distance"] < AVOID_ENTER_DIST_M:
                self.state = MissionState.AVOIDING
                self.avoiding_class = nearest["class"]
                self._do_avoid(nearest)
                return

        # ---------------------------------------------------------
        # 2. WP0 / MISSION BAŞLANGIÇ KONTROLÜ
        # ---------------------------------------------------------
        if self.state == MissionState.INIT:
            if self.current_target_index == 0 and distance < (self.waypoint_tolerance + 2.0):
                self.logger.info("WP0 (Start) point verified, mission starting.")
                self.current_target_index += 1
                self.state = MissionState.NAVIGATING
                return
            else:
                # Henüz start noktasında değiliz; WP0'a doğru ilerlemeye devam et,
                # ama mission'ı NAVIGATING'e geçirmeden (WP0'ı atlamadan).
                pass

        self.state = MissionState.NAVIGATING if self.state == MissionState.INIT else self.state

        # ---------------------------------------------------------
        # 3. MESAFE VE HEDEF KONTROLÜ
        # ---------------------------------------------------------
        if distance < self.waypoint_tolerance:
            self.logger.info(f"Reached Waypoint {self.current_target_index}! Remaining: {distance:.2f}m")
            self.current_target_index += 1
            return

        # ---------------------------------------------------------
        # 4. YÖN (HEADING) KONTROLÜ (P-Kontrolcü)
        # ---------------------------------------------------------
        target_bearing = calculate_bearing(
            self.current_lat, self.current_lon,
            target_lat, target_lon
        )

        self.logger.info(
            f"Target WP{self.current_target_index} | Distance: {distance:.2f}m | "
            f"Heading Error: {(target_bearing - self.current_heading):.1f}°",
            throttle_duration_sec=1.0
        )

        heading_error = target_bearing - self.current_heading

        if heading_error > 180:
            heading_error -= 360
        elif heading_error < -180:
            heading_error += 360

        abs_error = abs(heading_error)

        kp_angular = 0.015
        max_turn_speed = 0.4
        deadband_deg = 5.0

        if abs_error < deadband_deg:
            raw_angular_z = 0.0
        else:
            # KRİTİK DÜZELTME: Dönüş komutunu tersine çeviriyoruz (-1 ile çarpıyoruz)
            # ROS'ta +Z sola dönmektir. Eğer aracınız zıt yöne savruluyorsa sorun buradadır.
            # NOT: Bu işaret gerçek suda tekrar doğrulanmalı (farklı başlangıç
            # yönelimleriyle simülasyon/havuz testi yapılmadan güvenilmemeli).
            raw_angular_z = -1.0 * (heading_error * kp_angular)

        raw_angular_z = max(-max_turn_speed, min(max_turn_speed, raw_angular_z))

        alpha = 0.25
        cmd_angular_z = self.last_angular_z + alpha * (raw_angular_z - self.last_angular_z)
        self.last_angular_z = cmd_angular_z

        proximity_threshold = self.waypoint_tolerance + 1.5

        if distance < proximity_threshold:
            cmd_linear_x = 0.20
        else:
            if abs_error > 70:
                cmd_linear_x = 0.12
            elif abs_error > 40:
                cmd_linear_x = 0.17
            elif abs_error > 20:
                cmd_linear_x = 0.20
            else:
                cmd_linear_x = 0.23

        publish_cmd_vel(
            self.topics.cmd_vel_pub,
            linear_x=cmd_linear_x,
            angular_z=cmd_angular_z
        )

    def _do_avoid(self, obstacle):
        """Tek bir kaçınma komutu üretir (state AVOIDING iken her tick'te çağrılır)."""
        if obstacle["class"] == "red_buoy":
            self.logger.info(
                f"Red buoy ({obstacle['distance']:.1f}m)! Avoiding to starboard.",
                throttle_duration_sec=1.0
            )
            publish_cmd_vel(self.topics.cmd_vel_pub, linear_x=0.5, angular_z=-0.6)
        else:  # green_buoy
            self.logger.info(
                f"Green buoy ({obstacle['distance']:.1f}m)! Avoiding to port.",
                throttle_duration_sec=1.0
            )
            publish_cmd_vel(self.topics.cmd_vel_pub, linear_x=0.5, angular_z=0.6)


# ============================================================
# ROS 2 NODE (GÖREV YÖNETİCİSİ)
# ============================================================
class Task1Node(Node):
    def __init__(self):
        super().__init__('task1_mission_node')
        self.get_logger().info("Task 1 (Maneuvering) Node Starting...")

        # 1. Servis İstemcilerini (Clients) Oluştur ve Bekle
        self.mission_clients = create_mission_clients(self)
        wait_for_mission_services(self, self.mission_clients)

        # 2. Topic Aboneliklerini (Subscribers/Publishers) Oluştur
        self.mission_topics = create_mission_topics(
            self,
            gps_callback=self.gps_callback,
            heading_callback=self.heading_callback,
            state_callback=self.state_callback
        )

        # 3. Görev Sınıfını Başlat
        self.task = Task1Maneuvering(self, self.mission_topics, self.mission_clients)

        # Anlık Yönelim Değişkeni (GPS Callback'e aktarmak için)
        self.current_heading = 0.0

        # 4. Ana Kontrol Döngüsünü Başlat (Saniyede 10 kez çalışır: 0.1 sn)
        self.control_timer = self.create_timer(0.1, self.timer_callback)

    def gps_callback(self, msg):
        """Araçtan gelen NavSatFix verisini dinler."""
        self.task.update_gps(msg.latitude, msg.longitude, self.current_heading)

    def heading_callback(self, msg):
        """Araçtan gelen Float32 yön verisini dinler."""
        self.current_heading = msg.data
        self.task.last_heading_time = time.monotonic()

    def state_callback(self, msg):
        """Bridge'den gelen durum mesajlarını dinler (Gerekirse kullanılır)."""
        pass

    def timer_callback(self):
        """Görev mantığını sürekli tetikler.

        KRİTİK: Bu fonksiyon içinde beklenmeyen bir hata (örn. bozuk detection
        formatı) fırlarsa, düzeltilmezse araç son verilen cmd_vel komutuyla
        donmuş halde sürüklenmeye devam eder. Bu yüzden her tick try/except
        ile korunuyor ve hata durumunda araç durduruluyor.
        """
        # TODO: ZED kamerasından gelen tespitler (detections) buraya aktarılacak.
        # Şimdilik boş liste -> kaçınma path'i gerçek suda hiç test edilmemiş demektir,
        # entegre edilmeden kaçınma mantığına güvenilmemeli.
        current_detections = []

        try:
            self.task.update(detections=current_detections)
        except Exception as exc:  # noqa: BLE001 - kasıtlı geniş yakalama, failsafe için
            self.get_logger().error(f"Unexpected error in timer_callback: {exc}")
            try:
                stop_vehicle(self.mission_topics.cmd_vel_pub)
            except Exception as stop_exc:  # noqa: BLE001
                self.get_logger().error(f"Failed to stop vehicle: {stop_exc}")
            self.task.state = MissionState.FAILSAFE


# ============================================================
# ANA ÇALIŞTIRMA BLOĞU
# ============================================================
def main(args=None):
    rclpy.init(args=args)

    node = Task1Node()

    try:
        node.get_logger().info("Setting vehicle to MANUAL mode...")
        mode_ok = call_set_mode(node, node.mission_clients.set_mode_client, "MANUAL")
        if mode_ok is False:
            node.get_logger().error("Failed to switch to MANUAL mode! Mission not starting.")
            return

        node.get_logger().info("Arming vehicle...")
        arm_ok = call_trigger_service(node, node.mission_clients.arm_client, "ARM")
        if arm_ok is False:
            node.get_logger().error("ARM failed! Mission not starting.")
            return

        node.get_logger().info("Mission loop started.")

        while rclpy.ok() and not node.task.finished and node.task.state != MissionState.FAILSAFE:
            rclpy.spin_once(node, timeout_sec=0.1)

        if node.task.state == MissionState.FAILSAFE:
            node.get_logger().error("Mission terminated due to FAILSAFE.")
        else:
            node.get_logger().info("Mission finished. Stopping vehicle.")

        stop_vehicle(node.mission_topics.cmd_vel_pub)

        node.get_logger().info("Disarming vehicle...")
        call_trigger_service(node, node.mission_clients.disarm_client, "DISARM")

    except KeyboardInterrupt:
        node.get_logger().info("Mission terminated manually.")
        stop_vehicle(node.mission_topics.cmd_vel_pub)
        try:
            call_trigger_service(node, node.mission_clients.disarm_client, "DISARM")
        except Exception:  # noqa: BLE001
            pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

"""
    DEĞİŞİKLİK NOTLARI (bu revizyonda eklenenler):
    - GPS/heading watchdog: GPS_TIMEOUT_SEC süresi aşılırsa FAILSAFE + dur.
    - Geofence: home noktasından GEOFENCE_RADIUS_M'den uzaklaşılırsa FAILSAFE + dur.
    - timer_callback try/except: beklenmeyen hatada araç donup kalmak yerine durur.
    - WP0 atlama artık ayrı bir MissionState.INIT state'i ile korunuyor.
    - Kaçınma artık hysteresis'li bir state (AVOIDING): giriş 3m / çıkış 4m,
      zigzag riskini azaltmak için.
    - En yakın engel seçimi: detections listesi sırasına değil, gerçek mesafeye göre.
    - call_set_mode / call_trigger_service dönüş değerleri artık kontrol ediliyor
      (mavlink_utilities.py'nin bool/None dönmediğini varsayarsan bu kontrolleri
      kendi API'ne göre uyarlaman gerekebilir).

    HALA YAPILMASI GEREKENLER:
    - ZED+YOLO detections gerçek pipeline'a bağlanmalı (şu an hep boş liste).
    - COLREG/CPA-TCPA modülü bu task'a entegre edilmeli.
    - Heading işareti (-1.0 çarpanı) simülasyon/havuzda farklı başlangıç
      yönelimleriyle doğrulanmalı.
    - GEOFENCE_RADIUS_M ve GPS_TIMEOUT_SEC saha koşullarına göre kalibre edilmeli.
"""
