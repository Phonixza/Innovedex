"""
Dual Gripper Block Picker
Algorithm:
  1. สแกน Spot 1-5 ว่ามีบล็อกสีอะไร
  2. R ตรวจจาก Spot 1→2→3, L ตรวจจาก Spot 5→4→3
  3. ถ้าเจอทั้งคู่ → คีบพร้อมกัน → ส่ง R → ส่ง L
  4. ถ้าเจอแค่อันเดียว → คีบอันเดียว → ส่ง
  5. วนจนหมด
"""

import time, cv2, json, threading
import numpy as np
from ultralytics import YOLO
from telemetrix import telemetrix
from pathlib import Path

# ─── CONFIG ────────────────────────────────────────────────────────────────
MODEL_PATH  = r"D:\yolov_11_custom\yolov11_custom.pt"
CAMERA_ID   = 0
CONF        = 0.6
IOU         = 0.1
PORT        = "COM7"
CONFIG_FILE = "config.json"

FRAME_W, FRAME_H = 800, 500

SPOTS = {
    1: [55,  210],
    2: [185, 155],
    3: [310, 130],
    4: [425, 150],
    5: [550, 205],
}
DETECT_RANGE = 55

PICK_POS_L = {   # GripperL คีบจาก Spot 5, 4, 3 White
    5: {"base": 100, "arm": 62},
    4: {"base": 100, "arm": 75},
    3: {"base": 85,  "arm": 89},
}


PICK_POS_R = {   # GripperR คีบจาก Spot 1, 2, 3 Blue
    1: {"base": 83,  "arm": 115},
    2: {"base": 85,  "arm": 98},
    3: {"base": 85,  "arm": 88},
}


# White Base gripper
DROP_POS_L = {
    "Green Block":  {"base": 85,  "arm": 27,  "up": 0},
    "Yellow Block": {"base": 110, "arm": 0,   "up": 0},
    "Blue Block":   {"base": 90,  "arm": 160, "up": 0},
    "Red Block":    {"base": 100,  "arm": 154, "up": 0},
}

# Blue Base gripper
DROP_POS_R = {
    "Green Block":  {"base": 82,  "arm": 35,  "up": 0},
    "Yellow Block": {"base": 104, "arm": 20,  "up": 0},
    "Blue Block":   {"base": 80,  "arm": 170, "up": 0},
    "Red Block":    {"base": 103, "arm": 153, "up": 0},
}

CLASS_NAMES = {
    0: "Blue Block",
    1: "Green Block",
    2: "Red Block",
    3: "Yellow Block",
}
COLOR_BGR = {
    "Blue Block":   (255, 100,   0),
    "Green Block":  (0,   200,   0),
    "Red Block":    (0,     0, 220),
    "Yellow Block": (0,   210, 210),
}

# ── Pin ────────────────────────────────────────────────────────────────────
SERVO_GRIPPERL = 7    # ซ้าย  คีบจาก Spot 1→2→3
SERVO_GRIPPERR = 12   # ขวา  คีบจาก Spot 5→4→3
SERVO_UP       = 8    # 20=ขึ้น  90=ลง
SERVO_ARM      = 11
SERVO_BASE     = 4

GRIP_OPEN       = 120
GRIP_CLOSE      = 180
PICK_UP_ANGLE   = 0
PICK_DOWN_ANGLE = 65

SCAN_POS = {"base": 105, "arm": 15}

# ─── CONFIG LOAD/SAVE ──────────────────────────────────────────────────────
def load_config():
    global SPOTS, DETECT_RANGE
    if Path(CONFIG_FILE).exists():
        with open(CONFIG_FILE) as f:
            d = json.load(f)
        SPOTS        = {int(k): v for k, v in d.get("spots", SPOTS).items()}
        DETECT_RANGE = d.get("detect_range", DETECT_RANGE)

def save_config():
    with open(CONFIG_FILE, "w") as f:
        json.dump({"spots": SPOTS, "detect_range": DETECT_RANGE}, f, indent=2)

load_config()

# ─── ARDUINO ──────────────────────────────────────────────────────────────
def _safe_stepper(self, data):
    try:
        cb = self.stepper_info[data[2]].get("current_position_callback")
        if callable(cb): cb(data)
    except Exception: pass

telemetrix.Telemetrix._stepper_current_position_report = _safe_stepper

print("[INFO] Connecting Arduino...")
board = telemetrix.Telemetrix(com_port=PORT, arduino_wait=6)
time.sleep(2)

for pin, val in [
    (SERVO_GRIPPERL, GRIP_OPEN),
    (SERVO_GRIPPERR, GRIP_OPEN),
    (SERVO_UP,       PICK_UP_ANGLE),
    (SERVO_ARM,      85),
    (SERVO_BASE,     85),
]:
    board.set_pin_mode_servo(pin); time.sleep(0.1)
    board.servo_write(pin, val)
time.sleep(1)
print("[INFO] Arduino ready!")

# ─── SERVO ────────────────────────────────────────────────────────────────
servo_angles = {
    SERVO_GRIPPERL: GRIP_OPEN,
    SERVO_GRIPPERR: GRIP_OPEN,
    SERVO_UP:       PICK_UP_ANGLE,
    SERVO_ARM:      85,
    SERVO_BASE:     85,
}
_lock = threading.Lock()

def sw(pin, target, steps=5, delay=0.03):
    target = int(np.clip(target, 0, 180))
    start  = servo_angles.get(pin, target)
    if start == target: return
    for a in np.linspace(start, target, max(steps, 2), dtype=int):
        try:
            with _lock: board.servo_write(pin, int(a))
        except Exception as e:
            print(f"[WARN] sw {pin}: {e}"); break
        time.sleep(delay)
    servo_angles[pin] = target
    time.sleep(0.05)

def force_write(pin, target):
    """ส่ง servo command เสมอ ไม่สนค่าเดิม — ใช้ตอน UP ขึ้น/ลงหลัง drop"""
    target = int(np.clip(target, 0, 180))
    with _lock: board.servo_write(pin, target)
    servo_angles[pin] = target
    time.sleep(0.05)

def go_scan_pos():
    sw(SERVO_UP,   PICK_UP_ANGLE)
    sw(SERVO_BASE, SCAN_POS["base"])
    sw(SERVO_ARM,  SCAN_POS["arm"])
    time.sleep(0.3)

def home():
    sw(SERVO_UP,       PICK_UP_ANGLE)
    sw(SERVO_GRIPPERL, GRIP_OPEN)
    sw(SERVO_GRIPPERR, GRIP_OPEN)
    sw(SERVO_ARM,      85)
    sw(SERVO_BASE,     85)

# ─── ALGORITHM ────────────────────────────────────────────────────────────

def pick_R(spot_id: int):
    """GripperR คีบ Spot spot_id (คีบก่อน ไม่แตะ GripperL)"""
    p = PICK_POS_R[spot_id]
    print(f"[PICK-R] Spot{spot_id} B{p['base']} A{p['arm']}")
    sw(SERVO_UP,   PICK_UP_ANGLE)
    sw(SERVO_BASE, p["base"])
    sw(SERVO_ARM,  p["arm"])
    sw(SERVO_GRIPPERR, GRIP_OPEN)           # เปิดเฉพาะ R
    time.sleep(0.4)                          # รอให้หุ่นนิ่งก่อนลง
    sw(SERVO_UP, PICK_DOWN_ANGLE, steps=7, delay=0.025)
    time.sleep(0.25)
    sw(SERVO_GRIPPERR, GRIP_CLOSE)
    time.sleep(0.2)
    sw(SERVO_UP, PICK_UP_ANGLE, steps=7, delay=0.025)
    time.sleep(0.15)

def pick_L(spot_id: int):
    """GripperL คีบ Spot spot_id (R ยังถือบล็อกอยู่ ไม่เปิด R)"""
    p = PICK_POS_L[spot_id]
    print(f"[PICK-L] Spot{spot_id} B{p['base']} A{p['arm']}")
    sw(SERVO_UP,   PICK_UP_ANGLE)
    sw(SERVO_BASE, p["base"])
    sw(SERVO_ARM,  p["arm"])
    sw(SERVO_GRIPPERL, GRIP_OPEN)           # เปิดเฉพาะ L (R ยังปิด)
    time.sleep(0.4)                          # รอให้หุ่นนิ่งก่อนลง
    sw(SERVO_UP, PICK_DOWN_ANGLE, steps=7, delay=0.025)
    time.sleep(0.25)
    sw(SERVO_GRIPPERL, GRIP_CLOSE)
    time.sleep(0.2)
    sw(SERVO_UP, PICK_UP_ANGLE, steps=7, delay=0.025)
    time.sleep(0.15)

def drop_R(label: str):
    drop = DROP_POS_R.get(label)
    if not drop:
        print(f"[WARN] No DROP_POS_R for {label}"); return
    print(f"[DROP-R] {label} → B{drop['base']} A{drop['arm']} U{drop['up']}")
    force_write(SERVO_UP, PICK_UP_ANGLE)   # ← บังคับขึ้นสุดก่อนเสมอ
    time.sleep(1)                         # ← รอให้แขนขึ้นจริงๆ
    sw(SERVO_BASE, drop["base"])
    sw(SERVO_ARM,  drop["arm"])
    target_up = int(np.clip(drop["up"], 100, 180))
    with _lock: board.servo_write(SERVO_UP, target_up)
    servo_angles[SERVO_UP] = target_up
    time.sleep(0.25)
    sw(SERVO_GRIPPERR, GRIP_OPEN)
    time.sleep(0.2)
    force_write(SERVO_UP, PICK_UP_ANGLE)

def drop_L(label: str):
    drop = DROP_POS_L.get(label)
    if not drop:
        print(f"[WARN] No DROP_POS_L for {label}"); return
    print(f"[DROP-L] {label} → B{drop['base']} A{drop['arm']} U{drop['up']}")
    force_write(SERVO_UP, PICK_UP_ANGLE)   # ← บังคับขึ้นสุดก่อนเสมอ
    time.sleep(1)                         # ← รอให้แขนขึ้นจริงๆ
    sw(SERVO_BASE, drop["base"])
    sw(SERVO_ARM,  drop["arm"])
    target_up = int(np.clip(drop["up"], 100, 180))
    with _lock: board.servo_write(SERVO_UP, target_up)
    servo_angles[SERVO_UP] = target_up
    time.sleep(0.25)
    sw(SERVO_GRIPPERL, GRIP_OPEN)
    time.sleep(0.2)
    force_write(SERVO_UP, PICK_UP_ANGLE)

# ─── SEQUENCE ─────────────────────────────────────────────────────────────
is_busy    = False
running    = False
spot_done  = {i: False for i in SPOTS}
spot_color: dict = {}

def run_sequence():
    global is_busy, running, spot_done

    go_scan_pos()
    time.sleep(0.3)

    snap = dict(spot_color)
    print(f"[SEQ] Snapshot: {snap}")

    if not any(snap.values()):
        print("[SEQ] No blocks — aborting")
        running = False; is_busy = False; return

    # R เก็บตามลำดับ 1→2→3  |  L เก็บตามลำดับ 5→4→3
    # ถ้า spot ที่ควรไปว่าง → ข้ามไปหาอันถัดไปในลำดับของตัวเอง
    remaining = {sid: lbl for sid, lbl in snap.items() if lbl}

    R_ORDER = [1, 2, 3]   # R วิ่งซ้ายไปขวา
    L_ORDER = [5, 4, 3]   # L วิ่งขวาไปซ้าย

    def next_spot(order, taken):
        """หา spot ถัดไปที่ยังมีบล็อกและยังไม่ถูกคีบ"""
        for sid in order:
            if remaining.get(sid) and sid not in taken:
                return sid, remaining[sid]
        return None, None

    taken = set()

    while remaining:
        sid_r, label_r = next_spot(R_ORDER, taken)
        sid_l, label_l = next_spot(L_ORDER, taken | ({sid_r} if sid_r else set()))

        if not sid_r and not sid_l:
            break

        print(f"\n[SEQ] R:Spot{sid_r}({label_r})  L:Spot{sid_l}({label_l})")

        # Step 1: R คีบก่อน
        if sid_r:
            pick_R(sid_r)
            spot_done[sid_r] = True
            taken.add(sid_r)
            del remaining[sid_r]

        # Step 2: L คีบต่อ (R ยังถือบล็อกอยู่ ไม่เปิด R)
        if sid_l:
            pick_L(sid_l)
            spot_done[sid_l] = True
            taken.add(sid_l)
            del remaining[sid_l]

        # Step 3: วาง R
        if label_r:
            drop_R(label_r)

        # Step 4: วาง L
        if label_l:
            drop_L(label_l)

    print("\n[SEQ] ══ ALL DONE ══")
    home()
    running = False
    is_busy = False

# ─── CAMERA ────────────────────────────────────────────────────────────────
def open_camera(cam_id):
    """ลอง backend ต่าง ๆ จนกว่าจะเปิดได้"""
    for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
        cap = cv2.VideoCapture(cam_id, backend)
        if cap.isOpened():
            print(f"[INFO] Camera {cam_id} opened (backend={backend})")
            return cap
        cap.release()
    return None

print("[INFO] Loading YOLO...")
model = YOLO(MODEL_PATH)
print("[INFO] YOLO ready!")

# ─── หา CAMERA_ID อัตโนมัติถ้า index 0 ไม่ได้ ────────────────────────────
cap = open_camera(CAMERA_ID)
if cap is None:
    print(f"[WARN] Camera {CAMERA_ID} not found — scanning indices 0-4...")
    for try_id in range(5):
        cap = open_camera(try_id)
        if cap is not None:
            CAMERA_ID = try_id
            print(f"[INFO] Using camera index {CAMERA_ID}")
            break

if cap is None:
    print("[ERROR] No camera found! ตรวจสอบการเชื่อมต่อกล้องแล้วลองใหม่")
    board.shutdown()
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
for _ in range(5): cap.read(); time.sleep(0.05)

# ─── MOUSE / DRAW ──────────────────────────────────────────────────────────
dragging_spot = None
SPOT_R_VIZ = 14

def mouse_cb(event, x, y, flags, param):
    global dragging_spot
    if event == cv2.EVENT_LBUTTONDOWN:
        for sid, pos in SPOTS.items():
            if ((x-pos[0])**2+(y-pos[1])**2)**0.5 < SPOT_R_VIZ+8:
                dragging_spot = sid; break
    elif event == cv2.EVENT_MOUSEMOVE and dragging_spot:
        SPOTS[dragging_spot][:] = [x, y]
    elif event == cv2.EVENT_LBUTTONUP and dragging_spot:
        save_config(); dragging_spot = None

cv2.namedWindow("Innovedex")
cv2.setMouseCallback("Innovedex", mouse_cb)

def draw_spots(frame, live: dict):
    for sid, (sx, sy) in SPOTS.items():
        label   = live.get(sid)
        is_done = spot_done.get(sid, False)
        is_drag = (sid == dragging_spot)

        ov = frame.copy()
        rc = COLOR_BGR.get(label, (70,70,70)) if label else (70,70,70)
        cv2.circle(ov, (sx,sy), DETECT_RANGE, rc, 1)
        cv2.addWeighted(ov, 0.12, frame, 0.88, 0, frame)

        dot = (50,180,50) if is_done else COLOR_BGR.get(label,(70,70,70)) if label else (70,70,70)
        cv2.circle(frame, (sx,sy), SPOT_R_VIZ, dot, -1)
        cv2.circle(frame, (sx,sy), SPOT_R_VIZ+(3 if is_drag else 2), (255,255,255), 2)
        cv2.putText(frame, str(sid), (sx-4,sy+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        side = "R" if sid in [1,2,3] else "L"
        side_clr = (100,200,255) if side == 'R' else (255,200,100)
        cv2.putText(frame, side, (sx+SPOT_R_VIZ+2, sy-SPOT_R_VIZ),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, side_clr, 1)

        if is_done:  txt, tc = "DONE",           (80,200,80)
        elif label:  txt, tc = label.split()[0], COLOR_BGR.get(label,(200,200,200))
        else:        txt, tc = "empty",          (100,100,100)
        cv2.putText(frame, txt, (sx-22,sy+SPOT_R_VIZ+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, tc, 1)

        pp = PICK_POS_R.get(sid) or PICK_POS_L.get(sid) or {}
        if pp:
            cv2.putText(frame, f"B{pp['base']} A{pp['arm']}",
                        (sx-22,sy-SPOT_R_VIZ-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180,180,255), 1)

# ─── MAIN LOOP ────────────────────────────────────────────────────────────
print("[INFO] Ready!  Space=Start  R=Reset  S=Save  Q=Quit")
t_prev = time.perf_counter()

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" ") and not is_busy:
        spot_done = {i: False for i in SPOTS}
        is_busy = True; running = True
        threading.Thread(target=run_sequence, daemon=True).start()
    elif key == ord("r") and not is_busy:
        threading.Thread(target=home, daemon=True).start()
    elif key == ord("s"):
        save_config(); print("[INFO] Saved")
    elif key == ord("d") and not is_busy:
        if Path(CONFIG_FILE).exists():
            Path(CONFIG_FILE).unlink(); print("[INFO] config.json deleted")
    elif key == ord("z") and not is_busy:
        v = max(0,   servo_angles[SERVO_BASE]-5)
        board.servo_write(SERVO_BASE, v); servo_angles[SERVO_BASE] = v
        print(f"[TEST] BASE={v}")
    elif key == ord("x") and not is_busy:
        v = min(180, servo_angles[SERVO_BASE]+5)
        board.servo_write(SERVO_BASE, v); servo_angles[SERVO_BASE] = v
        print(f"[TEST] BASE={v}")
    elif key == ord("c") and not is_busy:
        v = max(0,   servo_angles[SERVO_ARM]-5)
        board.servo_write(SERVO_ARM, v); servo_angles[SERVO_ARM] = v
        print(f"[TEST] ARM={v}")
    elif key == ord("v") and not is_busy:
        v = min(180, servo_angles[SERVO_ARM]+5)
        board.servo_write(SERVO_ARM, v); servo_angles[SERVO_ARM] = v
        print(f"[TEST] ARM={v}")
    elif key == ord("b") and not is_busy:
        v = max(0,   servo_angles[SERVO_UP]-5)
        board.servo_write(SERVO_UP, v); servo_angles[SERVO_UP] = v
        print(f"[TEST] UP={v}")
    elif key == ord("n") and not is_busy:
        v = min(180, servo_angles[SERVO_UP]+5)
        board.servo_write(SERVO_UP, v); servo_angles[SERVO_UP] = v
        print(f"[TEST] UP={v}")
    elif key == ord("m") and not is_busy:
        board.servo_write(SERVO_GRIPPERL, GRIP_OPEN)
        servo_angles[SERVO_GRIPPERL] = GRIP_OPEN
        print("[TEST] GRIPL=open")
    elif key == ord(",") and not is_busy:
        board.servo_write(SERVO_GRIPPERL, GRIP_CLOSE)
        servo_angles[SERVO_GRIPPERL] = GRIP_CLOSE
        print("[TEST] GRIPL=close")
    elif key == ord(".") and not is_busy:
        board.servo_write(SERVO_GRIPPERR, GRIP_OPEN)
        servo_angles[SERVO_GRIPPERR] = GRIP_OPEN
        print("[TEST] GRIPR=open")
    elif key == ord("/") and not is_busy:
        board.servo_write(SERVO_GRIPPERR, GRIP_CLOSE)
        servo_angles[SERVO_GRIPPERR] = GRIP_CLOSE
        print("[TEST] GRIPR=close")
    elif key == ord("p"):
        print(f'\n[TEST] BASE={servo_angles[SERVO_BASE]} ARM={servo_angles[SERVO_ARM]} '
              f'UP={servo_angles[SERVO_UP]} GL={servo_angles[SERVO_GRIPPERL]} GR={servo_angles[SERVO_GRIPPERR]}')
        print(f'  → base:{servo_angles[SERVO_BASE]} arm:{servo_angles[SERVO_ARM]} up:{servo_angles[SERVO_UP]}\n')

    # ── อ่านภาพ — reconnect อัตโนมัติถ้ากล้องหลุด ────────────────────────
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Camera lost — reconnecting...")
        cap.release()
        time.sleep(1)
        cap = open_camera(CAMERA_ID)
        if cap is None:
            print("[ERROR] Camera reconnect failed")
            break
        continue

    results = model.predict(source=frame, conf=CONF, iou=IOU,
                            verbose=False, device="cuda:0")[0]

    live: dict = {sid: None for sid in SPOTS}
    if results.boxes is not None:
        for box in results.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cx = (x1+x2)//2; cy = (y1+y2)//2
            lbl  = CLASS_NAMES.get(int(box.cls[0]), "")
            conf = float(box.conf[0])
            clr  = COLOR_BGR.get(lbl, (200,200,200))
            cv2.rectangle(frame, (x1,y1),(x2,y2), clr, 2)
            cv2.putText(frame, f"{lbl} {conf:.2f}", (x1,y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr, 1)
            for sid, (sx,sy) in SPOTS.items():
                dist = ((cx-sx)**2+(cy-sy)**2)**0.5
                if (x1<=sx<=x2 and y1<=sy<=y2) or dist < DETECT_RANGE:
                    live[sid] = lbl; break

    for sid in SPOTS:
        if not spot_done.get(sid):
            spot_color[sid] = live[sid]

    draw_spots(frame, live)

    t_now  = time.perf_counter()
    fps    = 1.0 / max(t_now-t_prev, 1e-9)
    t_prev = t_now

    cv2.rectangle(frame, (0,0),(FRAME_W,26), (20,20,20), -1)
    if running:   status, sclr = "RUNNING", (0,220,100)
    else:         status, sclr = "READY",   (180,180,180)
    cv2.putText(frame,
                f"FPS:{fps:.0f}  |  {status}  |  Space=Start  R=Reset  S=Save  Q=Quit",
                (8,18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, sclr, 1)

    if not running:
        cv2.putText(frame,
            "Z/X=BASE  C/V=ARM  B/N=UP  M/,=GRIPL  ./=GRIPR  P=print  D=del",
            (8,FRAME_H-6), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120,120,255), 1)

    y0 = FRAME_H - 10 - len(SPOTS)*20
    cv2.rectangle(frame, (0,y0-20),(200,FRAME_H), (20,20,20), -1)
    cv2.putText(frame, "SPOT STATUS", (6,y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,0), 1)
    for i, sid in enumerate(sorted(SPOTS.keys())):
        lbl  = spot_color.get(sid)
        done = spot_done.get(sid, False)
        side = "R" if sid in [1,2,3] else "L"
        if done:   txt, clr = f"Spot{sid}[{side}]: DONE",             (80,200,80)
        elif lbl:  txt, clr = f"Spot{sid}[{side}]: {lbl.split()[0]}", COLOR_BGR.get(lbl,(200,200,200))
        else:      txt, clr = f"Spot{sid}[{side}]: empty",            (100,100,100)
        cv2.putText(frame, txt, (6,y0+16+i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, clr, 1)

    lx = FRAME_W - 220
    cv2.rectangle(frame, (lx-4,y0-20),(FRAME_W,FRAME_H), (20,20,20), -1)
    cv2.putText(frame, "PICK_POS R/L", (lx,y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,0), 1)
    all_pos = {**{f"R{k}":v for k,v in PICK_POS_R.items()}, **{f"L{k}":v for k,v in PICK_POS_L.items()}}
    for i, (lbl, pp) in enumerate(all_pos.items()):
        cv2.putText(frame, f"{lbl}: B{pp['base']:3d} A{pp['arm']:3d}",
                    (lx,y0+16+i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,255), 1)

    cv2.imshow("Innovedex", frame)

cap.release()
cv2.destroyAllWindows()
board.shutdown()
print("[INFO] Shutdown complete!")
