# FlowPOS 2.0

A computer-vision-based retail point-of-sale system for small shops. Camera 1 recognizes customers walking in and issues them a shopping PIN. Camera 2 scans their items at checkout using object detection and completes the sale. A browser dashboard now handles almost everything — customers and staff barely ever need to touch a keyboard or the terminal.

**Team:** Victoria, Edet, Abdullah, Ayo, Zakka, Doyin, Tobi, Bishop, Oyindamola
**Program:** NCAIR, Group 4, Cohort 35

---

## Table of Contents

1. [FlowPOS 1.0 — What It Was](#flowpos-10--what-it-was)
2. [FlowPOS 2.0 — What It Is Now](#flowpos-20--what-it-is-now)
3. [Technical Stack](#technical-stack)
4. [How It Operates, Step by Step](#how-it-operates-step-by-step)
5. [Real Performance Data](#real-performance-data)
6. [What's Working (Verified)](#whats-working-verified)
7. [Known Issues, Limitations & Unverified Behavior](#known-issues-limitations--unverified-behavior)
8. [Bugs Found and Fixed During Development](#bugs-found-and-fixed-during-development)
9. [Scalability & Future Improvements](#scalability--future-improvements)
10. [Project Structure](#project-structure)
11. [Setup & Running](#setup--running)

---

## FlowPOS 1.0 — What It Was

FlowPOS 1.0 was a continuous, always-on two-camera shelf-tracking system — conceptually the same category of approach as Amazon's Just Walk Out, and it ran into the same category of problem: expensive to run reliably, constant contention between two cameras trying to track the same moment, and unreliable event detection. It was retired rather than tuned further.

- Entry was controlled by a tripwire-style access pipeline, not true face recognition.
- Camera 2 (the item-scanning camera) could technically see items placed on the counter, but detection was never actually connected to billing. There was no real checkout flow — no total, no receipt, no inventory update.
- A dashboard existed, but it was dense and hard to use — too many panels and status indicators competing for attention at once.
- Almost every action depended on the terminal. The web UI, where it existed, was not the primary way anyone actually used the system.

FlowPOS 2.0 replaces continuous tracking with a two-checkpoint model instead: an entrance camera that recognizes-or-enrolls a customer and issues a shopping PIN, and a checkout camera that scans items and totals a purchase. The two checkpoints never need to agree on the same moment, which is what made v1 fragile.

---

## FlowPOS 2.0 — What It Is Now

### Entry (Camera 1) — Recognize or Enroll

- Camera 1 runs continuously, watching the entrance.
- MediaPipe detects a face in frame and checks that it's centered in a viewfinder-style guide box — this works at any distance, not just a narrow pixel-width range like earlier versions relied on.
- Once centered and held steady for about a second, the frame is cropped, enhanced (CLAHE contrast preprocessing), and passed to ArcFace (via the DeepFace library) to produce a face embedding — a numeric fingerprint of that face.
- That embedding is compared against every enrolled customer's stored centroid embedding using cosine/centroid similarity.
- **Match found (similarity ≥ 0.6):** the customer is welcomed by ID, and a 4-digit shopping PIN is issued for that visit, shown on screen.
- **No match:** the system hands off to Enrollment.

### Enrollment — New Customer Signup

- No name or ID is typed by anyone. A Customer ID (SI-001, SI-002, …) is generated automatically and used as the customer's identifier everywhere — there's no separate "name" field.
- **Liveness check first, and only here.** Before any photos are captured, the customer is given a randomized head-turn challenge (left/right/up/down), tracked using MediaPipe Face Mesh and head-pose estimation (solvePnP-based). This exists specifically to stop someone enrolling using a photo or video of somebody else — a static image can't follow a randomized live prompt. Returning customers are never challenged again after this; recognizing them on later visits only compares their stored embedding.
- **Five-pose capture.** Once liveness passes, five photos are captured — straight, left, right, up, down — each requiring the face to be centered and held steady before it's captured.
- **Duplicate detection.** The five captured embeddings are averaged and checked against every existing enrolled customer. If the match is close enough to suggest this might be the same person re-enrolling, the enrollment is **not** completed automatically — it's parked for admin review under Admin Panel → Review Flagged Enrollment, along with a saved photo so staff can make the call, rather than silently creating a second identity for the same person.
- On success, a reference photo is saved to disk and the customer's embedding, centroid, and photo location are all stored — this photo later shows up next to their transactions in the Admin Panel.
- A shopping PIN is issued immediately, the same as for a returning customer.

### Checkout (Camera 2) — Scan, Confirm, Pay

- The customer enters their PIN (in the browser, not the terminal) to start a checkout session.
- Entry is automatically paused while checkout is running — the two cameras never compete for attention at the same time.
- Camera 2 continuously scans whatever is placed in view using a custom-trained YOLOv8 model (see [Real Performance Data](#real-performance-data)) — roughly every 0.6 seconds.
- Once the same set of items is detected 3 times in a row, the scan is auto-confirmed. A manual "Confirm Now" override exists for cases where auto-stabilization is taking a while.
- A review screen shows the detected items and the total before anything is charged.
- The customer (or staff, on their behalf) chooses **Accept & Pay**, **Rescan** (if the detected items are wrong), or **Cancel**.
- On a completed sale: a full itemized transaction record is saved (customer, items, quantities, total, timestamp, session duration), inventory stock is decremented automatically per item sold, and the customer's own purchase count and total spent are updated.

### Admin Panel

Everything below is reachable from the browser — no terminal required for any of it:

- **Overview** — live status (is entrance active, how many people are shopping right now), today's sales and transaction count, a real-time recent-activity feed, and quick previews of inventory and camera status.
- **Cameras** — set where each camera's video comes from (blank = laptop webcam, or paste an IP Webcam address for a phone camera). This is the single place both Entry's and Checkout's camera sources are configured.
- **Customers** — every enrolled customer, with enrollment date, total purchases, and total spent. Customers can be deleted here, which also removes their saved photo and embedding.
- **Transactions** — every completed sale, each row showing a small photo of the customer (or their initials, if no photo is on file) right alongside their name/ID, items purchased, time, and total — so a transaction can be visually matched back to who actually made it, useful for reviewing a suspected underpayment or shoplifting incident after the fact.
- **Inventory** — real-time stock levels and pricing for every item, with low-stock items visually flagged. Stock can be added to, set to an exact quantity, or repriced, all from the browser, and decrements automatically after every sale.
- **System Info** — a read-only summary of what's actually running: camera sources, which face-detection/recognition/item-detection models are in use, the database, and the currency.

### Web Dashboard — The Biggest Structural Change

- `python main.py` starts both the terminal menu **and** a Flask web server together, in one command.
- Every camera window that used to pop up as a native OpenCV window is instead streamed live into the browser.
- Every place the code used to pause and wait for terminal input (a PIN, a camera address, a retry prompt) is instead caught and re-shown as a popup with a text box in the browser — the underlying, hardware-tested logic didn't need to change to make this work.
- Every place the code waited for a keypress (Accept/Rescan/Cancel/Confirm during checkout) is a set of real, labeled buttons in the browser, translated behind the scenes into the same keypress the code already expects.

---

## Technical Stack

| Purpose | Tool |
|---|---|
| Face detection | MediaPipe |
| Face recognition | ArcFace (via the DeepFace library) |
| Liveness / head-pose tracking | MediaPipe Face Mesh + head-pose estimation |
| Item detection | YOLOv8 (custom-trained, nano variant) |
| Face recognition backend | TensorFlow (via DeepFace) |
| Camera capture | OpenCV |
| Phone camera support | IP Webcam (Android app) or similar, over local network |
| Web backend | Flask |
| Database | SQLite |
| Web frontend | Plain HTML/CSS/JavaScript (no framework) |

---

## How It Operates, Step by Step

**A returning customer:**
1. Walks up to Camera 1. Face is detected, centered, held steady.
2. Face is matched against enrolled customers (≥ 0.6 similarity).
3. Welcomed by ID, given a 4-digit PIN.
4. Walks to the counter, opens Checkout in the browser, enters the PIN.
5. Places items in view of Camera 2. Items are detected and stabilize after 3 consistent reads.
6. Reviews the item list and total, taps **Accept & Pay**.
7. Sale is saved, inventory decrements, transaction record is created, session ends.

**A brand-new customer:**
1. Walks up to Camera 1. No match is found.
2. Hands off to Enrollment: liveness check, then five-pose photo capture.
3. If the face is too similar to an existing customer, enrollment is parked for admin review instead of completing.
4. Otherwise, a Customer ID is generated, a reference photo and embedding are saved, and a PIN is issued — same checkout flow as above from here.

---

## Real Performance Data

### Item Detection (YOLOv8) — Formally Measured

This model was trained and evaluated on a real, held-out validation set — these numbers come directly from that training run, not an estimate:

- **Model:** YOLOv8n (nano variant) — 3.0 million parameters, 6.3MB final weights
- **Dataset:** 2,484 training images, 235 validation images, sourced via Roboflow
- **Classes (4):** coke, fanta, sprite, water
- **Training:** 100 epochs configured, early-stopped at epoch 87 (best weights from epoch 67), roughly 55 minutes on a Tesla T4 GPU
- **Validation results:** Precision **99.7%**, Recall **98.6%**, mAP@50 **99.5%**, mAP@50–95 **81.2%**
- Every individual class independently scores above 99% mAP@50, with no significant confusion between classes visible in the confusion matrix.

### Face Recognition (ArcFace) — Observed, Not Formally Measured

- Matching uses a fixed similarity threshold of **0.6**.
- Real successful matches observed during testing consistently scored between roughly **0.65 and 0.80** — comfortably above the threshold.
- **This threshold has not been through a formal accuracy study.** There is no labeled test set, and no computed false-accept rate or false-reject rate. 0.6 works in the testing done so far, but "works in testing" and "formally validated" are not the same claim, and this system does not yet make the second one.
- Related: liveness and detection timing constants (hold times, stability windows, detection confidence thresholds) are reasonable starting values chosen during development, not values tuned against a large body of real footage.

---

## What's Working (Verified)

Everything below has been tested end-to-end, either through direct code execution or confirmed on real hardware:

- Real-time face detection and recognition at the entrance, including correct handling of returning customers, new customers, and the guide-box centering logic.
- Full enrollment flow: liveness challenge, five-pose capture, duplicate detection and flagging, photo and embedding saving.
- Full checkout flow on real hardware, using both a laptop webcam and a phone via IP Webcam: PIN entry, item scanning and stabilization, review, accept, a real saved transaction, and inventory decrement.
- The entire web dashboard: live camera streaming into the browser for both cameras, the PIN/input-prompt popup system, real button-driven checkout actions, and every Admin Panel page (Overview, Cameras, Customers, Transactions with photos, Inventory with editing, System Info).
- Stopping the entrance engine (both the 'Q' key and the equivalent web/menu control) genuinely releases the camera instead of silently continuing.
- `python main.py` starting the terminal and the web dashboard together in one command.

---

## Known Issues, Limitations & Unverified Behavior

- **No login or authentication on the Admin Panel or web dashboard.** Anyone who can reach the dashboard's address on the network can view customer data, edit inventory, or delete a customer — there's no access gate in front of any of it yet.
- **Face recognition threshold (0.6) is not formally validated** — real, but not rigorous. No false-accept/false-reject study exists yet.
- **Item catalog is small.** Only 4 SKUs are currently trained (coke, fanta, sprite, water). A real shop's full shelf would need proportionally more training data per new item added.
- **Single camera per role.** The system assumes one entrance camera and one checkout camera — not built for multiple simultaneous checkout lanes or multiple entrances.
- **Lighting-dependent.** Like any camera-based vision system, both recognition and detection accuracy depend on reasonably consistent lighting. Not yet stress-tested under poor or highly variable lighting.
- **Guide-box directional hints are unverified for a mirrored camera preview.** If a camera's preview is flipped (like a selfie camera), the "move left/move right" hints during enrollment would read backwards — not yet confirmed against a real mirrored preview. The liveness challenge's own left/right convention carries the same caveat.
- **Customers enrolled before the reference-photo fix have no photo on file** and will show initials instead of a real photo on the Transactions page until they're re-enrolled.
- **Two separate paths can start the entrance camera** — one from `main.py` directly, one from the web dashboard's "Start Entrance" button. Not yet unified into a single path.
- **Checkout's action buttons are all shown together the entire time checkout is running**, since the backend doesn't yet expose a distinct "still scanning" vs. "reviewing the result" signal the browser can read. Safe — an early click is simply ignored by the underlying code — but not fully precise.
- **The checkout camera image still shows on-screen text from before the web UI existed** (e.g. "Press 'A' to Accept"), baked directly into the video frame by the scanning code. The real, working controls are the browser buttons next to it; the text is a harmless leftover.

---

## Bugs Found and Fixed During Development

A full, honest record of what was actually broken and how it was resolved — every one of these was found by running the real code (against a real camera, or a mocked pipeline built specifically to reproduce the failure), not by code review alone, and re-verified after the fix:

**Environment & dependency stability:**
1. `mediapipe.solutions` missing at runtime — an unpinned install pulled a MediaPipe version that had dropped the API the whole face pipeline depends on. Pinned to a confirmed-working version.
2. `opencv-python` and `opencv-contrib-python` installed together silently corrupt the shared `cv2` module both packages provide. Pinned to `opencv-contrib-python` only.
3. A protobuf version mismatch after an interrupted TensorFlow-related install, fixed by reinstalling matching, version-compatible packages instead of the newest available (which would have re-triggered the same problem).

**Core recognition & enrollment logic:**
4. A camera-resource leak in the entrance loop — the face-detection model used to be created *before* the `try`/`finally` block that releases the camera, so a setup-time crash could leave the camera handle open indefinitely. Fixed by moving camera-dependent setup inside the `try` block.
5. A shared duplicate-detection threshold wasn't actually shared — a circular import between two files made one of them silently fall back to its own separate hardcoded default, invisible until checked with an identity comparison instead of an equality one. Fixed with a lazy import inside the function that needs it.
6. An undefined color constant used in the entrance loop would crash it with a `NameError` on the very first frame, before a single face was ever detected.
7. A crash that only happened on a *successful* face-crop detection — code used to fall back to a default value the same way for both "no result" and "a real result," but a real multi-element image array's truthiness is undefined in Python, so the moment detection actually succeeded, it crashed. Fixed with an explicit "is this actually empty?" check.
8. Pressing 'Q' or using "Stop Entrance Engine" used to silently do nothing — only force-quitting the whole program actually stopped the camera. The stop signal was being set in one place but checked in another that never heard about it. Fixed so the request is properly delivered and acknowledged, confirmed against real hardware afterward.
9. The face-detection model used during enrollment's five-photo capture used to be recreated from scratch on every single frame instead of being reused once for the whole sequence — wasteful, and the source of visible lag during capture. Fixed to reuse one model for the whole enrollment.

**Web dashboard integration:**
10. Inventory could be viewed from the browser but never edited — add-stock, set-exact-quantity, and set-price actions were missing from the web dashboard entirely.
11. Text drawn near the edge of the camera image (like the welcome/PIN message) was being cropped out of view in the browser due to how the image was being fitted into its display box.
12. No distinction existed between "camera still starting up" and "camera not running at all" — both showed the same message, making a genuine, expected one-time model-loading delay look like a bug.
13. The welcome/PIN success message was hard to read (green text against certain backgrounds) — changed to black for that specific message.
14. "Shopping right now" count was wildly inflated (found at 74, should have been near 0–2) — every time an already-shopping customer was re-recognized at the entrance, a brand-new session was created instead of reusing their existing one, and nothing ever expired an old one except a completed checkout. Fixed so an existing active session is reused instead of duplicated.
15. The most serious bug found in this project: checkout would connect, show exactly one real frame with real item detections, then instantly cancel. A stop signal meant only for the entrance camera was being read by checkout's scanning loop too, because both cameras shared the same underlying keypress-reading mechanism — if entrance had ever been stopped and not restarted before a checkout began, checkout would see that leftover signal and cancel immediately, as if someone had pressed Cancel. Reproduced directly with the real scanning code, fixed, reproduced again to confirm the fix held, then confirmed working on real hardware with both a laptop webcam and an IP Webcam.
16. Checkout's camera address used to be a separate, hidden setting disconnected from the Cameras page in the Admin Panel — changing "Camera 2" there had no effect on checkout, and a stale phone IP address required a full server restart to fix.
17. A normal, successful enrollment never actually saved a reference photo — only the rare "flagged as possible duplicate" path did. Every ordinary enrollment was leaving the photo field empty. Fixed so every successful enrollment now saves a real photo, the same way the duplicate-review case already did.

---

## Scalability & Future Improvements

**From the item-detection model's own training notes:**
- Grow past the current 4 SKUs — aim for 100+ training images per new item, ideally 300+, matching what the existing 4 items were trained on.
- Keep checking the confusion matrix as the catalog grows, to catch visually similar packaging before it becomes a real accuracy problem.
- Export the model to ONNX or TFLite for lighter-weight deployment on smaller hardware.
- Add data augmentation if overfitting starts to appear as more training data is added (not currently a problem — the real training curves showed clean convergence).

**System-level:**
- Add authentication to the Admin Panel and web dashboard — currently open to anyone who can reach it.
- Run a real, formal accuracy study for face recognition — a labeled test set, computed false-accept and false-reject rates — rather than relying on informal observation of the 0.6 threshold.
- Add a "block/ban" status for a customer, distinct from permanently deleting their record.
- Support multiple simultaneous checkout lanes and/or multiple entrances for busier locations.
- Unify the two separate entrance-start paths (`main.py`'s own prompt and the web dashboard's button) into one.
- Give checkout's scanning state a real "still scanning" vs. "reviewing" signal the browser can read, instead of showing all actions at once.
- Build a re-enrollment or photo-backfill path for customers enrolled before the photo-saving fix.
- Stress-test recognition and detection accuracy under a wider range of real-world lighting conditions.

---

## Project Structure
FlowPOS_2.0/
├── main.py # Entry point — starts terminal menu + web dashboard together
├── sis_server.py # Flask backend for the web dashboard
├── index.html # Web dashboard frontend
├── camera_stream.py # Camera connection handling (webcam / IP Webcam)
├── face_db.py # Customer + transaction database access
├── embedding_loader.py # In-memory face embedding cache
├── duplicate_checker.py # Duplicate-enrollment detection
├── inventory.db # SQLite database
├── models/
│ └── yolov8n.pt # Trained item-detection model
├── customers/ # Saved customer reference photos
├── enrollment/
│ ├── entrance.py # Camera 1 — recognize-or-enroll loop
│ ├── enrollment.py # New customer enrollment flow
│ └── liveness.py # Liveness / head-pose challenge
├── checkout/
│ └── checkout.py # Camera 2 — item scanning and payment flow
├── services/
│ ├── enrollment_manager.py # Staff ID generation, startup, flagged-duplicate storage
│ ├── shopping_session.py # PIN-based shopping session management
│ └── inventory.py # Stock and pricing management
├── m4/
│ └── price_catalog.py # Item price list
└── admin_panel.py # Terminal-based admin panel (legacy — most functions now also in the web dashboard)


---

## Setup & Running

```powershell
python main.py
```

This single command starts both the terminal menu and the web dashboard together.

Open the dashboard in a browser:

http://localhost:5000


On first use, set your camera sources under **Admin Panel → Cameras** — leave a field blank to use the laptop's built-in webcam, or enter a phone's IP Webcam address to use that instead.
