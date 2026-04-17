============================================================
  GlassHouse v2 -- CSI Data Capture Guide
============================================================

Hi! This folder has everything you need to collect WiFi CSI
data from the GlassHouse sensor nodes. Follow the steps below.

YOU WILL BE DOING THIS ALONE, so read the "Solo Capture"
section carefully -- the script has a countdown timer so
you can start it and walk into position before it records.


------------------------------------------------------------
  STEP 1: Install Python (if not already installed)
------------------------------------------------------------

You need Python 3.10 or newer. Check by opening a terminal:

    python --version

If not installed, download from https://www.python.org/downloads/
  -> CHECK "Add Python to PATH" during install!


------------------------------------------------------------
  STEP 2: Install Dependencies
------------------------------------------------------------

Open a terminal IN THIS FOLDER (right-click > "Open in Terminal"
or cd to this folder), then run:

    pip install -r requirements.txt

This installs two small libraries (pyserial for the USB port,
cobs for decoding the data frames).


------------------------------------------------------------
  STEP 3: Hardware Setup
------------------------------------------------------------

You should have:
  - 1 coordinator ESP32 (the one with the USB cable to your laptop)
  - 4 perimeter ESP32 nodes (powered by USB power banks or wall plugs)

Setup:
  1. Place the 4 perimeter nodes at the 4 corners of the room
  2. Plug in / power on ALL 4 perimeter nodes
  3. Plug the coordinator into your laptop via USB
  4. Wait ~15 seconds for everything to boot and connect

Find the COM port:
  - Windows: Open Device Manager > "Ports (COM & LPT)"
    Look for "USB-SERIAL" or "USB JTAG" -- note the COM number
    (e.g., COM9)
  - Mac/Linux: Run "ls /dev/tty*" and look for ttyUSB0 or ttyACM0


------------------------------------------------------------
  STEP 4: Quick Test (do this first!)
------------------------------------------------------------

Before collecting real data, make sure packets are flowing:

    python -m debug.capture --port COM9 --seconds 10 --label test

You should see a progress line like:

    3.2s  pkts=47  links=40  vitals=5  iq=2

If pkts stays at 0, see Troubleshooting below.
Once this works, delete captures/capture_test.jsonl and continue.


------------------------------------------------------------
  STEP 5: Collect Data (Solo Capture Workflow)
------------------------------------------------------------

We need 5 captures, each 10 MINUTES long.
The --delay flag gives you a countdown to get into position.

*** HOW SOLO CAPTURE WORKS ***

  1. You type the command and press Enter
  2. The script prints a countdown: "Starting in 20s... 19s..."
  3. You walk to your position and stand still
  4. When the countdown hits 0, it starts recording
  5. Stand still for the full 10 minutes
  6. The script stops automatically and says "Done"
  7. Walk back to the laptop for the next capture

You will hear/see nothing while it records -- that's normal.
Just stay in position until you see "Done" on the screen
(walk back and check), or wait the full 10 minutes.

------------------------------------------------------------

--- Capture 1: EMPTY ROOM ---

  This one is different -- nobody in the room at all!

  1. Type this command but DON'T press Enter yet:

     python -m debug.capture --port COM9 --seconds 600 --label empty --delay 20

  2. Press Enter, then LEAVE THE ROOM and close the door
  3. Wait 11 minutes (10 min capture + the 20s countdown)
  4. Come back -- it should say "Done"

--- Capture 2: Standing in Q1 ---

    python -m debug.capture --port COM9 --seconds 600 --label occupied_q1 --delay 20

  Press Enter, walk to Q1 (see map below), stand still for
  10 minutes. Come back and check "Done" on screen.

--- Capture 3: Standing in Q2 ---

    python -m debug.capture --port COM9 --seconds 600 --label occupied_q2 --delay 20

--- Capture 4: Standing in Q3 ---

    python -m debug.capture --port COM9 --seconds 600 --label occupied_q3 --delay 20

--- Capture 5: Standing in Q4 ---

    python -m debug.capture --port COM9 --seconds 600 --label occupied_q4 --delay 20


Total time: about 55-60 minutes for all 5 captures.


------------------------------------------------------------
  QUADRANT MAP
------------------------------------------------------------

  Stand roughly in the CENTER of each quadrant.

       Node 1 ─────────────── Node 2
         │                      │
         │    Q1    │    Q2     │
         │          │           │
         │──────────┼───────────│
         │          │           │
         │    Q3    │    Q4     │
         │                      │
       Node 3 ─────────────── Node 4

  The coordinator (plugged into your laptop) sits near one
  of the nodes or off to the side -- it doesn't matter where.


------------------------------------------------------------
  STEP 6: Verify Your Captures
------------------------------------------------------------

After all 5 captures, check the "captures" folder. You should
see these files:

    captures/capture_empty.jsonl
    captures/capture_occupied_q1.jsonl
    captures/capture_occupied_q2.jsonl
    captures/capture_occupied_q3.jsonl
    captures/capture_occupied_q4.jsonl

Each file should be at least 500 KB for a 10-minute capture.
If a file is tiny (under 10 KB), something went wrong --
see Troubleshooting.


------------------------------------------------------------
  STEP 7: Return the Data
------------------------------------------------------------

Copy the entire "captures" folder back (USB stick, email, etc.).
That's it!


------------------------------------------------------------
  TROUBLESHOOTING
------------------------------------------------------------

"No module named serial"
  -> Run: pip install pyserial
     (Note: it's pyserial, NOT serial)

"No module named cobs"
  -> Run: pip install cobs

"could not open port COM9"
  -> Wrong COM port. Check Device Manager for the right one.
  -> Replace COM9 with your actual port in ALL commands, e.g.:
     python -m debug.capture --port COM7 --seconds 600 --label empty --delay 20

"Permission denied" on the port
  -> Close any other program using the serial port
     (Arduino IDE, PuTTY, another terminal, etc.)

File is 0 KB / no packets captured
  -> Make sure all 4 perimeter nodes AND the coordinator are
     powered on and have had ~15 seconds to connect
  -> Redo the quick test from Step 4
  -> If test also gets 0 packets, check the USB cable and try
     a different USB port on your laptop

"pkts=0" shown while capturing
  -> The nodes might still be booting. Wait 15-20 seconds
     after power-on, then try again.

Script crashes immediately
  -> Make sure you are running the command FROM THIS FOLDER
     (the folder containing this README), not from inside
     the debug/ or python/ subfolder.

A capture was interrupted / you had to stop early
  -> Just re-run the same command. It will overwrite the old file.
  -> Make sure you stand still for the FULL 10 minutes.

You accidentally moved during a capture
  -> Re-run that capture. Moving corrupts the data for that
     quadrant. It's better to redo it than to guess.


------------------------------------------------------------
  QUICK REFERENCE
------------------------------------------------------------

  Command format:
    python -m debug.capture --port COMX --seconds 600 --label LABEL --delay 20

  Labels (use exactly as written):
    empty           <- nobody in the room
    occupied_q1     <- person standing in Q1
    occupied_q2     <- person standing in Q2
    occupied_q3     <- person standing in Q3
    occupied_q4     <- person standing in Q4

  Settings:
    Port:     COM9  (change with --port to match YOUR port)
    Duration: 600s  (= 10 minutes)
    Delay:    20s   (countdown before recording starts)
    Baud:     921600 (don't change this)

============================================================
  Questions? Text Jack.
============================================================
