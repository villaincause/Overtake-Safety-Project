/*
  Overtake Safety - LED display for Arduino Nano

  The PC runs the camera + model and sends one character over USB serial:
      '1' = GREEN   no vehicle ahead, safe to pass
      '2' = SMALL   small vehicle ahead   (bikes, bicycle)   - spare LED for now
      '3' = BLUE    medium vehicle ahead  (car, CNG, rickshaw, van ...)
      '4' = RED     large vehicle ahead   (bus, pickup, truck)
      '0' = all LEDs off (PC program closed normally)

  The PC repeats the current code every 0.5 s. If nothing arrives for
  TIMEOUT_MS (PC crashed, cable pulled, camera frozen) ALL LEDs blink together,
  so a dead system is never mistaken for a green "safe" light.

  Testing by hand: open the Serial Monitor at 9600 baud and send 1, 2, 3, 4 or 0.
  The Serial Monitor also shows a line each time the code changes.

  Wiring: each LED long leg (+) -> 220 ohm resistor -> pin, short leg (-) -> GND.
*/

// ----- change these if you wire the LEDs to different pins ------------------
const uint8_t PIN_GREEN = 2;   // code 1  clear
const uint8_t PIN_SMALL = 3;   // code 2  small vehicles (any spare LED until the 4th color arrives)
const uint8_t PIN_BLUE  = 4;   // code 3  medium vehicles
const uint8_t PIN_RED   = 5;   // code 4  large vehicles
// -----------------------------------------------------------------------------

const unsigned long BAUD       = 9600;   // must match the PC program (--baud, default 9600)
const unsigned long TIMEOUT_MS = 3000;   // no message for this long = "no signal"
const unsigned long BLINK_MS   = 400;    // blink speed in "no signal" state

const uint8_t LED_PINS[4] = {PIN_GREEN, PIN_SMALL, PIN_BLUE, PIN_RED};  // index 0..3 = code 1..4

int currentCode = 0;
bool everReceived = false;
unsigned long lastMessageAt = 0;

void setAll(bool on) {
  for (uint8_t i = 0; i < 4; i++) digitalWrite(LED_PINS[i], on ? HIGH : LOW);
}

void showCode(int code) {
  for (uint8_t i = 0; i < 4; i++) digitalWrite(LED_PINS[i], (code == i + 1) ? HIGH : LOW);
}

void setup() {
  Serial.begin(BAUD);
  for (uint8_t i = 0; i < 4; i++) pinMode(LED_PINS[i], OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);

  // Lamp test: light each LED once so you can check the wiring without the PC.
  for (uint8_t i = 0; i < 4; i++) {
    digitalWrite(LED_PINS[i], HIGH);
    delay(300);
    digitalWrite(LED_PINS[i], LOW);
  }
  Serial.println("Overtake LEDs ready. Send 1=green 2=small 3=blue 4=red 0=off");
}

void loop() {
  // Read everything that arrived; keep the newest valid code.
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c >= '0' && c <= '4') {
      int code = c - '0';
      if (!everReceived || code != currentCode) {   // print only on change, not every heartbeat
        Serial.print("Code ");
        Serial.println(code);
      }
      currentCode = code;
      lastMessageAt = millis();
      everReceived = true;
    }
  }

  bool signalOk = everReceived && (millis() - lastMessageAt < TIMEOUT_MS);
  digitalWrite(LED_BUILTIN, signalOk ? HIGH : LOW);   // board's own LED = "PC link is alive"

  if (everReceived && currentCode == 0) {
    setAll(false);                                    // deliberate shutdown: stay dark
  } else if (signalOk) {
    showCode(currentCode);
  } else {
    setAll((millis() / BLINK_MS) % 2 == 0);           // no signal: blink all together
  }
}
