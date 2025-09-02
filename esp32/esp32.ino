#include <Adafruit_NeoPixel.h>

// ---- Pins (your wiring)
const int STBY = 23;
const int AIN1 = 25, AIN2 = 26, PWMA = 27;   // Left/M1
const int BIN1 = 32, BIN2 = 33, PWMB = 14;   // Right/M2
const int LED_PIN = 18;
const int NUM_LEDS = 6;

// ---- LED strip
Adafruit_NeoPixel strip(NUM_LEDS, LED_PIN, NEO_GRB + NEO_KHZ800);

// ---- PWM for TB6612 (quiet)
const int PWM_FREQ = 20000;   // 20 kHz
const int PWM_RES  = 8;       // 0..255
const int CH_LEFT  = 0;
const int CH_RIGHT = 1;

void setEmotion(const String& e) {
  uint32_t c = strip.Color(0,0,0);
  if (e=="HAPPY")      c = strip.Color(0,150,40);
  else if (e=="EXCITED") c = strip.Color(255,40,0);
  else if (e=="SLEEPY")  c = strip.Color(0,0,80);
  else if (e=="CONFUSED")c = strip.Color(120,0,120);
  for (int i=0;i<NUM_LEDS;i++) strip.setPixelColor(i,c);
  strip.show();
}

void motorsStandby(bool en) { digitalWrite(STBY, en ? HIGH : LOW); }

void setMotorLR(int left, int right) {
  left  = constrain(left,  -255, 255);
  right = constrain(right, -255, 255);

  if (left >= 0) { digitalWrite(AIN1, HIGH); digitalWrite(AIN2, LOW); }
  else           { digitalWrite(AIN1, LOW);  digitalWrite(AIN2, HIGH); }

  if (right >= 0){ digitalWrite(BIN1, HIGH); digitalWrite(BIN2, LOW); }
  else           { digitalWrite(BIN1, LOW);  digitalWrite(BIN2, HIGH); }

  ledcWrite(CH_LEFT,  abs(left));
  ledcWrite(CH_RIGHT, abs(right));
  motorsStandby( (abs(left)+abs(right)) > 0 );
}

void allStop() { setMotorLR(0,0); }

// MOVE:<dir>:<speed>:<ms>  dir=F,B,L,R,S  speed=0..255
void cmdMOVE(const String& line) {
  int p1=line.indexOf(':'), p2=line.indexOf(':', p1+1), p3=line.indexOf(':', p2+1);
  if (p1<0||p2<0||p3<0) { Serial.println("ERR:MOVE format"); return; }
  String dir=line.substring(p1+1,p2);
  int speed=line.substring(p2+1,p3).toInt();
  int ms=line.substring(p3+1).toInt();
  speed = constrain(speed,0,255);

  int L=0,R=0;
  if (dir=="F")      { L= speed; R= speed; }
  else if (dir=="B") { L=-speed; R=-speed; }
  else if (dir=="L") { L=-speed; R= speed; }
  else if (dir=="R") { L= speed; R=-speed; }
  else if (dir=="S") { L=0; R=0; }
  else { Serial.println("ERR:MOVE dir"); return; }

  setMotorLR(L,R);
  if (ms>0) { delay(ms); allStop(); }
  Serial.println("ACK:MOVE");
}

void cmdSPIN(const String& line) {
  int p=line.indexOf(':');
  int ms = (p>0) ? line.substring(p+1).toInt() : 500;
  setMotorLR(255,255);
  delay(ms);
  allStop();
  Serial.println("ACK:SPIN");
}

void cmdEMO(const String& line) {
  String e = line.substring(4); e.trim();
  setEmotion(e);
  Serial.println("ACK:EMO");
}

void setup() {
  Serial.begin(115200);

  pinMode(STBY, OUTPUT);
  pinMode(AIN1, OUTPUT); pinMode(AIN2, OUTPUT);
  pinMode(BIN1, OUTPUT); pinMode(BIN2, OUTPUT);

  ledcSetup(CH_LEFT,  PWM_FREQ, PWM_RES);
  ledcSetup(CH_RIGHT, PWM_FREQ, PWM_RES);
  ledcAttachPin(PWMA, CH_LEFT);
  ledcAttachPin(PWMB, CH_RIGHT);

  strip.begin(); strip.show();
  setEmotion("CONFUSED");
  allStop();
  motorsStandby(true);

  Serial.println("READY");
}

void loop() {
  static String line;
  while (Serial.available()) {
    char ch=Serial.read();
    if (ch=='\n' || ch=='\r') {
      line.trim();
      if (line.length()) {
        if      (line.startsWith("EMO:"))  cmdEMO(line);
        else if (line.startsWith("SPIN:")) cmdSPIN(line);
        else if (line.startsWith("MOVE:")) cmdMOVE(line);
        else Serial.println("ERR:UNKNOWN");
      }
      line="";
    } else {
      line += ch;
      if (line.length()>160) line=""; // flood guard
    }
  }
}
