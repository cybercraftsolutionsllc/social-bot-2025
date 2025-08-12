#include <Adafruit_NeoPixel.h>

// Pins (mapping from your build)
const int STBY = 23;
const int AIN1 = 25, AIN2 = 26, PWMA = 27;   // Left/M1
const int BIN1 = 32, BIN2 = 33, PWMB = 14;   // Right/M2
const int LED_PIN = 18;
const int NUM_LEDS = 6;

Adafruit_NeoPixel strip(NUM_LEDS, LED_PIN, NEO_GRB + NEO_KHZ800);

void setEmotion(const String& e) {
  uint32_t c = strip.Color(0,0,0);
  if (e=="HAPPY")   c = strip.Color(0,150,40);
  else if (e=="EXCITED") c = strip.Color(255,40,0);
  else if (e=="SLEEPY")  c = strip.Color(0,0,80);
  else if (e=="CONFUSED")c = strip.Color(120,0,120);
  for (int i=0;i<NUM_LEDS;i++) strip.setPixelColor(i,c);
  strip.show();
}

void motorsStop(){
  digitalWrite(PWMA, LOW); digitalWrite(PWMB, LOW);
  digitalWrite(AIN1, LOW); digitalWrite(AIN2, LOW);
  digitalWrite(BIN1, LOW); digitalWrite(BIN2, LOW);
}

void motorsSpinMs(int ms){
  digitalWrite(STBY, HIGH);
  digitalWrite(AIN1, HIGH); digitalWrite(AIN2, LOW); digitalWrite(PWMA, HIGH);
  digitalWrite(BIN1, HIGH); digitalWrite(BIN2, LOW); digitalWrite(PWMB, HIGH);
  delay(ms);
  motorsStop();
}

void setup() {
  Serial.begin(115200);
  pinMode(STBY, OUTPUT);
  pinMode(AIN1, OUTPUT); pinMode(AIN2, OUTPUT); pinMode(PWMA, OUTPUT);
  pinMode(BIN1, OUTPUT); pinMode(BIN2, OUTPUT); pinMode(PWMB, OUTPUT);
  digitalWrite(STBY, HIGH);
  strip.begin(); strip.show();
  setEmotion("CONFUSED");
}

void loop() {
  static String line;
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch=='\n' || ch=='\r') {
      line.trim();
      if (line.startsWith("EMO:")) {
        setEmotion(line.substring(4));
      } else if (line.startsWith("SPIN:")) {
        motorsSpinMs(line.substring(5).toInt());
      }
      line = "";
    } else {
      line += ch;
    }
  }
}
