#include <LiquidCrystal.h>

LiquidCrystal lcd{8, 11, 6, 7, 1, 0};

void setup() {
  lcd.begin(16,2);
}

void loop() {
 lcd.print("Hello Ahmad!");
 delay(150);
 lcd.clear();
 delay(150);

 


}
