
#include <LiquidCrystal.h>

LiquidCrystal lcd{12, 11, 5, 4, 3, 2};


void setup() {
    lcd.begin(16,2);
pinMode(8,OUTPUT);//Defining the pin number (GREEN LED), Defining it's OUTPUT or INPUT
pinMode(9,OUTPUT);//Defining the pin number (RED LED), Defining it's OUTPUT or INPUT


}

void loop() {


lcd.print("FUCK YOU");
digitalWrite(8,HIGH);
digitalWrite(9,LOW);
delay(500);

digitalWrite(8,LOW);
digitalWrite(9,HIGH);
lcd.clear();
delay(500);
     
}
