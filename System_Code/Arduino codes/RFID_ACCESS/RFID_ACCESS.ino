//Viral Science
//RFID
#include <SPI.h>
#include <MFRC522.h>
#include <Servo.h>
 
#define SS_PIN 10
#define RST_PIN 9
#define LED_G 4 //define green LED pin
#define LED_R 5 //define red LED
#define BUZZER 2 //buzzer pin
MFRC522 mfrc522(SS_PIN, RST_PIN);   // Create MFRC522 instance.
Servo myServo; //define servo name

void setup() 
{
  Serial.begin(9600);   // Initiate a serial communication
  SPI.begin();      // Initiate  SPI bus
  mfrc522.PCD_Init();   // Initiate MFRC522
  myServo.attach(3); //servo pin
  myServo.write(0); //servo start position
  pinMode(LED_G, OUTPUT);
  pinMode(LED_R, OUTPUT);
  pinMode(BUZZER, OUTPUT);
  noTone(BUZZER);
  Serial.println("Put your card to the reader...");
  Serial.println();

}
void loop() 
{
  // Look for new cards
  if ( ! mfrc522.PICC_IsNewCardPresent()) 
  {
    return;
  }
  // Select one of the cards
  if ( ! mfrc522.PICC_ReadCardSerial()) 
  {
    return;
  }
  //Show UID on serial monitor
  Serial.print("UID tag :");
  String content= "";
  byte letter;
  for (byte i = 0; i < mfrc522.uid.size; i++) 
  {
     Serial.print(mfrc522.uid.uidByte[i] < 0x10 ? " 0" : " ");
     Serial.print(mfrc522.uid.uidByte[i], HEX);
     content.concat(String(mfrc522.uid.uidByte[i] < 0x10 ? " 0" : " "));
     content.concat(String(mfrc522.uid.uidByte[i], HEX));
  }
  Serial.println();
  Serial.print("Message : ");
  content.toUpperCase();
  //change here the UID of the card/cards that you want to give access
  if (content.substring(1) == "04 34 9E 02 70 71 80n n    bee" || content.substring(1) == "50 0E 5A 1B" || content.substring(1) == "13 CB 63 A7") 
  {
    Serial.println("Authorized access");
    Serial.println();
    delay(500); //wait for 500ms
    digitalWrite(LED_G, HIGH);//turn on the green led
    tone(BUZZER, 500);//turn on the buzzer on 500 frequency
    delay(300); // wait for 300ms
    noTone(BUZZER); // turn off buzzer
    myServo.write(180); //servo turn 180 degree
    delay(5000); // wait 5000ms
    myServo.write(0); // turn off servo
    digitalWrite(LED_G, LOW); // turn off green lite 
  }
 
 else   {
    Serial.println(" Access denied");
    digitalWrite(LED_R, HIGH); // red led turn on 
    tone(BUZZER, 300);//turn on the buzzer on 300 frequency
    delay(1000);//wait 1000ms
    digitalWrite(LED_R, LOW);//turn off red lite 
    noTone(BUZZER);// turn off buzzer
  }
} 
