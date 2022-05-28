
int redPin = 8;
int greenPin = 9;
int bluePin = 10;


void setup() {
pinMode(redPin,OUTPUT);//Defining the pin number , Defining it's OUTPUT or INPUT
pinMode(greenPin,OUTPUT);//Defining the pin number , Defining it's OUTPUT or INPUT
pinMode(bluePin,OUTPUT);//Defining the pin number , Defining it's OUTPUT or INPUT

}

void loop() {



setColor(255,0,0); //RED COLOR
delay(500);

setColor(0,250,0); //GREEN COLOR
delay(500);

setColor(0,0,255); //BLUE COLOR
delay(500);

setColor(255,255,255); //WHITE COLOR
delay(500);

setColor(170,05,255); //WHITE COLOR
delay(500);

+
}

void setColor(int redValue, int greenValue, int blueValue){
  analogWrite(redPin,redValue);
  analogWrite(greenPin,greenValue);
  analogWrite(bluePin,blueValue);
}

     
