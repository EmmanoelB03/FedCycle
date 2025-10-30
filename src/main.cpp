#include <WiFi.h>
#include <HTTPClient.h>

void setup() {
  Serial.begin(115200);

  WiFi.begin("Wokwi-GUEST", "", 6);
  while (WiFi.status() != WL_CONNECTED) {
    delay(100);
  }

  HTTPClient http;
  http.begin("http://host.wokwi.internal:3000/");
  int httpCode = http.GET();

  if (httpCode > 0) {
    String payload = http.getString();
    Serial.println("Payload da API:");
    Serial.println(payload);
  } else {
    Serial.println("Erro ao chamar a API");
  }
  http.end();
}
void loop() {}
