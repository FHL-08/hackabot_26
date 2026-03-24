/*
 * MONA velocity TCP client for one robot (BOT 1/2/3).
 *
 * Set BOT_NUM and BOT_ID_STR, then flash to each robot.
 *
 * Accepted command formats:
 *   1) six-float broadcast:  b1_L b1_R b2_L b2_R b3_L b3_R
 *   2) bot(id,l,r)
 *   3) id l r
 */

#include <Mona_ESP_lib.h>
#include <WiFi.h>
#include <math.h>

#ifndef TWO_PI
#define TWO_PI 6.28318530717958647693
#endif

static const int BOT_NUM = 1;
static const char* BOT_ID_STR = "bot1";

const bool DEBUG_SERIAL = false;

const char* WIFI_SSID = "TP-Link_6C24";
const char* WIFI_PASSWORD = "17346559";
const bool USE_STATIC_IP = false;
IPAddress BOT_IP(192, 168, 0, 241);
IPAddress GATEWAY(192, 168, 0, 1);
IPAddress SUBNET(255, 255, 255, 0);
IPAddress DNS1(8, 8, 8, 8);
IPAddress DNS2(1, 1, 1, 1);
const char* LAPTOP_HOST = "192.168.0.122";
const uint16_t LAPTOP_PORT = 5005;

const unsigned long WIFI_TIMEOUT_MS = 20000;
const unsigned long RETRY_MS = 1000;

const unsigned long RATE_CHECK_MS = 2000;

const uint32_t CONTROL_PERIOD_MS = 80;
const uint32_t CONTROL_PERIOD_US = CONTROL_PERIOD_MS * 1000U;
const float DT = CONTROL_PERIOD_MS / 1000.0f;
const float TICKS_PER_REV = 500.0f;

// One-pole low-pass filter on wheel speed before control.
const float SPEED_LPF_CUTOFF_HZ = 2.0f;
float speed_lpf_alpha = 0.0f;

// Left wheel model/controller.
float a_L = -9.303261f;
float b_L = 1.04f;
float K_lqr_L = 12.9639f;
float N_ff_L = 8.945f;

// Right wheel model/controller.
float a_R = -7.259638f;
float b_R = 0.8254f;
float K_lqr_R = 13.0532f;
float N_ff_R = 8.795f;

float L_dob = 5.0f;
float eta_smc = 1.5f;
float eps_UUB = 0.01f;
float eps_smc = 0.2785 / eps_UUB;

volatile long enc_left = 0;
volatile long enc_right = 0;

volatile float desired_V_L = 0.0f;
volatile float desired_V_R = 0.0f;

volatile float speed_raw_L = 0.0f;
volatile float speed_raw_R = 0.0f;
volatile float speed_filt_L = 0.0f;
volatile float speed_filt_R = 0.0f;

volatile float p_L = 0.0f;
volatile float vm_L = 0.0f;
volatile float p_R = 0.0f;
volatile float vm_R = 0.0f;

volatile int PWM_left = 0;
volatile int PWM_right = 0;

volatile unsigned long tcpMsgCount = 0;
volatile unsigned long controlTickCount = 0;
unsigned long lastRateCheck = 0;

portMUX_TYPE muxLeftEnc = portMUX_INITIALIZER_UNLOCKED;
portMUX_TYPE muxRightEnc = portMUX_INITIALIZER_UNLOCKED;
portMUX_TYPE muxCmd = portMUX_INITIALIZER_UNLOCKED;
portMUX_TYPE muxLeftState = portMUX_INITIALIZER_UNLOCKED;
portMUX_TYPE muxRightState = portMUX_INITIALIZER_UNLOCKED;

hw_timer_t* timerControl = NULL;

WiFiClient tcp;
String rxLine;

void dbg(const char* msg)
{
    if (DEBUG_SERIAL)
    {
        Serial.println(msg);
    }
}

void IRAM_ATTR leftEncoderISR()
{
    portENTER_CRITICAL_ISR(&muxLeftEnc);
    if (digitalRead(Mot_left_feedback_2))
    {
        enc_left++;
    }
    else
    {
        enc_left--;
    }
    portEXIT_CRITICAL_ISR(&muxLeftEnc);
}

void IRAM_ATTR rightEncoderISR()
{
    portENTER_CRITICAL_ISR(&muxRightEnc);
    if (digitalRead(Mot_right_feedback_2))
    {
        enc_right++;
    }
    else
    {
        enc_right--;
    }
    portEXIT_CRITICAL_ISR(&muxRightEnc);
}

void setupEncoders()
{
    pinMode(Mot_left_feedback, INPUT);
    pinMode(Mot_left_feedback_2, INPUT);
    pinMode(Mot_right_feedback, INPUT);
    pinMode(Mot_right_feedback_2, INPUT);
    attachInterrupt(digitalPinToInterrupt(Mot_left_feedback), leftEncoderISR, CHANGE);
    attachInterrupt(digitalPinToInterrupt(Mot_right_feedback), rightEncoderISR, CHANGE);
}

void monaRemapMotorPwmAfterWiFi()
{
    pinMode(Mot_right_forward, OUTPUT);
    pinMode(Mot_right_backward, OUTPUT);
    pinMode(Mot_left_forward, OUTPUT);
    pinMode(Mot_left_backward, OUTPUT);
    analogWriteResolution(Mot_right_forward, Mot_res);
    analogWriteFrequency(Mot_right_forward, Mot_freq);
    analogWriteResolution(Mot_right_backward, Mot_res);
    analogWriteFrequency(Mot_right_backward, Mot_freq);
    analogWriteResolution(Mot_left_forward, Mot_res);
    analogWriteFrequency(Mot_left_forward, Mot_freq);
    analogWriteResolution(Mot_left_backward, Mot_res);
    analogWriteFrequency(Mot_left_backward, Mot_freq);
    Motors_stop();
    dbg("[motors] PWM remapped after WiFi");
}

void resetControllerStates()
{
    portENTER_CRITICAL(&muxLeftState);
    speed_raw_L = 0.0f;
    speed_filt_L = 0.0f;
    p_L = 0.0f;
    vm_L = 0.0f;
    PWM_left = 0;
    portEXIT_CRITICAL(&muxLeftState);

    portENTER_CRITICAL(&muxRightState);
    speed_raw_R = 0.0f;
    speed_filt_R = 0.0f;
    p_R = 0.0f;
    vm_R = 0.0f;
    PWM_right = 0;
    portEXIT_CRITICAL(&muxRightState);
}

void IRAM_ATTR controlISR()
{
    controlTickCount++;

    long ticksL = 0;
    long ticksR = 0;
    portENTER_CRITICAL_ISR(&muxLeftEnc);
    ticksL = enc_left;
    enc_left = 0;
    portEXIT_CRITICAL_ISR(&muxLeftEnc);

    portENTER_CRITICAL_ISR(&muxRightEnc);
    ticksR = enc_right;
    enc_right = 0;
    portEXIT_CRITICAL_ISR(&muxRightEnc);

    const float rawL = ((float)ticksL / TICKS_PER_REV) * (float)TWO_PI / DT;
    const float rawR = ((float)ticksR / TICKS_PER_REV) * (float)TWO_PI / DT;

    float filtL = 0.0f;
    float pL = 0.0f;
    float vmL = 0.0f;
    portENTER_CRITICAL_ISR(&muxLeftState);
    filtL = speed_lpf_alpha * speed_filt_L + (1.0f - speed_lpf_alpha) * rawL;
    speed_raw_L = rawL;
    speed_filt_L = filtL;
    pL = p_L;
    vmL = vm_L;
    portEXIT_CRITICAL_ISR(&muxLeftState);

    float filtR = 0.0f;
    float pR = 0.0f;
    float vmR = 0.0f;
    portENTER_CRITICAL_ISR(&muxRightState);
    filtR = speed_lpf_alpha * speed_filt_R + (1.0f - speed_lpf_alpha) * rawR;
    speed_raw_R = rawR;
    speed_filt_R = filtR;
    pR = p_R;
    vmR = vm_R;
    portEXIT_CRITICAL_ISR(&muxRightState);

    float targetL = 0.0f;
    float targetR = 0.0f;
    portENTER_CRITICAL_ISR(&muxCmd);
    targetL = desired_V_L;
    targetR = desired_V_R;
    portEXIT_CRITICAL_ISR(&muxCmd);

    const float u_nom_L = -K_lqr_L * (filtL - targetL) + N_ff_L * targetL;
    const float d_hat_L = pL + L_dob * filtL;
    const float s_L = filtL - vmL;
    const float smc_term_L = a_L * s_L + eta_smc * tanhf(s_L / eps_smc);
    const float u_raw_L = u_nom_L - (d_hat_L / b_L) - (smc_term_L / b_L);
    const int pwmL = (int)constrain((int)roundf(u_raw_L), -255, 255);

    pL = pL + DT * (-L_dob * pL - L_dob * (a_L + L_dob) * filtL - L_dob * b_L * (float)pwmL);
    vmL = vmL + DT * (a_L * vmL + b_L * u_nom_L);

    const float u_nom_R = -K_lqr_R * (filtR - targetR) + N_ff_R * targetR;
    const float d_hat_R = pR + L_dob * filtR;
    const float s_R = filtR - vmR;
    const float smc_term_R = a_R * s_R + eta_smc * tanhf(s_R / eps_smc);
    const float u_raw_R = u_nom_R - (d_hat_R / b_R) - (smc_term_R / b_R);
    const int pwmR = (int)constrain((int)roundf(u_raw_R), -255, 255);

    pR = pR + DT * (-L_dob * pR - L_dob * (a_R + L_dob) * filtR - L_dob * b_R * (float)pwmR);
    vmR = vmR + DT * (a_R * vmR + b_R * u_nom_R);

    portENTER_CRITICAL_ISR(&muxLeftState);
    p_L = pL;
    vm_L = vmL;
    PWM_left = pwmL;
    portEXIT_CRITICAL_ISR(&muxLeftState);

    portENTER_CRITICAL_ISR(&muxRightState);
    p_R = pR;
    vm_R = vmR;
    PWM_right = pwmR;
    portEXIT_CRITICAL_ISR(&muxRightState);
}

void setupControlTimers()
{
    // 80 prescaler => 1 MHz timer tick (1 us).
    timerControl = timerBegin(0, 80, true);
    timerAttachInterrupt(timerControl, &controlISR, true);
    timerAlarmWrite(timerControl, CONTROL_PERIOD_US, true);
    timerAlarmEnable(timerControl);
}

void stopControlTimers()
{
    if (timerControl != NULL)
    {
        timerAlarmDisable(timerControl);
    }
}

void startControlTimers()
{
    if (timerControl != NULL)
    {
        timerAlarmEnable(timerControl);
    }
}

void applyMotorOutputs()
{
    int pwmL = 0;
    int pwmR = 0;

    portENTER_CRITICAL(&muxLeftState);
    pwmL = PWM_left;
    portEXIT_CRITICAL(&muxLeftState);

    portENTER_CRITICAL(&muxRightState);
    pwmR = PWM_right;
    portEXIT_CRITICAL(&muxRightState);

    if (pwmL >= 0)
    {
        Left_mot_backward(pwmL);
    }
    else
    {
        Left_mot_forward(abs(pwmL));
    }

    if (pwmR >= 0)
    {
        Right_mot_backward(pwmR);
    }
    else
    {
        Right_mot_forward(abs(pwmR));
    }
}

void sendTcpLine(const String& s, bool logSerial)
{
    if (DEBUG_SERIAL && logSerial)
    {
        Serial.print("[tcp>>] ");
        Serial.println(s);
    }
    if (!tcp.connected())
    {
        return;
    }
    tcp.print(s);
    tcp.print('\n');
}

void sendStateBack()
{
    if (!DEBUG_SERIAL)
    {
        return;
    }

    float wL = 0.0f;
    float wR = 0.0f;
    int pwmL = 0;
    int pwmR = 0;

    portENTER_CRITICAL(&muxLeftState);
    wL = speed_filt_L;
    pwmL = PWM_left;
    portEXIT_CRITICAL(&muxLeftState);

    portENTER_CRITICAL(&muxRightState);
    wR = speed_filt_R;
    pwmR = PWM_right;
    portEXIT_CRITICAL(&muxRightState);

    char buf[140];
    snprintf(buf, sizeof(buf), "%s state: omegaL=%.4f omegaR=%.4f pwmL=%d pwmR=%d", BOT_ID_STR, wL, wR, pwmL, pwmR);
    sendTcpLine(String(buf), true);
}

void handleTcpLine(const String& line)
{
    String t = line;
    t.trim();
    if (t.length() == 0)
    {
        return;
    }

    tcpMsgCount++;

    float v[6];
    const int n6 = sscanf(t.c_str(), "%f %f %f %f %f %f", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]);
    if (n6 == 6)
    {
        if (BOT_NUM >= 1 && BOT_NUM <= 3)
        {
            const int slot = (BOT_NUM - 1) * 2;
            portENTER_CRITICAL(&muxCmd);
            desired_V_L = v[slot];
            desired_V_R = v[slot + 1];
            portEXIT_CRITICAL(&muxCmd);
            sendStateBack();
        }
        return;
    }

    int id = -1;
    float l = 0.0f;
    float r = 0.0f;
    if (sscanf(t.c_str(), "bot(%d,%f,%f)", &id, &l, &r) == 3 || sscanf(t.c_str(), "%d %f %f", &id, &l, &r) == 3)
    {
        if (id == BOT_NUM)
        {
            portENTER_CRITICAL(&muxCmd);
            desired_V_L = l;
            desired_V_R = r;
            portEXIT_CRITICAL(&muxCmd);
            sendStateBack();
        }
        return;
    }

    if (DEBUG_SERIAL)
    {
        Serial.println("[cmd] need 6 floats, or bot(id,l,r), or id l r");
    }
}

bool wifiConnectedOk()
{
    return WiFi.status() == WL_CONNECTED && WiFi.localIP() != IPAddress(0, 0, 0, 0);
}

void ensureWifi()
{
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    unsigned long t0 = millis();
    while (!wifiConnectedOk())
    {
        if (millis() - t0 > WIFI_TIMEOUT_MS)
        {
            delay(500);
            WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
            t0 = millis();
        }
        delay(50);
    }
}

void connectLaptop()
{
    while (!tcp.connect(LAPTOP_HOST, LAPTOP_PORT))
    {
        delay(RETRY_MS);
        if (!wifiConnectedOk())
        {
            return;
        }
    }
    rxLine = "";
    portENTER_CRITICAL(&muxCmd);
    desired_V_L = 0.0f;
    desired_V_R = 0.0f;
    portEXIT_CRITICAL(&muxCmd);

    if (DEBUG_SERIAL)
    {
        sendTcpLine(String(BOT_ID_STR) + String(" ready"), true);
        sendStateBack();
    }
}

void readTcpNonBlocking()
{
    while (tcp.available() > 0)
    {
        char c = (char)tcp.read();
        if (c == '\n')
        {
            handleTcpLine(rxLine);
            rxLine = "";
        }
        else if (c != '\r')
        {
            rxLine += c;
        }
    }
}

void printRates()
{
    if (!DEBUG_SERIAL)
    {
        return;
    }

    const unsigned long now = millis();
    if (now - lastRateCheck < RATE_CHECK_MS)
    {
        return;
    }

    const float elapsed = (now - lastRateCheck) / 1000.0f;

    const unsigned long tcpCount = tcpMsgCount;
    const unsigned long ctrlTicks = controlTickCount;

    tcpMsgCount = 0;
    controlTickCount = 0;
    lastRateCheck = now;

    float wL = 0.0f;
    float wR = 0.0f;
    int pwmL = 0;
    int pwmR = 0;

    portENTER_CRITICAL(&muxLeftState);
    wL = speed_filt_L;
    pwmL = PWM_left;
    portEXIT_CRITICAL(&muxLeftState);

    portENTER_CRITICAL(&muxRightState);
    wR = speed_filt_R;
    pwmR = PWM_right;
    portEXIT_CRITICAL(&muxRightState);

    Serial.print("[rate] TCP: ");
    Serial.print((float)tcpCount / elapsed, 1);
    Serial.print(" msg/s | Ctrl: ");
    Serial.print((float)ctrlTicks / elapsed, 1);
    Serial.print(" Hz | wL=");
    Serial.print(wL, 2);
    Serial.print(" wR=");
    Serial.print(wR, 2);
    Serial.print(" pwmL=");
    Serial.print(pwmL);
    Serial.print(" pwmR=");
    Serial.println(pwmR);
}

void setup()
{
    Mona_ESP_init();
    Serial.begin(115200);
    delay(100);

    const float tau = 1.0f / (TWO_PI * SPEED_LPF_CUTOFF_HZ);
    speed_lpf_alpha = tau / (tau + DT);

    setupEncoders();
    setupControlTimers();

    if (USE_STATIC_IP)
    {
        WiFi.config(BOT_IP, GATEWAY, SUBNET, DNS1, DNS2);
    }
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);

    ensureWifi();
    monaRemapMotorPwmAfterWiFi();
    connectLaptop();
}

void loop()
{
    if (!wifiConnectedOk())
    {
        stopControlTimers();
        tcp.stop();
        Motors_stop();

        ensureWifi();
        monaRemapMotorPwmAfterWiFi();
        resetControllerStates();

        startControlTimers();
        connectLaptop();
        return;
    }

    if (!tcp.connected())
    {
        stopControlTimers();
        tcp.stop();
        Motors_stop();

        resetControllerStates();
        startControlTimers();
        connectLaptop();
        return;
    }

    applyMotorOutputs();
    readTcpNonBlocking();
    printRates();
}
