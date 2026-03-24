/*
 * MONA standalone wheel-speed control example (no Wi-Fi).
 *
 * What this sketch shows:
 * - How to run the same wheel-speed controller used by network firmware.
 * - How to set desired left/right wheel speeds from user code.
 * - How to test quickly over Serial without laptop_server.py.
 *
 * Serial commands at 115200 baud:
 *   set <left_rad_s> <right_rad_s>
 *   stop
 *   help
 */

#include <Mona_ESP_lib.h>
#include <math.h>

#ifndef TWO_PI
#define TWO_PI 6.28318530717958647693
#endif

const bool DEBUG_SERIAL = true;

const uint32_t CONTROL_PERIOD_MS = 80;
const uint32_t CONTROL_PERIOD_US = CONTROL_PERIOD_MS * 1000U;
const float DT = CONTROL_PERIOD_MS / 1000.0f;
const float TICKS_PER_REV = 500.0f;

// One-pole low-pass filter on measured wheel speed before control.
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
float eps_smc = 0.2785f / eps_UUB;

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

unsigned long lastPrintMs = 0;
const unsigned long STATUS_PERIOD_MS = 500;

portMUX_TYPE muxLeftEnc = portMUX_INITIALIZER_UNLOCKED;
portMUX_TYPE muxRightEnc = portMUX_INITIALIZER_UNLOCKED;
portMUX_TYPE muxCmd = portMUX_INITIALIZER_UNLOCKED;
portMUX_TYPE muxLeftState = portMUX_INITIALIZER_UNLOCKED;
portMUX_TYPE muxRightState = portMUX_INITIALIZER_UNLOCKED;

hw_timer_t* timerControl = NULL;
String rxLine;

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

// Public helper: call this from your own code to command wheel speeds.
void setDesiredWheelSpeeds(float left_rad_s, float right_rad_s)
{
    portENTER_CRITICAL(&muxCmd);
    desired_V_L = left_rad_s;
    desired_V_R = right_rad_s;
    portEXIT_CRITICAL(&muxCmd);
}

void stopRobot()
{
    setDesiredWheelSpeeds(0.0f, 0.0f);
}

void IRAM_ATTR controlISR()
{
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

void setupControlTimer()
{
    // 80 prescaler => 1 MHz timer tick (1 us).
    timerControl = timerBegin(0, 80, true);
    timerAttachInterrupt(timerControl, &controlISR, true);
    timerAlarmWrite(timerControl, CONTROL_PERIOD_US, true);
    timerAlarmEnable(timerControl);
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

void printHelp()
{
    Serial.println("Commands:");
    Serial.println("  set <left_rad_s> <right_rad_s>");
    Serial.println("  stop");
    Serial.println("  help");
    Serial.println("Example: set 8.0 8.0");
}

void handleSerialLine(const String& line)
{
    String t = line;
    t.trim();
    if (t.length() == 0)
    {
        return;
    }

    float leftCmd = 0.0f;
    float rightCmd = 0.0f;
    if (sscanf(t.c_str(), "set %f %f", &leftCmd, &rightCmd) == 2)
    {
        setDesiredWheelSpeeds(leftCmd, rightCmd);
        Serial.print("[cmd] target set: left=");
        Serial.print(leftCmd, 3);
        Serial.print(" rad/s right=");
        Serial.print(rightCmd, 3);
        Serial.println(" rad/s");
        return;
    }

    if (t.equalsIgnoreCase("stop"))
    {
        stopRobot();
        Serial.println("[cmd] stop");
        return;
    }

    if (t.equalsIgnoreCase("help"))
    {
        printHelp();
        return;
    }

    Serial.println("[cmd] unknown command");
    printHelp();
}

void readSerialNonBlocking()
{
    while (Serial.available() > 0)
    {
        char c = (char)Serial.read();
        if (c == '\n')
        {
            handleSerialLine(rxLine);
            rxLine = "";
        }
        else if (c != '\r')
        {
            rxLine += c;
        }
    }
}

void printStatus()
{
    if (!DEBUG_SERIAL)
    {
        return;
    }

    const unsigned long now = millis();
    if (now - lastPrintMs < STATUS_PERIOD_MS)
    {
        return;
    }
    lastPrintMs = now;

    float targetL = 0.0f;
    float targetR = 0.0f;
    float measuredL = 0.0f;
    float measuredR = 0.0f;
    int pwmL = 0;
    int pwmR = 0;

    portENTER_CRITICAL(&muxCmd);
    targetL = desired_V_L;
    targetR = desired_V_R;
    portEXIT_CRITICAL(&muxCmd);

    portENTER_CRITICAL(&muxLeftState);
    measuredL = speed_filt_L;
    pwmL = PWM_left;
    portEXIT_CRITICAL(&muxLeftState);

    portENTER_CRITICAL(&muxRightState);
    measuredR = speed_filt_R;
    pwmR = PWM_right;
    portEXIT_CRITICAL(&muxRightState);

    Serial.print("[state] targetL =");
    Serial.print(targetL, 3);
    Serial.print(" targetR =");
    Serial.print(targetR, 3);
    Serial.print(" measuredL =");
    Serial.print(measuredL, 3);
    Serial.print(" measuredR =");
    Serial.print(measuredR, 3);
    Serial.print(" pwmL =");
    Serial.print(pwmL);
    Serial.print(" pwmR =");
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
    resetControllerStates();
    setupControlTimer();

    stopRobot();
    printHelp();
}

void loop()
{
    applyMotorOutputs();
    readSerialNonBlocking();
    printStatus();
}
