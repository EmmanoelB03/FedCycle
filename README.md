```markdown
# 🔄 FedCycle: Federated Learning & TinyML on Constrained Edge Devices

**FedCycle** is a complete, end-to-end architecture demonstrating the application of **Federated Learning (FL)** and **Tiny Machine Learning (TinyML)** directly on resource-constrained embedded systems (ESP32). 

Unlike traditional approaches that rely on heavy frameworks like TensorFlow Lite, FedCycle implements a custom Neural Network inference and Backpropagation engine in **pure C++**, combined with a Python-based Global Server for FedAvg aggregation.

---

## 📑 Table of Contents

- [About the Project](#-about-the-project)
- [Key Architectural Innovations](#-key-architectural-innovations)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Folder Structure](#-folder-structure)
- [How to Run](#-how-to-run)
- [Author & Contact](#-author--contact)

---

## 📖 About the Project

This repository serves as a foundation for research in Distributed AI. It proves that microcontrollers with severe memory constraints can not only run inferences but also **train models locally** using dynamic memory allocation and communicate their learnings to a global aggregator to build collective intelligence.

### The Machine Learning Model
The system uses a 2-layer Neural Network (Hidden Layer: 2048 -> 128, Output Layer: 128 -> 10) originally extracted from PyTorch. The embedded device performs the forward pass (ReLU, Softmax), computes the Cross-Entropy Loss, and updates weights via Backpropagation.

---

## 🚀 Key Architectural Innovations

This project overcomes several critical hardware limitations commonly found in embedded AI:

1. **PSRAM Dynamic Allocation:** Overcomes the ESP32's SRAM limits by dynamically allocating large weight matrices into the external PSRAM (`ps_malloc` / `free`), freeing up memory for heavy network stacks (TCP/IP/WiFi).
2. **DROM Cache Window Bypass (Xorshift):** Simulators like Wokwi suffer from Boot Loops (DROM Cache Overflow) when mapping 1MB of static weights alongside Wi-Fi drivers. FedCycle bypasses this by generating the large hidden layer ($W_1$) dynamically at runtime using a deterministic **Xorshift** generator.
3. **Extreme Transfer Learning (Layer Freezing):** To prevent "Catastrophic Forgetting" and amnesia across federated rounds, the heavy hidden layer is frozen as a static feature extractor. The device only computes gradients and updates the final decision layer ($W_2$), drastically reducing computational load.
4. **Bidirectional FL Synchronization:** Implements a full FL cycle:
   - **Downlink (HTTP GET):** Downloads the latest Global Model before training.
   - **Uplink (HTTP POST):** Uploads the locally updated weights along with custom HTTP headers (`X-Loss`, `X-Accuracy`) for server-side metric tracking.

---

## ⚙️ System Architecture

* **The Edge Node (ESP32 / C++):** Runs the training loops, handles memory cleanup, and manages Wi-Fi connections gracefully to avoid cache collisions.
* **The Global Server (Python / Flask):** Acts as the central school. It waits for updates from multiple clients, applies the **Federated Averaging (FedAvg)** algorithm using NumPy, and automatically plots performance metrics (Loss/Accuracy) using Matplotlib.

---

## 🛠️ Tech Stack

* **Embedded:** C++, ESP-IDF / Arduino Core, PlatformIO
* **Simulation:** Wokwi (ESP32-WROVER with PSRAM enabled)
* **Server:** Python 3, Flask, NumPy, Matplotlib
* **Deep Learning Concept:** PyTorch (for initial weight extraction)

---

## 📁 Folder Structure

```text
FedCycle/
├── src/
│   └── main.cpp                  # Edge Node firmware (TinyML + Backprop + HTTP)
├── pesos_classificacao.h         # Extracted initial weights (W2 and b2)
├── server.py                     # Global Aggregator (FedAvg + Plotting)
├── platformio.ini                # Environment configuration (PSRAM enabled)
└── README.md
```

---

## 🚦 How to Run

### 1. Start the Global Server
Ensure you have Python installed along with the required libraries:
```bash
pip install flask numpy matplotlib
```
Run the server in your terminal:
```bash
python3 server.py
```
*The server will start on port 5000 and wait for client connections.*

### 2. Run the Edge Node
1. Update the `serverName` IP address in `src/main.cpp` to match your local machine's IP where the Python server is running.
2. Build the project using **PlatformIO** (`Build`).
3. Start the simulation in **Wokwi** (or flash to a physical ESP32-WROVER).
4. **Simulate Multiple Clients:** Stop and Start the Wokwi simulation multiple times. After 3 runs, the Python server will execute the FedAvg algorithm, save the new global model, and generate a `.png` chart with the network's evolution.

---

## 🤝 Contributing

Contributions to expand this to multi-node physical networks, BLE communication, or quantization are welcome! Open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Author & Contact

**Emmanoel B.**
- **Email:** emmanoel.barbosa03@gmail.com
- **GitHub:** [EmmanoelB03](https://github.com/EmmanoelB03)
```

