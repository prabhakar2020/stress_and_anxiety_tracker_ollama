# stress_and_axiety_tracker_ollama
AI powered health monitoring system that analyzes heartbeat data to detect stress levels, calculate heart rate and HRV metrics, and generate intelligent health insights using local AI models using OLLAM-qwen2.5:7b, biometric telemetry analysis, and physiological signal processing.
This project runs 100% locally on your local computer using an open-source AI model (Qwen2.5) and Python.

### Here are the steps to get everything running smoothly on your local machine
1. Update WSL Ubuntu and Install Python Pip
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv -y
2. Create a Dedicated Project Directory & Virtual Environment
mkdir /home/prabhakar/linux_data/jupyter_project
cd /home/prabhakar/linux_data/jupyter_project
3. Create a virtual environment (named .venv):
python3 -m venv .venv
4. Activate the virtual environment
source .venv/bin/activate
5. Install Dependencies
pip install pandas numpy neurokit2 langchain langchain-community
6. Install Ollama inside WSL
curl -fsSL https://ollama.com/install.sh | sh
7. Start the Ollama service background daemon
ollama serve
Note: If you encounter any errors then please reverify Ollama is Alive or not using "curl http://127.0.0.1:11434/"
8. Pull/ Download the Llama 3 model to make sure it's downloaded on your local machine. It will download llama3 module approax ~4.8GB
ollama run qwen2.5:7b

<img width="694" height="426" alt="image" src="https://github.com/user-attachments/assets/c9617d97-ca79-4e70-a0ed-07b7a25e0422" />
<img width="692" height="482" alt="image" src="https://github.com/user-attachments/assets/3af0efa2-e683-4ef3-8187-92b257218823" />

