TaxiRL: Procedural Content Generation using Reinforcement Learning
A web-based interactive simulator that demonstrates Reinforcement Learning concepts using a custom 10×10 Taxi grid environment, powered by PPO and procedural level generation.

Banner License Python RL

TaxiRL is an educational and interactive reinforcement learning simulator where an AI agent learns to pick up and drop off passengers in a 10x10 procedurally generated grid world. Trained using Proximal Policy Optimization (PPO) from Stable-Baselines3, it allows both human and agent play modes with visual feedback and reward tracking in real-time.

This project blends AI, Flask web development, and procedural content generation for an intuitive learning experience.

 Features
- Custom 10x10 Grid Environment with level loading from JSON
- PPO Agent trained using Stable-Baselines3
- Human vs Agent Play in the same game interface
- Procedural Content Generation for dynamic levels
- Real-time Reward Tracking and agent performance visualization
- Flask-based Web Interface with image-based grid rendering
- Modular Codebase: Easily adaptable and extendable
- Tech Stack
Component	Tools/Technologies
Language	Python 3.8+
RL Framework	Stable-Baselines3
Environment	OpenAI Gym + custom PCG
Web Framework	Flask
Visualization	Matplotlib, Base64 Encoded Images
Deployment Ready	Flask Web App (Localhost, easy to port to cloud)
🏁 Getting Started
1. Clone the Repository
git clone https://github.com/ikcod/gym_pcgrl.git
cd gym_pcgrl
2. Create a Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
4. Train the PPO Agent (Optional)
If you'd like to retrain the agent:

python train.py
Trained models are stored in /taxi_logs/final_model.zip.

5. Run the Flask Web App
python app.py
Navigate to http://127.0.0.1:5000 in your browser.

 Project Structure
.
├── app.py                  # Flask web server
├── train.py                # PPO training script
├── pcgrl_env.py            # Custom Taxi-v3 Gym environment
├── static/                 # UI assets (icons, backgrounds)
├── templates/              # HTML templates
├── game_levels.json        # Custom level definitions
├── taxi_logs/              # Trained model files
└── requirements.txt
🧪 Results
PPO Agent converges after ~50k timesteps
Agent consistently outperforms human gameplay in medium/hard levels
Intuitive GUI provides easy comparison between agent and player behavior

References
PCGRL Paper – A. Khalifa et al.
