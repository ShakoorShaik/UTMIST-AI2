from environment.environment import RenderMode, CameraResolution
from environment.agent import run_real_time_match
from user.train_agent import UserInputAgent, BasedAgent, ConstantAgent, ClockworkAgent, SB3Agent, RecurrentPPOAgent #add anymore custom Agents (from train_agent.py) here as needed
from user.my_agent import SubmittedAgent
import os
from pathlib import Path

# Create a fake runtime directory so pygame/SDL have somewhere to write
os.environ.setdefault("XDG_RUNTIME_DIR", f"/tmp/{os.getuid()}-runtime")
Path(os.environ["XDG_RUNTIME_DIR"]).mkdir(parents=True, exist_ok=True)

# Tell pygame to not use real hardware
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")  # avoids X server requirement
# ---------------------------------
import pygame
pygame.init()

my_agent = UserInputAgent()

#Input your file path here in SubmittedAgent if you are loading a model:
opponent = SubmittedAgent(file_path=None)

match_time = 99999

# Run a single real-time match
run_real_time_match(
    agent_1=my_agent,
    agent_2=opponent,
    max_timesteps=30 * 999990000,  # Match time in frames (adjust as needed)
    resolution=CameraResolution.LOW,
)