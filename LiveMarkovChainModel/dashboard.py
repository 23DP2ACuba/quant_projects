import tkinter as tk
from tkinter import ttk, messagebox
import time
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle  
import matplotlib.dates as mdates   
from datetime import datetime as datetime
from ibapp import IBApp, OHLCBar
from markov_model import MarkovRegime

class Dashboard:
    def __init__(self):
        pass
    