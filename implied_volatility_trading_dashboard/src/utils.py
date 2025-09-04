import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class DisplayUtils:
    @staticmethod
    def create_root():
        root = tk.Tk()
        return root