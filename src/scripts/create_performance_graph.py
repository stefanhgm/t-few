import re
from collections import defaultdict
from glob import glob
import json
from math import sqrt

import matplotlib.pyplot as plt
import matplotlib

import numpy as np
import argparse

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


performance_per_dataset_setting = {
    'income': {
        'SAINT': [(0.74, 0.03), (0.65, 0.15), (0.79, 0.03), (0.81, 0.03), (0.84, 0.02), (0.84, 0.02), (0.87, 0.01), (0.88, 0.00), (0.91, 0.00)],
        'TabNet': [(0.56, 0.04), (0.59, 0.07), (0.62, 0.11), (0.64, 0.06), (0.71, 0.04), (0.73, 0.05), (0.80, 0.02), (0.83, 0.02), (0.92, 0.00)],
        'NODE': [(0.54, 0.02), (0.54, 0.04), (0.65, 0.04), (0.67, 0.03), (0.75, 0.02), (0.78, 0.01), (0.78, 0.01), (0.83, 0.01), (0.82, 0.00)],
        'Log. Reg.': [(0.68, 0.15), (0.72, 0.13), (0.80, 0.03), (0.82, 0.01), (0.83, 0.03), (0.85, 0.01), (0.87, 0.01), (0.88, 0.00), (0.90, 0.00)],
        'LightGBM': [(0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.78, 0.03), (0.81, 0.03), (0.87, 0.01), (0.88, 0.00), (0.93, 0.00)],
        'XGBoost': [(0.50, 0.00), (0.59, 0.06), (0.77, 0.02), (0.79, 0.03), (0.82, 0.02), (0.84, 0.01), (0.87, 0.01), (0.88, 0.00), (0.93, 0.00)],
        'TabPFN': [(0.73, 0.08), (0.71, 0.09), (0.76, 0.09), (0.80, 0.04), (0.82, 0.04), (0.84, 0.01), (0.86, 0.01), (0.87, 0.01), (0.89, 0.00)],
        'Log. Reg. ordinal': [(0.55, 0.04), (0.56, 0.06), (0.58, 0.07), (0.70, 0.06), (0.76, 0.03), (0.79, 0.01), (0.80, 0.01), (0.80, 0.00), (0.81, 0.00)],
        'LightGBM ordinal': [(0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.78, 0.01), (0.81, 0.01), (0.86, 0.01), (0.89, 0.00), (0.93, 0.00)],
        'XGBoost ordinal': [(0.50, 0.00), (0.63, 0.04), (0.74, 0.04), (0.76, 0.04), (0.79, 0.03), (0.84, 0.02), (0.86, 0.01), (0.88, 0.00), (0.93, 0.00)],
        'TabPFN ordinal': [(0.64, 0.11), (0.64, 0.06), (0.72, 0.04), (0.77, 0.02), (0.80, 0.02), (0.81, 0.01), (0.83, 0.01), (0.85, 0.01), (0.87, 0.00)],
        'TabLLM': [(0.84, 0.00), (0.84, 0.01), (0.84, 0.02), (0.84, 0.04), (0.84, 0.01), (0.84, 0.02), (0.86, 0.01), (0.87, 0.00), (0.89, 0.01)],
        'Text GPT-3': [(0.75, 0.01), (0.79, 0.03), (0.80, 0.03), (0.82, 0.02), (0.82, 0.01), (0.84, 0.02), (0.84, 0.02), (0.85, 0.01), (0.86, 0.00)],
        'Text T0': [(0.65, 0.01), (0.67, 0.03), (0.66, 0.07), (0.72, 0.02), (0.75, 0.03), (0.79, 0.04), (0.82, 0.02), (0.83, 0.02), (0.86, 0.01)],
        'Table-To-Text': [(0.50, 0.00), (0.64, 0.07), (0.64, 0.11), (0.72, 0.05), (0.74, 0.03), (0.79, 0.03), (0.81, 0.01), (0.84, 0.01), (0.84, 0.01)],
        'Text Template': [(0.84, 0.00), (0.84, 0.01), (0.84, 0.02), (0.84, 0.04), (0.84, 0.01), (0.84, 0.02), (0.86, 0.01), (0.87, 0.00), (0.89, 0.01)],
        'List Template': [(0.79, 0.01), (0.83, 0.01), (0.83, 0.03), (0.83, 0.02), (0.84, 0.01), (0.85, 0.01), (0.86, 0.01), (0.87, 0.01), (0.88, 0.01)],
        'List Only Values': [(0.73, 0.01), (0.74, 0.04), (0.75, 0.04), (0.80, 0.03), (0.82, 0.01), (0.84, 0.01), (0.84, 0.01), (0.86, 0.01), (0.87, 0.01)],
        'List Perm. Names': [(0.65, 0.00), (0.75, 0.03), (0.74, 0.05), (0.82, 0.02), (0.83, 0.02), (0.84, 0.02), (0.86, 0.01), (0.86, 0.01), (0.88, 0.01)],
        'List Perm. Values': [(0.26, 0.00), (0.40, 0.04), (0.48, 0.10), (0.65, 0.06), (0.72, 0.03), (0.79, 0.03), (0.81, 0.02), (0.83, 0.01), (0.84, 0.01)],
        'TabLLM (T0 3B + Text)': [(0.76, 0.00), (0.77, 0.06), (0.80, 0.04), (0.83, 0.02), (0.83, 0.03), (0.85, 0.01), (0.86, 0.00), (0.86, 0.01), (0.88, 0.01)],
        'GPT-3': [(0.75, 0.01)],
    },
    'car': {
        'SAINT': [(0.56, 0.08), (0.64, 0.08), (0.76, 0.03), (0.85, 0.03), (0.92, 0.02), (0.96, 0.01), (0.98, 0.01), (0.99, 0.00), (1.00, 0.00)],
        'TabNet': [(0.50, 0.00), (0.54, 0.05), (0.64, 0.05), (0.66, 0.05), (0.73, 0.07), (0.81, 0.04), (0.93, 0.02), (0.98, 0.01), (1.00, 0.00)],
        'NODE': [(0.51, 0.10), (0.57, 0.06), (0.69, 0.02), (0.74, 0.03), (0.80, 0.02), (0.82, 0.01), (0.91, 0.01), (0.96, 0.01), (0.93, 0.01)],
        'Log. Reg.': [(0.61, 0.02), (0.65, 0.10), (0.74, 0.07), (0.83, 0.02), (0.93, 0.02), (0.96, 0.01), (0.97, 0.01), (0.98, 0.00), (0.98, 0.00)],
        'LightGBM': [(0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.85, 0.06), (0.93, 0.01), (0.98, 0.01), (0.99, 0.01), (1.00, 0.00)],
        'XGBoost': [(0.50, 0.00), (0.59, 0.04), (0.70, 0.08), (0.82, 0.03), (0.91, 0.02), (0.95, 0.01), (0.98, 0.01), (0.99, 0.01), (1.00, 0.00)],
        'TabPFN': [(0.64, 0.06), (0.75, 0.05), (0.87, 0.04), (0.92, 0.02), (0.97, 0.00), (0.99, 0.01), (1.00, 0.00), (1.00, 0.00), (1.00, 0.00)],
        'Log. Reg. ordinal': [(0.62, 0.06), (0.63, 0.05), (0.64, 0.07), (0.75, 0.04), (0.73, 0.03), (0.73, 0.03), (0.74, 0.03), (0.76, 0.02), (0.78, 0.03)],
        'LightGBM ordinal': [(0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.75, 0.04), (0.91, 0.05), (0.98, 0.01), (0.99, 0.00), (1.00, 0.00)],
        'XGBoost ordinal': [(0.50, 0.00), (0.55, 0.03), (0.70, 0.04), (0.78, 0.03), (0.90, 0.03), (0.94, 0.01), (0.98, 0.01), (0.99, 0.01), (1.00, 0.00)],
        'TabPFN ordinal': [(0.59, 0.06), (0.65, 0.08), (0.75, 0.04), (0.82, 0.06), (0.89, 0.01), (0.93, 0.01), (0.98, 0.01), (0.99, 0.01), (1.00, 0.00)],
        'TabLLM': [(0.82, 0.02), (0.83, 0.03), (0.85, 0.03), (0.86, 0.03), (0.91, 0.02), (0.96, 0.02), (0.98, 0.01), (0.99, 0.00), (1.00, 0.00)],
        'Text GPT-3': [(0.72, 0.02), (0.75, 0.03), (0.75, 0.02), (0.78, 0.01), (0.83, 0.01), (0.87, 0.02), (0.90, 0.01), (0.93, 0.02), (0.93, 0.02)],
        'Text T0': [(0.85, 0.01), (0.85, 0.02), (0.84, 0.03), (0.86, 0.02), (0.89, 0.02), (0.92, 0.02), (0.94, 0.01), (0.98, 0.01), (0.99, 0.00)],
        'Table-To-Text': [(0.61, 0.01), (0.69, 0.04), (0.74, 0.04), (0.79, 0.02), (0.88, 0.01), (0.91, 0.02), (0.94, 0.01), (0.96, 0.01), (0.95, 0.01)],
        'Text Template': [(0.82, 0.02), (0.83, 0.03), (0.85, 0.03), (0.86, 0.03), (0.91, 0.02), (0.96, 0.02), (0.98, 0.01), (0.99, 0.00), (1.00, 0.00)],
        'List Template': [(0.79, 0.02), (0.84, 0.03), (0.85, 0.02), (0.86, 0.03), (0.91, 0.02), (0.95, 0.01), (0.98, 0.01), (0.99, 0.00), (1.00, 0.00)],
        'List Only Values': [(0.48, 0.03), (0.62, 0.04), (0.67, 0.03), (0.70, 0.03), (0.75, 0.02), (0.87, 0.02), (0.94, 0.01), (0.98, 0.01), (0.99, 0.01)],
        'List Perm. Names': [(0.39, 0.02), (0.54, 0.10), (0.58, 0.06), (0.70, 0.03), (0.86, 0.02), (0.94, 0.01), (0.97, 0.02), (0.99, 0.01), (0.99, 0.00)],
        'List Perm. Values': [(0.38, 0.02), (0.48, 0.08), (0.55, 0.05), (0.63, 0.04), (0.69, 0.03), (0.78, 0.02), (0.90, 0.03), (0.98, 0.01), (1.00, 0.00)],
        'TabLLM (T0 3B + Text)': [(0.78, 0.02), (0.80, 0.03), (0.84, 0.03), (0.84, 0.04), (0.89, 0.03), (0.91, 0.01), (0.96, 0.01), (0.98, 0.01), (0.99, 0.00)],
        'GPT-3': [(0.82, 0.01)],
    },
    'heart': {
        'SAINT': [(0.80, 0.12), (0.83, 0.10), (0.88, 0.07), (0.90, 0.01), (0.90, 0.04), (0.90, 0.02), (0.90, 0.01), (0.92, 0.01), (0.93, 0.01)],
        'TabNet': [(0.56, 0.12), (0.70, 0.05), (0.73, 0.14), (0.80, 0.04), (0.83, 0.05), (0.84, 0.03), (0.88, 0.02), (0.88, 0.03), (0.89, 0.03)],
        'NODE': [(0.52, 0.10), (0.78, 0.08), (0.83, 0.03), (0.86, 0.02), (0.88, 0.02), (0.88, 0.01), (0.91, 0.02), (0.92, 0.03), (0.92, 0.03)],
        'Log. Reg.': [(0.69, 0.17), (0.75, 0.13), (0.82, 0.06), (0.87, 0.05), (0.91, 0.01), (0.90, 0.02), (0.92, 0.01), (0.93, 0.01), (0.93, 0.01)],
        'LightGBM': [(0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.91, 0.01), (0.91, 0.01), (0.91, 0.01), (0.93, 0.00), (0.94, 0.01)],
        'XGBoost': [(0.50, 0.00), (0.55, 0.14), (0.84, 0.07), (0.88, 0.04), (0.91, 0.01), (0.91, 0.01), (0.90, 0.01), (0.92, 0.01), (0.94, 0.01)],
        'TabPFN': [(0.84, 0.06), (0.88, 0.05), (0.87, 0.06), (0.91, 0.02), (0.92, 0.02), (0.92, 0.02), (0.92, 0.01), (0.92, 0.02), (0.92, 0.02)],
        'Log. Reg. ordinal': [(0.60, 0.15), (0.68, 0.11), (0.73, 0.05), (0.76, 0.05), (0.80, 0.02), (0.81, 0.02), (0.83, 0.02), (0.83, 0.02), (0.83, 0.02)],
        'LightGBM ordinal': [(0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.91, 0.01), (0.91, 0.02), (0.91, 0.01), (0.92, 0.01), (0.94, 0.01)],
        'XGBoost ordinal': [(0.50, 0.00), (0.56, 0.15), (0.84, 0.07), (0.90, 0.03), (0.91, 0.01), (0.90, 0.01), (0.90, 0.01), (0.92, 0.01), (0.94, 0.01)],
        'TabPFN ordinal': [(0.79, 0.08), (0.85, 0.07), (0.88, 0.05), (0.90, 0.02), (0.92, 0.01), (0.92, 0.01), (0.92, 0.00), (0.92, 0.02), (0.92, 0.02)],
        'TabLLM': [(0.54, 0.04), (0.76, 0.14), (0.83, 0.05), (0.87, 0.04), (0.87, 0.06), (0.91, 0.01), (0.90, 0.01), (0.92, 0.01), (0.92, 0.01)],
        'Text Template': [(0.54, 0.04), (0.76, 0.14), (0.83, 0.05), (0.87, 0.04), (0.87, 0.06), (0.91, 0.01), (0.90, 0.01), (0.92, 0.01), (0.92, 0.01)],
        'List Template': [(0.52, 0.03), (0.73, 0.12), (0.83, 0.05), (0.87, 0.04), (0.88, 0.04), (0.91, 0.02), (0.91, 0.01), (0.92, 0.01), (0.92, 0.01)],
        'Text GPT-3': [(0.51, 0.04), (0.72, 0.05), (0.82, 0.03), (0.85, 0.05), (0.88, 0.03), (0.91, 0.02), (0.89, 0.02), (0.91, 0.01), (0.91, 0.01)],
        'Text T0': [(0.44, 0.03), (0.74, 0.07), (0.82, 0.10), (0.87, 0.02), (0.88, 0.02), (0.89, 0.04), (0.90, 0.01), (0.89, 0.02), (0.89, 0.03)],
        'Table-To-Text': [(0.56, 0.05), (0.73, 0.09), (0.78, 0.08), (0.86, 0.06), (0.88, 0.03), (0.91, 0.02), (0.91, 0.02), (0.90, 0.02), (0.91, 0.01)],
        'List Only Values': [(0.40, 0.04), (0.67, 0.16), (0.83, 0.06), (0.84, 0.05), (0.88, 0.03), (0.89, 0.03), (0.92, 0.02), (0.90, 0.00), (0.90, 0.01)],
        'List Perm. Names': [(0.57, 0.02), (0.78, 0.07), (0.85, 0.02), (0.82, 0.06), (0.87, 0.05), (0.90, 0.02), (0.92, 0.02), (0.91, 0.01), (0.91, 0.01)],
        'List Perm. Values': [(0.23, 0.02), (0.63, 0.20), (0.79, 0.12), (0.83, 0.07), (0.88, 0.04), (0.89, 0.04), (0.90, 0.02), (0.91, 0.01), (0.91, 0.01)],
        'TabLLM (T0 3B + Text)': [(0.56, 0.03), (0.68, 0.13), (0.82, 0.04), (0.85, 0.02), (0.86, 0.03), (0.90, 0.01), (0.91, 0.01), (0.93, 0.01), (0.93, 0.01), (0.94, 0.01)],
        'GPT-3': [(0.72, 0.01)],
    },
    'diabetes': {
        'SAINT': [(0.46, 0.12), (0.65, 0.11), (0.73, 0.06), (0.73, 0.06), (0.79, 0.03), (0.81, 0.03), (0.81, 0.04), (0.77, 0.03), (0.83, 0.03)],
        'TabNet': [(0.56, 0.04), (0.56, 0.06), (0.64, 0.09), (0.66, 0.06), (0.71, 0.04), (0.73, 0.04), (0.74, 0.05), (0.74, 0.07), (0.81, 0.03)],
        'NODE': [(0.49, 0.13), (0.67, 0.09), (0.69, 0.08), (0.73, 0.05), (0.77, 0.04), (0.80, 0.04), (0.81, 0.03), (0.83, 0.02), (0.83, 0.03)],
        'Log. Reg.': [(0.60, 0.15), (0.68, 0.11), (0.73, 0.05), (0.76, 0.05), (0.80, 0.02), (0.81, 0.02), (0.83, 0.02), (0.83, 0.02), (0.83, 0.02)],
        'LightGBM': [(0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.79, 0.02), (0.79, 0.04), (0.79, 0.02), (0.79, 0.03), (0.83, 0.03)],
        'XGBoost': [(0.50, 0.00), (0.59, 0.16), (0.72, 0.07), (0.69, 0.08), (0.73, 0.05), (0.78, 0.05), (0.80, 0.03), (0.80, 0.01), (0.84, 0.03)],
        'TabPFN': [(0.61, 0.13), (0.67, 0.11), (0.71, 0.07), (0.77, 0.03), (0.82, 0.03), (0.83, 0.03), (0.83, 0.03), (0.81, 0.02), (0.81, 0.03)],
        'Log. Reg. ordinal': [(0.60, 0.15), (0.68, 0.11), (0.73, 0.05), (0.76, 0.05), (0.80, 0.02), (0.81, 0.02), (0.83, 0.02), (0.83, 0.02), (0.83, 0.02)],
        'LightGBM ordinal': [(0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.79, 0.02), (0.79, 0.04), (0.79, 0.02), (0.79, 0.03), (0.83, 0.03)],
        'XGBoost ordinal': [(0.50, 0.00), (0.59, 0.16), (0.72, 0.07), (0.69, 0.08), (0.73, 0.05), (0.78, 0.05), (0.80, 0.03), (0.80, 0.01), (0.84, 0.03)],
        'TabPFN ordinal': [(0.61, 0.13), (0.67, 0.11), (0.71, 0.07), (0.77, 0.03), (0.82, 0.03), (0.83, 0.03), (0.83, 0.03), (0.81, 0.02), (0.81, 0.03)],
        'TabLLM': [(0.68, 0.06), (0.61, 0.09), (0.63, 0.08), (0.69, 0.07), (0.68, 0.04), (0.73, 0.03), (0.79, 0.04), (0.78, 0.02), (0.78, 0.04)],
        'Text Template': [(0.68, 0.06), (0.61, 0.09), (0.63, 0.08), (0.69, 0.07), (0.68, 0.04), (0.73, 0.03), (0.79, 0.04), (0.78, 0.02), (0.78, 0.04)],
        'List Template': [(0.64, 0.06), (0.64, 0.09), (0.64, 0.10), (0.67, 0.07), (0.70, 0.05), (0.76, 0.04), (0.78, 0.03), (0.78, 0.03), (0.78, 0.04)],
        'Text GPT-3': [(0.61, 0.06), (0.61, 0.07), (0.56, 0.12), (0.67, 0.08), (0.74, 0.04), (0.77, 0.02), (0.79, 0.03), (0.76, 0.03), (0.78, 0.04)],
        'Text T0': [(0.58, 0.04), (0.53, 0.05), (0.53, 0.06), (0.54, 0.09), (0.59, 0.05), (0.68, 0.02), (0.73, 0.04), (0.72, 0.05), (0.72, 0.03)],
        'Table-To-Text': [(0.58, 0.04), (0.51, 0.10), (0.53, 0.07), (0.56, 0.05), (0.57, 0.04), (0.59, 0.04), (0.72, 0.05), (0.74, 0.04), (0.75, 0.06)],
        'List Only Values': [(0.55, 0.05), (0.54, 0.07), (0.52, 0.05), (0.59, 0.08), (0.63, 0.04), (0.67, 0.07), (0.73, 0.03), (0.75, 0.06), (0.77, 0.04)],
        'List Perm. Names': [(0.56, 0.07), (0.60, 0.09), (0.68, 0.12), (0.74, 0.05), (0.74, 0.03), (0.72, 0.04), (0.76, 0.04), (0.77, 0.04), (0.77, 0.04)],
        'List Perm. Values': [(0.44, 0.03), (0.47, 0.09), (0.43, 0.06), (0.55, 0.07), (0.61, 0.05), (0.65, 0.05), (0.73, 0.03), (0.76, 0.03), (0.78, 0.02)],
        'TabLLM (T0 3B + Text)': [(0.62, 0.05), (0.57, 0.07), (0.60, 0.08), (0.67, 0.05), (0.67, 0.06), (0.76, 0.03), (0.77, 0.04), (0.81, 0.05), (0.80, 0.04), (0.82, 0.04)],
        'GPT-3': [(0.76, 0.04)],
    },
    'bank': {
        'Log. Reg.': [(0.55, 0.09), (0.66, 0.09), (0.75, 0.06), (0.81, 0.02), (0.84, 0.02), (0.86, 0.02), (0.88, 0.01), (0.89, 0.00), (0.91, 0.00)],
        'Log. Reg. ordinal': [(0.51, 0.02), (0.60, 0.12), (0.68, 0.09), (0.78, 0.04), (0.82, 0.01), (0.84, 0.03), (0.86, 0.01), (0.87, 0.00), (0.88, 0.00)],
        'LightGBM': [(0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.77, 0.03), (0.84, 0.03), (0.88, 0.01), (0.89, 0.00), (0.94, 0.00)],
        'LightGBM ordinal': [(0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.78, 0.03), (0.84, 0.02), (0.87, 0.01), (0.89, 0.00), (0.94, 0.00)],
        'XGBoost': [(0.50, 0.00), (0.56, 0.09), (0.68, 0.04), (0.76, 0.03), (0.83, 0.02), (0.85, 0.03), (0.88, 0.01), (0.90, 0.01), (0.94, 0.00)],
        'XGBoost ordinal': [(0.50, 0.00), (0.56, 0.09), (0.69, 0.05), (0.75, 0.04), (0.82, 0.02), (0.84, 0.03), (0.87, 0.01), (0.89, 0.00), (0.93, 0.00)],
        'SAINT': [(0.51, 0.10), (0.61, 0.11), (0.70, 0.04), (0.77, 0.03), (0.81, 0.03), (0.85, 0.02), (0.88, 0.01), (0.88, 0.01), (0.93, 0.00)],
        'TabNet': [(0.51, 0.06), (0.58, 0.05), (0.64, 0.10), (0.62, 0.04), (0.71, 0.06), (0.73, 0.03), (0.80, 0.04), (0.83, 0.03), (0.93, 0.00)],
        'NODE': [(0.52, 0.02), (0.55, 0.06), (0.64, 0.06), (0.73, 0.06), (0.78, 0.02), (0.83, 0.03), (0.85, 0.01), (0.86, 0.01), (0.76, 0.02)],
        'TabPFN': [(0.59, 0.14), (0.66, 0.08), (0.69, 0.02), (0.76, 0.03), (0.82, 0.03), (0.86, 0.02), (0.89, 0.00), (0.90, 0.00), (0.91, 0.00)],
        'TabPFN ordinal': [(0.57, 0.10), (0.67, 0.05), (0.71, 0.05), (0.78, 0.04), (0.83, 0.01), (0.86, 0.02), (0.87, 0.00), (0.88, 0.00), (0.89, 0.00)],
        'Text Template': [(0.63, 0.01), (0.59, 0.10), (0.64, 0.05), (0.65, 0.05), (0.64, 0.06), (0.69, 0.03), (0.82, 0.05), (0.87, 0.01), (0.88, 0.01)],
        'TabLLM': [(0.63, 0.01), (0.59, 0.10), (0.64, 0.05), (0.65, 0.05), (0.64, 0.06), (0.69, 0.03), (0.82, 0.05), (0.87, 0.01), (0.88, 0.01)],
        'Text GPT-3': [(0.63, 0.01), (0.61, 0.04), (0.62, 0.02), (0.63, 0.03), (0.64, 0.02), (0.66, 0.04), (0.76, 0.04), (0.81, 0.02), (0.82, 0.01)],
        'Text T0': [(0.54, 0.01), (0.56, 0.08), (0.60, 0.06), (0.59, 0.06), (0.60, 0.04), (0.62, 0.04), (0.67, 0.04), (0.79, 0.03), (0.85, 0.01)],
        'Table-To-Text': [(0.42, 0.01), (0.48, 0.07), (0.50, 0.05), (0.56, 0.03), (0.57, 0.04), (0.59, 0.05), (0.63, 0.03), (0.68, 0.02), (0.74, 0.01)],
        'List Template': [(0.60, 0.01), (0.59, 0.10), (0.66, 0.02), (0.65, 0.04), (0.66, 0.05), (0.74, 0.07), (0.85, 0.02), (0.87, 0.01), (0.87, 0.01)],
        'List Only Values': [(0.56, 0.01), (0.58, 0.09), (0.60, 0.04), (0.63, 0.03), (0.67, 0.03), (0.71, 0.05), (0.79, 0.03), (0.84, 0.01), (0.86, 0.01)],
        'List Perm. Names': [(0.64, 0.00), (0.55, 0.10), (0.62, 0.07), (0.63, 0.04), (0.63, 0.05), (0.68, 0.04), (0.82, 0.02), (0.86, 0.01), (0.88, 0.00)],
        'List Perm. Values': [(0.38, 0.01), (0.47, 0.11), (0.53, 0.06), (0.55, 0.07), (0.57, 0.05), (0.65, 0.04), (0.75, 0.07), (0.84, 0.01), (0.85, 0.01)],
        'TabLLM (T0 3B + Text)': [(0.61, 0.01), (0.60, 0.10), (0.65, 0.05), (0.64, 0.07), (0.65, 0.05), (0.70, 0.02), (0.77, 0.05), (0.88, 0.01), (0.89, 0.01)],
        'GPT-3': [(0.45, 0.01)],
    },
    'blood': {
        'Log. Reg.': [(0.54, 0.09), (0.59, 0.08), (0.72, 0.03), (0.70, 0.06), (0.74, 0.02), (0.76, 0.02), (0.76, 0.02), (0.76, 0.03), (0.76, 0.03)],
        'Log. Reg. ordinal': [(0.54, 0.09), (0.59, 0.08), (0.72, 0.03), (0.70, 0.06), (0.74, 0.02), (0.76, 0.02), (0.76, 0.02), (0.76, 0.03), (0.76, 0.03)],
        'LightGBM': [(0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.69, 0.04), (0.71, 0.05), (0.71, 0.07), (0.67, 0.05), (0.74, 0.04)],
        'LightGBM ordinal': [(0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.69, 0.04), (0.71, 0.05), (0.71, 0.07), (0.67, 0.05), (0.74, 0.04)],
        'XGBoost': [(0.50, 0.00), (0.58, 0.07), (0.66, 0.04), (0.67, 0.06), (0.68, 0.05), (0.71, 0.06), (0.70, 0.07), (0.67, 0.06), (0.71, 0.04)],
        'XGBoost ordinal': [(0.50, 0.00), (0.58, 0.07), (0.66, 0.04), (0.67, 0.06), (0.68, 0.05), (0.71, 0.06), (0.70, 0.07), (0.67, 0.06), (0.71, 0.04)],
        'SAINT': [(0.47, 0.12), (0.66, 0.08), (0.66, 0.03), (0.67, 0.06), (0.67, 0.05), (0.71, 0.03), (0.76, 0.05), (0.73, 0.02), (0.74, 0.03)],
        'TabNet': [(0.47, 0.09), (0.61, 0.06), (0.60, 0.09), (0.66, 0.06), (0.63, 0.06), (0.66, 0.04), (0.72, 0.06), (0.72, 0.02), (0.71, 0.03)],
        'NODE': [(0.49, 0.04), (0.60, 0.07), (0.62, 0.04), (0.67, 0.03), (0.71, 0.05), (0.76, 0.03), (0.74, 0.03), (0.76, 0.03), (0.74, 0.03)],
        'TabPFN': [(0.52, 0.08), (0.64, 0.04), (0.67, 0.01), (0.70, 0.04), (0.73, 0.04), (0.75, 0.04), (0.76, 0.04), (0.76, 0.03), (0.74, 0.03)],
        'TabPFN ordinal': [(0.52, 0.08), (0.64, 0.04), (0.67, 0.01), (0.70, 0.04), (0.73, 0.04), (0.75, 0.04), (0.76, 0.04), (0.76, 0.03), (0.74, 0.03)],
        'Text Template': [(0.61, 0.04), (0.58, 0.09), (0.66, 0.03), (0.66, 0.07), (0.68, 0.04), (0.68, 0.04), (0.68, 0.06), (0.70, 0.08), (0.68, 0.04)],
        'TabLLM': [(0.61, 0.04), (0.58, 0.09), (0.66, 0.03), (0.66, 0.07), (0.68, 0.04), (0.68, 0.04), (0.68, 0.06), (0.70, 0.08), (0.68, 0.04)],
        'Text GPT-3': [(0.63, 0.04), (0.61, 0.07), (0.65, 0.04), (0.63, 0.02), (0.64, 0.03), (0.62, 0.05), (0.67, 0.06), (0.68, 0.05), (0.66, 0.05)],
        'Text T0': [(0.49, 0.04), (0.51, 0.03), (0.59, 0.08), (0.59, 0.06), (0.64, 0.04), (0.65, 0.06), (0.66, 0.05), (0.68, 0.06), (0.66, 0.03)],
        'Table-To-Text': [(0.61, 0.04), (0.59, 0.04), (0.59, 0.03), (0.57, 0.03), (0.62, 0.07), (0.56, 0.07), (0.57, 0.07), (0.64, 0.07), (0.61, 0.05)],
        'List Template': [(0.56, 0.05), (0.54, 0.08), (0.64, 0.02), (0.64, 0.08), (0.67, 0.05), (0.66, 0.06), (0.67, 0.05), (0.70, 0.06), (0.67, 0.06)],
        'List Only Values': [(0.45, 0.05), (0.49, 0.07), (0.57, 0.03), (0.57, 0.06), (0.62, 0.06), (0.61, 0.04), (0.64, 0.04), (0.68, 0.07), (0.67, 0.05)],
        'List Perm. Names': [(0.52, 0.04), (0.49, 0.07), (0.62, 0.03), (0.62, 0.06), (0.65, 0.05), (0.65, 0.04), (0.68, 0.06), (0.72, 0.06), (0.68, 0.04)],
        'List Perm. Values': [(0.51, 0.03), (0.51, 0.06), (0.54, 0.04), (0.52, 0.07), (0.55, 0.03), (0.59, 0.06), (0.59, 0.02), (0.62, 0.06), (0.62, 0.05)],
        'TabLLM (T0 3B + Text)': [(0.42, 0.05), (0.47, 0.04), (0.62, 0.04), (0.62, 0.09), (0.65, 0.07), (0.67, 0.04), (0.69, 0.04), (0.71, 0.06), (0.67, 0.04)],
        'GPT-3': [(0.66, 0.04)],
    },
    'california': {
        'Log. Reg.': [(0.58, 0.11), (0.69, 0.13), (0.80, 0.06), (0.84, 0.03), (0.88, 0.01), (0.90, 0.00), (0.91, 0.00), (0.91, 0.00), (0.92, 0.00)],
        'Log. Reg. ordinal': [(0.58, 0.11), (0.69, 0.13), (0.80, 0.06), (0.84, 0.03), (0.88, 0.01), (0.90, 0.00), (0.91, 0.00), (0.91, 0.00), (0.92, 0.00)],
        'LightGBM': [(0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.81, 0.02), (0.87, 0.01), (0.90, 0.01), (0.92, 0.00), (0.97, 0.00)],
        'LightGBM ordinal': [(0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.81, 0.02), (0.87, 0.01), (0.90, 0.01), (0.92, 0.00), (0.97, 0.00)],
        'XGBoost': [(0.50, 0.00), (0.62, 0.10), (0.74, 0.03), (0.79, 0.04), (0.82, 0.04), (0.87, 0.01), (0.90, 0.01), (0.92, 0.01), (0.97, 0.00)],
        'XGBoost ordinal': [(0.50, 0.00), (0.62, 0.10), (0.74, 0.03), (0.79, 0.04), (0.82, 0.04), (0.87, 0.01), (0.90, 0.01), (0.92, 0.01), (0.97, 0.00)],
        'SAINT': [(0.59, 0.09), (0.64, 0.12), (0.73, 0.06), (0.76, 0.06), (0.81, 0.02), (0.84, 0.01), (0.88, 0.02), (0.91, 0.02), (0.96, 0.00)],
        'TabNet': [(0.50, 0.08), (0.57, 0.06), (0.67, 0.02), (0.69, 0.05), (0.72, 0.03), (0.79, 0.02), (0.84, 0.02), (0.87, 0.01), (0.96, 0.00)],
        'NODE': [(0.58, 0.06), (0.57, 0.07), (0.70, 0.05), (0.77, 0.03), (0.80, 0.01), (0.86, 0.02), (0.86, 0.02), (0.87, 0.01), (0.87, 0.01)],
        'TabPFN': [(0.63, 0.13), (0.63, 0.11), (0.80, 0.03), (0.85, 0.03), (0.89, 0.01), (0.91, 0.01), (0.92, 0.00), (0.93, 0.00), (0.94, 0.00)],
        'TabPFN ordinal': [(0.63, 0.13), (0.63, 0.11), (0.80, 0.03), (0.85, 0.03), (0.89, 0.01), (0.91, 0.01), (0.92, 0.00), (0.93, 0.00), (0.94, 0.00)],
        'Text Template': [(0.61, 0.01), (0.63, 0.05), (0.60, 0.07), (0.70, 0.08), (0.77, 0.08), (0.77, 0.04), (0.81, 0.02), (0.83, 0.01), (0.86, 0.02)],
        'TabLLM': [(0.61, 0.01), (0.63, 0.05), (0.60, 0.07), (0.70, 0.08), (0.77, 0.08), (0.77, 0.04), (0.81, 0.02), (0.83, 0.01), (0.86, 0.02)],
        'Text GPT-3': [(0.56, 0.00), (0.55, 0.03), (0.57, 0.05), (0.61, 0.06), (0.73, 0.05), (0.73, 0.04), (0.82, 0.01), (0.84, 0.01), (0.85, 0.01)],
        'Text T0': [(0.49, 0.01), (0.52, 0.02), (0.51, 0.02), (0.52, 0.02), (0.54, 0.04), (0.56, 0.04), (0.69, 0.02), (0.73, 0.03), (0.80, 0.02)],
        'Table-To-Text': [(0.49, 0.01), (0.50, 0.01), (0.51, 0.01), (0.52, 0.02), (0.57, 0.04), (0.58, 0.04), (0.74, 0.03), (0.79, 0.02), (0.82, 0.01)],
        'List Template': [(0.61, 0.01), (0.64, 0.05), (0.62, 0.06), (0.68, 0.07), (0.77, 0.07), (0.79, 0.02), (0.82, 0.02), (0.84, 0.01), (0.87, 0.01)],
        'List Only Values': [(0.58, 0.01), (0.57, 0.08), (0.55, 0.03), (0.65, 0.09), (0.74, 0.08), (0.77, 0.03), (0.83, 0.01), (0.84, 0.02), (0.86, 0.02)],
        'List Perm. Names': [(0.54, 0.01), (0.52, 0.03), (0.52, 0.04), (0.52, 0.03), (0.66, 0.06), (0.74, 0.01), (0.81, 0.02), (0.84, 0.02), (0.86, 0.02)],
        'List Perm. Values': [(0.47, 0.01), (0.48, 0.02), (0.50, 0.01), (0.52, 0.02), (0.57, 0.03), (0.64, 0.04), (0.71, 0.04), (0.76, 0.01), (0.78, 0.02)],
        'TabLLM (T0 3B + Text)': [(0.57, 0.01), (0.59, 0.03), (0.57, 0.04), (0.66, 0.07), (0.77, 0.06), (0.79, 0.02), (0.81, 0.01), (0.83, 0.01), (0.85, 0.01)],
        'GPT-3': [(0.56, 0.00)],
    },
    'credit-g': {
        'Log. Reg.': [(0.50, 0.08), (0.56, 0.06), (0.58, 0.08), (0.68, 0.08), (0.66, 0.07), (0.71, 0.06), (0.75, 0.04), (0.76, 0.02), (0.79, 0.03)],
        'Log. Reg. ordinal': [(0.56, 0.05), (0.54, 0.06), (0.55, 0.05), (0.61, 0.05), (0.68, 0.05), (0.66, 0.03), (0.68, 0.04), (0.71, 0.02), (0.72, 0.02)],
        'LightGBM': [(0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.61, 0.09), (0.68, 0.03), (0.72, 0.02), (0.75, 0.02), (0.78, 0.02)],
        'LightGBM ordinal': [(0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.68, 0.07), (0.66, 0.04), (0.72, 0.02), (0.75, 0.03), (0.76, 0.04)],
        'XGBoost': [(0.50, 0.00), (0.51, 0.07), (0.59, 0.05), (0.66, 0.03), (0.67, 0.06), (0.68, 0.02), (0.73, 0.02), (0.75, 0.03), (0.78, 0.04)],
        'XGBoost ordinal': [(0.50, 0.00), (0.54, 0.11), (0.57, 0.08), (0.64, 0.05), (0.66, 0.06), (0.68, 0.04), (0.74, 0.02), (0.76, 0.03), (0.76, 0.04)],
        'SAINT': [(0.56, 0.08), (0.53, 0.05), (0.60, 0.05), (0.66, 0.06), (0.66, 0.06), (0.68, 0.05), (0.72, 0.04), (0.73, 0.03), (0.77, 0.04)],
        'TabNet': [(0.48, 0.05), (0.52, 0.07), (0.49, 0.03), (0.52, 0.03), (0.56, 0.05), (0.60, 0.05), (0.61, 0.02), (0.66, 0.04), (0.64, 0.03)],
        'NODE': [(0.54, 0.09), (0.54, 0.10), (0.54, 0.09), (0.59, 0.07), (0.63, 0.04), (0.68, 0.02), (0.68, 0.05), (0.70, 0.02), (0.65, 0.03)],
        'TabPFN': [(0.58, 0.08), (0.59, 0.03), (0.64, 0.06), (0.69, 0.07), (0.70, 0.07), (0.72, 0.06), (0.75, 0.04), (0.75, 0.02), (0.75, 0.03)],
        'TabPFN ordinal': [(0.55, 0.08), (0.51, 0.07), (0.57, 0.06), (0.62, 0.03), (0.66, 0.05), (0.70, 0.02), (0.73, 0.01), (0.73, 0.03), (0.75, 0.04)],
        'Text Template': [(0.53, 0.05), (0.69, 0.04), (0.66, 0.04), (0.66, 0.05), (0.72, 0.06), (0.70, 0.07), (0.71, 0.07), (0.72, 0.03), (0.72, 0.02)],
        'TabLLM': [(0.53, 0.05), (0.69, 0.04), (0.66, 0.04), (0.66, 0.05), (0.72, 0.06), (0.70, 0.07), (0.71, 0.07), (0.72, 0.03), (0.72, 0.02)],
        'Text GPT-3': [(0.52, 0.04), (0.53, 0.04), (0.56, 0.03), (0.56, 0.05), (0.55, 0.05), (0.57, 0.08), (0.60, 0.06), (0.61, 0.04), (0.63, 0.05)],
        'Text T0': [(0.49, 0.02), (0.50, 0.06), (0.54, 0.06), (0.55, 0.04), (0.60, 0.06), (0.61, 0.02), (0.61, 0.02), (0.63, 0.03), (0.65, 0.02)],
        'Table-To-Text': [(0.50, 0.06), (0.65, 0.04), (0.60, 0.05), (0.60, 0.07), (0.65, 0.05), (0.67, 0.05), (0.65, 0.05), (0.68, 0.04), (0.64, 0.05)],
        'List Template': [(0.53, 0.05), (0.64, 0.04), (0.60, 0.06), (0.64, 0.05), (0.70, 0.05), (0.66, 0.08), (0.67, 0.03), (0.70, 0.03), (0.70, 0.04)],
        'List Only Values': [(0.66, 0.06), (0.71, 0.03), (0.67, 0.06), (0.69, 0.06), (0.72, 0.06), (0.69, 0.05), (0.69, 0.07), (0.70, 0.06), (0.68, 0.04)],
        'List Perm. Names': [(0.44, 0.01), (0.58, 0.09), (0.59, 0.08), (0.60, 0.07), (0.70, 0.06), (0.69, 0.06), (0.67, 0.05), (0.70, 0.05), (0.70, 0.03)],
        'List Perm. Values': [(0.50, 0.05), (0.55, 0.06), (0.56, 0.07), (0.58, 0.04), (0.64, 0.03), (0.66, 0.08), (0.67, 0.09), (0.68, 0.03), (0.69, 0.03)],
        'TabLLM (T0 3B + Text)': [(0.54, 0.03), (0.65, 0.05), (0.63, 0.05), (0.63, 0.03), (0.73, 0.04), (0.69, 0.05), (0.68, 0.06), (0.73, 0.05), (0.73, 0.03)],
        'GPT-3': [(0.47, 0.04)],
    },
    'jungle': {
        'Log. Reg.': [(0.62, 0.09), (0.69, 0.09), (0.68, 0.04), (0.76, 0.03), (0.79, 0.01), (0.79, 0.00), (0.80, 0.01), (0.80, 0.00), (0.81, 0.00)],
        'Log. Reg. ordinal': [(0.62, 0.09), (0.69, 0.09), (0.68, 0.04), (0.76, 0.03), (0.79, 0.01), (0.79, 0.00), (0.80, 0.01), (0.80, 0.00), (0.81, 0.00)],
        'LightGBM': [(0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.79, 0.02), (0.84, 0.02), (0.88, 0.01), (0.91, 0.00), (0.98, 0.00)],
        'LightGBM ordinal': [(0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.50, 0.00), (0.79, 0.02), (0.84, 0.02), (0.88, 0.01), (0.91, 0.00), (0.98, 0.00)],
        'XGBoost': [(0.50, 0.00), (0.58, 0.07), (0.72, 0.05), (0.78, 0.03), (0.81, 0.02), (0.84, 0.02), (0.87, 0.01), (0.91, 0.01), (0.98, 0.00)],
        'XGBoost ordinal': [(0.50, 0.00), (0.58, 0.07), (0.72, 0.05), (0.78, 0.03), (0.81, 0.02), (0.84, 0.02), (0.87, 0.01), (0.91, 0.01), (0.98, 0.00)],
        'SAINT': [(0.64, 0.05), (0.69, 0.06), (0.72, 0.05), (0.79, 0.02), (0.81, 0.01), (0.83, 0.01), (0.88, 0.01), (0.90, 0.00), (1.00, 0.00)],
        'TabNet': [(0.53, 0.09), (0.60, 0.05), (0.62, 0.03), (0.69, 0.04), (0.73, 0.04), (0.75, 0.02), (0.79, 0.02), (0.84, 0.01), (0.99, 0.00)],
        'NODE': [(0.60, 0.01), (0.71, 0.03), (0.68, 0.04), (0.74, 0.02), (0.75, 0.04), (0.78, 0.01), (0.79, 0.01), (0.80, 0.00), (0.81, 0.00)],
        'TabPFN': [(0.65, 0.08), (0.72, 0.04), (0.71, 0.07), (0.78, 0.02), (0.81, 0.01), (0.84, 0.01), (0.88, 0.01), (0.91, 0.00), (0.93, 0.00)],
        'TabPFN ordinal': [(0.65, 0.08), (0.72, 0.04), (0.71, 0.07), (0.78, 0.02), (0.81, 0.01), (0.84, 0.01), (0.88, 0.01), (0.91, 0.00), (0.93, 0.00)],
        'Text Template': [(0.60, 0.00), (0.64, 0.01), (0.64, 0.02), (0.65, 0.03), (0.71, 0.02), (0.78, 0.02), (0.81, 0.02), (0.84, 0.01), (0.89, 0.01)],
        'TabLLM': [(0.60, 0.00), (0.64, 0.01), (0.64, 0.02), (0.65, 0.03), (0.71, 0.02), (0.78, 0.02), (0.81, 0.02), (0.84, 0.01), (0.89, 0.01)],
        'Text GPT-3': [(0.56, 0.01), (0.58, 0.02), (0.55, 0.02), (0.60, 0.06), (0.68, 0.03), (0.74, 0.03), (0.77, 0.01), (0.81, 0.01), (0.85, 0.01)],
        'Text T0': [(0.63, 0.00), (0.63, 0.04), (0.64, 0.05), (0.62, 0.06), (0.70, 0.01), (0.71, 0.03), (0.74, 0.02), (0.78, 0.02), (0.82, 0.01)],
        'Table-To-Text': [(0.51, 0.01), (0.60, 0.02), (0.60, 0.04), (0.63, 0.05), (0.69, 0.03), (0.75, 0.01), (0.78, 0.03), (0.82, 0.01), (0.85, 0.01)],
        'List Template': [(0.63, 0.00), (0.65, 0.01), (0.66, 0.03), (0.66, 0.04), (0.71, 0.03), (0.78, 0.02), (0.81, 0.03), (0.84, 0.01), (0.88, 0.01)],
        'List Only Values': [(0.58, 0.00), (0.60, 0.03), (0.62, 0.03), (0.63, 0.02), (0.65, 0.04), (0.73, 0.01), (0.76, 0.02), (0.82, 0.02), (0.88, 0.01)],
        'List Perm. Names': [(0.40, 0.00), (0.53, 0.06), (0.55, 0.05), (0.63, 0.10), (0.72, 0.03), (0.79, 0.02), (0.80, 0.03), (0.84, 0.02), (0.89, 0.01)],
        'List Perm. Values': [(0.48, 0.00), (0.50, 0.02), (0.52, 0.03), (0.53, 0.03), (0.55, 0.01), (0.59, 0.02), (0.63, 0.01), (0.72, 0.02), (0.75, 0.01)],
        'TabLLM (T0 3B + Text)': [(0.54, 0.00), (0.63, 0.02), (0.64, 0.04), (0.67, 0.03), (0.72, 0.03), (0.77, 0.02), (0.80, 0.02), (0.83, 0.01), (0.87, 0.01)],
        'GPT-3': [(0.64, 0.00)],
    },
}

non_zero_shot = ['SAINT', 'NODE', 'TabNet', 'Log. Reg.', 'LightGBM', 'XGBoost',
                 'TabNet ordinal', 'Log. Reg. ordinal', 'LightGBM ordinal', 'XGBoost ordinal',
                 'TabPFN', 'TabPFN ordinal']
zero_shot = ['GPT-3']

test_set_sizes_per_dataset = {
    'income': 1,  # 9769,
    'car': 1,  # 346,
    'heart': 1,  # 184,
    'diabetes': 1,  # 154
    'bank': 1,  # 154
    'blood': 1,  # 154
    'california': 1,  # 154
    'credit-g': 1,  # 154
    'jungle': 1,  # 154
}


def create_performance_graph(args):
    orig_shots = [0, 4, 8, 16, 32, 64, 128, 256, 512]
    total_samples = sum(test_set_sizes_per_dataset.values())


    # Determine means and sd for each dataset
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    # plt.rcParams["font.family"] = "Times"
    # included_performance_scores = ['Log. Reg.', 'LightGBM', 'XGBoost', 'SAINT', 'TabNet', 'NODE', 'TabPFN', 'GPT-3', 'TabLLM']
    # markers = ['x', '*', '8', 'd', '^', 'p', 'P', 'X', 'o']
    # colors = ['C1', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C2', 'C0']
    included_performance_scores = ['List Template', 'Text Template', 'Table-To-Text', 'Text T0', 'Text GPT-3', 'List Only Values', 'List Perm. Names', 'List Perm. Values']
    markers = ['x', 'o', '*', '8', 'd', '^', 'p', 'P', 'X']
    colors = ['C1', 'C0', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C2']


    # Create list of performance results for each setting
    for k, setting in enumerate(included_performance_scores):
        if setting in zero_shot:
            shots = [0]
        elif setting in non_zero_shot:
            shots = orig_shots[1:]
        else:
            shots = orig_shots
        weighted_mean = [0.] * len(shots)
        weighted_variance = [0.] * len(shots)

        for dataset in test_set_sizes_per_dataset.keys():
            for i, shot in enumerate(shots):
                weighted_mean[i] += test_set_sizes_per_dataset[dataset] * performance_per_dataset_setting[dataset][setting][i][0]
                weighted_variance[i] += test_set_sizes_per_dataset[dataset] * (performance_per_dataset_setting[dataset][setting][i][1] ** 2)

        means = np.array(weighted_mean) / total_samples
        stds = np.sqrt(np.array(weighted_variance)) / total_samples

        ax.plot(range(setting in non_zero_shot, 1 if setting in zero_shot else len(orig_shots)), means, marker=markers[k], label=setting, color=colors[k], markersize=12 if setting == 'GPT-3' else None)
        ax.fill_between(range(setting in non_zero_shot, 1 if setting in zero_shot else len(orig_shots)), (means - stds), (means + stds), alpha=.1, color=colors[k])
        ax.set_ylabel('Average AUC (SD) across tabular datasets')
        ax.set_xlabel('Number of labeled training examples (shots)')
        ax.legend(loc='lower right')._legend_box.align='right'
        # legend._legend_box.align = 'right'
        # ax.set_xscale('symlog', base=2)
        ax.set_xlim(xmin=0, xmax=len(orig_shots) - 1)
        ax.set_xticks(range(0, len(orig_shots)), [str(s) for s in orig_shots])
        ax.set_ylim(ymin=0.475, ymax=0.88)

    plt.tight_layout()
    # plt.show()
    # plt.savefig('figure_baselines.pdf', dpi=600)
    plt.savefig('figure_serialization.pdf', dpi=600)
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    create_performance_graph(args)


