@echo off
rem This batch file runs 9 Python scripts in parallel, each using 2 threads and saving outputs in D:\2024_PHD-DATA_CFP_PROCESSING_v2.

set PYTHON_PATH=C:\Users\jcp\AppData\Local\Programs\Python\Python39\python.exe
set SCRIPT_PATH=C:\Users\jcp\PycharmProjects\RETFound_MAE_JCP\preprocessing_imgs\per_dir_v2\img_stats_parallel_v2.py

rem Affinity masks for optimized CPU usage (leaving the first two threads unused)
start "Processing 101-120k" /affinity 0xC cmd /k ""%PYTHON_PATH%" "%SCRIPT_PATH%" "D:\2024_PHD-DATA_CFP-imgs_cropped\101-120k_240129-1215""
start "Processing 121-140k" /affinity 0x30 cmd /k ""%PYTHON_PATH%" "%SCRIPT_PATH%" "D:\2024_PHD-DATA_CFP-imgs_cropped\121-140k_240129-2346""
start "Processing 141-160k" /affinity 0xC0 cmd /k ""%PYTHON_PATH%" "%SCRIPT_PATH%" "D:\2024_PHD-DATA_CFP-imgs_cropped\141-160k_240130-0008""
start "Processing 161-181k" /affinity 0x300 cmd /k ""%PYTHON_PATH%" "%SCRIPT_PATH%" "D:\2024_PHD-DATA_CFP-imgs_cropped\161-181k_240130-0031""
start "Processing 1-20k" /affinity 0xC00 cmd /k ""%PYTHON_PATH%" "%SCRIPT_PATH%" "D:\2024_PHD-DATA_CFP-imgs_cropped\1-20k_240128-2147""
start "Processing 21-40k" /affinity 0x3000 cmd /k ""%PYTHON_PATH%" "%SCRIPT_PATH%" "D:\2024_PHD-DATA_CFP-imgs_cropped\21-40k_240128-2211""
start "Processing 41-60k" /affinity 0xC000 cmd /k ""%PYTHON_PATH%" "%SCRIPT_PATH%" "D:\2024_PHD-DATA_CFP-imgs_cropped\41-60k_240128-2240""
start "Processing 61-80k" /affinity 0x30000 cmd /k ""%PYTHON_PATH%" "%SCRIPT_PATH%" "D:\2024_PHD-DATA_CFP-imgs_cropped\61-80k_240128-2307""
start "Processing 81-100k" /affinity 0xC0000 cmd /k ""%PYTHON_PATH%" "%SCRIPT_PATH%" "D:\2024_PHD-DATA_CFP-imgs_cropped\81-100k_240128-2346""

echo All processes have completed.
pause
